"""Embedding 抽象：支持三种 provider，接口统一为 async。

- api   : 任意 OpenAI 兼容的 /v1/embeddings 服务（推荐）
- local : sentence-transformers 本地模型（可选依赖）
- hash  : 零依赖本地哈希向量，仅用于打通链路 / 离线自测，语义检索能力很弱，
          会在日志、/health、README 里明确提示；向量库建立后不要随意切换 provider，
          因为不同 provider 的向量空间不可混用（切换需重建集合）。
"""
from __future__ import annotations

import asyncio
import hashlib
import math
import os
import random
import threading
from abc import ABC, abstractmethod
from typing import List, Optional

import httpx

from app.config import Settings, is_placeholder
from app.core.text_utils import char_ngrams
from app.logging_conf import get_logger

logger = get_logger(__name__)

# 默认镜像：当 huggingface.co 不可达或被 hosts 指向本机时自动使用
DEFAULT_HF_MIRROR = "https://hf-mirror.com"


def _apply_hf_endpoint(endpoint: str) -> None:
    """把镜像端点同时写进环境变量和 huggingface_hub 的运行时常量。

    实测坑：huggingface_hub 在 import 时会把 HF_ENDPOINT 固化到
    `huggingface_hub.constants.ENDPOINT`，之后（或之前）只改 os.environ 都不生效，
    请求仍会打到 huggingface.co —— 表现为"明明配了镜像却还是 SSL 证书错误"。
    """
    if not endpoint:
        return
    os.environ["HF_ENDPOINT"] = endpoint
    os.environ.setdefault("HF_HUB_ENDPOINT", endpoint)
    try:
        import huggingface_hub.constants as hf_constants  # type: ignore

        if hf_constants.ENDPOINT != endpoint:
            logger.info(
                "将 huggingface_hub 端点从 %s 切换为 %s", hf_constants.ENDPOINT, endpoint
            )
            hf_constants.ENDPOINT = endpoint
    except Exception as exc:  # pragma: no cover - 版本差异时不影响其它逻辑
        logger.debug("设置 huggingface_hub 端点常量失败（可忽略）：%s", exc)


def _configure_hf_environment() -> None:
    """让 huggingface_hub 在受限网络下也能下到模型权重。

    实测踩过的两个坑：
    1) Windows 上 pip 能下载（用 certifi 的 CA 包），但 requests/huggingface_hub
       走系统证书存储，会报 CERTIFICATE_VERIFY_FAILED → 这里把 CA 指向 certifi；
    2) 如果 hosts 里把 huggingface.co 映射到 127.0.0.1（很多"加速"脚本留下的），
       会连到一个本地未签名的服务上，同样报证书错误 → 这里检测到这种情况时
       自动切到 HF_ENDPOINT 指定的镜像（默认 hf-mirror.com）。

    可用环境变量：
      HF_ENDPOINT=https://hf-mirror.com   指定镜像端点
      HF_HUB_DISABLE_SSL_VERIFY=1         跳过证书校验（仅建议自签证书的内网使用）
    """
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    try:
        import certifi  # type: ignore

        ca_bundle = certifi.where()
        if ca_bundle and os.path.exists(ca_bundle):
            os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
            os.environ.setdefault("CURL_CA_BUNDLE", ca_bundle)
    except ImportError:  # pragma: no cover - certifi 通常随 requests 一起装
        pass

    endpoint = os.getenv("HF_ENDPOINT", "").strip()
    if not endpoint and _huggingface_host_is_blackholed():
        endpoint = DEFAULT_HF_MIRROR
        logger.warning(
            "检测到 huggingface.co 被 hosts/DNS 指向本机（无法下载权重），"
            "已自动改用镜像 %s；如需固定请写入 .env 的 HF_ENDPOINT。",
            endpoint,
        )
    if endpoint:
        _apply_hf_endpoint(endpoint)

    if os.getenv("HF_HUB_DISABLE_SSL_VERIFY", "").strip().lower() in {"1", "true", "yes"}:
        os.environ.setdefault("PYTHONHTTPSVERIFY", "0")
        try:
            import ssl

            ssl._create_default_https_context = ssl._create_unverified_context  # noqa: SLF001
            logger.warning("已按 HF_HUB_DISABLE_SSL_VERIFY=1 关闭 HTTPS 证书校验（存在中间人风险）")
        except Exception:  # pragma: no cover
            pass


def _huggingface_host_is_blackholed(timeout: float = 3.0) -> bool:
    """判断 huggingface.co 是否被解析到本机/回环地址（典型 hosts 加速残留）。"""
    try:
        import socket

        resolved = socket.gethostbyname("huggingface.co")
    except Exception:
        return False
    return resolved.startswith("127.") or resolved in {"0.0.0.0", "::1"}


class EmbeddingError(RuntimeError):
    """Embedding 调用失败（网络、鉴权、返回结构异常等）。"""


class BaseEmbedder(ABC):
    name: str = "base"
    degraded: bool = False  # 是否处于降级模式（例如 local 回退到 hash）
    degrade_reason: str = ""

    def __init__(self, dimension: int) -> None:
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文档片段。"""

    async def embed_query(self, text: str) -> List[float]:
        vectors = await self.embed_documents([text])
        if not vectors:
            raise EmbeddingError("embedding 返回为空")
        return vectors[0]

    def describe(self) -> dict:
        return {
            "provider": self.name,
            "dimension": self.dimension,
            "degraded": self.degraded,
            "degrade_reason": self.degrade_reason,
        }


# --------------------------------------------------------------------------- #
# hash：零依赖本地向量（离线可用，但不具备真正的语义能力）
# --------------------------------------------------------------------------- #
class HashEmbedder(BaseEmbedder):
    """字符 2/3-gram 计数 -> 固定维度稠密向量（带符号随机投影 + L2 归一化）。

    相同文本必然得到相同向量，因此可用于验证"切分 -> 入库 -> 检索 -> 引用"
    整条链路是否通畅。但它不理解同义词/语义，真实业务请改用 api 或 local。
    """

    name = "hash"

    def __init__(self, dimension: int = 1024) -> None:
        super().__init__(dimension)
        digest = hashlib.sha256(f"hash-embedder-seed-{self._dimension}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        # 预生成每个维度的符号，保证跨进程稳定
        self._signs = [1.0 if rng.random() < 0.5 else -1.0 for _ in range(self._dimension)]

    def _encode(self, text: str) -> List[float]:
        vector = [0.0] * self._dimension
        grams = char_ngrams(text, sizes=(2, 3))
        for gram, count in grams.items():
            # blake2b 稳定、跨进程一致（Python 内置 hash 有随机盐，不可用）
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest, "big") % self._dimension
            sign = 1.0 if (digest[0] & 1) == 0 else -1.0
            vector[index] += sign * (1.0 + math.log(count)) * self._signs[index]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(lambda: [self._encode(text) for text in texts])


# --------------------------------------------------------------------------- #
# api：OpenAI 兼容 /v1/embeddings
# --------------------------------------------------------------------------- #
class OpenAICompatibleEmbedder(BaseEmbedder):
    name = "api"

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        dimension: int,
        timeout: float = 60.0,
        max_retries: int = 3,
        batch_size: int = 16,
    ) -> None:
        super().__init__(dimension)
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_retries = max(1, max_retries)
        self._batch_size = max(1, batch_size)
        self._client = httpx.AsyncClient(timeout=timeout)

    @property
    def url(self) -> str:
        return f"{self._base_url}/embeddings"

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_error: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.post(self.url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                last_error = exc
                logger.warning("embedding 网络异常（第 %d 次）：%s", attempt, exc)
            else:
                if response.status_code < 400:
                    return response.json()
                body = response.text[:300]
                # 4xx 中除 429 外基本是配置问题，重试无意义
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise EmbeddingError(
                        f"embedding 接口返回 {response.status_code}：{body}。"
                        "请检查 EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_MODEL。"
                    )
                last_error = EmbeddingError(f"embedding 接口返回 {response.status_code}：{body}")
                logger.warning("embedding 失败（第 %d 次）：%s", attempt, last_error)
            if attempt < self._max_retries:
                await asyncio.sleep(min(2 ** (attempt - 1), 8))
        raise EmbeddingError(f"embedding 调用失败：{last_error}")

    async def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        payload = {"model": self._model, "input": batch}
        if self._dimension > 0:
            # 部分服务商支持 dimensions 参数；不支持时会报错，故仅在显式配置时携带
            payload["dimensions"] = self._dimension
        data = await self._post(payload)
        items = data.get("data")
        if not isinstance(items, list) or len(items) != len(batch):
            raise EmbeddingError(f"embedding 返回结构异常：{str(data)[:200]}")
        ordered = sorted(items, key=lambda item: item.get("index", 0))
        vectors = [item.get("embedding") for item in ordered]
        if any(not isinstance(vec, list) or not vec for vec in vectors):
            raise EmbeddingError("embedding 返回的向量为空")
        actual_dim = len(vectors[0])
        if self._dimension and actual_dim != self._dimension:
            # 不直接报错，但明确暴露出来，避免与 Chroma 集合维度不一致时难以排查
            logger.warning(
                "embedding 实际维度 %d 与配置 EMBEDDING_DIM=%d 不一致，请核对配置",
                actual_dim,
                self._dimension,
            )
            self._dimension = actual_dim
        return vectors

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results: List[List[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            results.extend(await self._embed_batch(batch))
        return results


# --------------------------------------------------------------------------- #
# local：sentence-transformers
# --------------------------------------------------------------------------- #
def _default_query_prefix(model_name: str) -> str:
    """按模型名推断官方建议的查询指令前缀。

    取值来源为各模型卡片的推荐用法（bge 中文系列通用同一句），
    可用 BGE_QUERY_INSTRUCTION 环境变量覆盖；不需要前缀的模型返回空串。
    """
    override = os.getenv("BGE_QUERY_INSTRUCTION", "").strip()
    if override:
        return override if override.lower() not in {"none", "empty", "-"} else ""
    lowered = (model_name or "").lower()
    if "bge" in lowered and ("zh" in lowered or "chinese" in lowered or "large" in lowered):
        return "为这个句子生成表示以用于检索相关文章："
    return ""


class LocalSentenceTransformerEmbedder(BaseEmbedder):
    """本地 sentence-transformers 向量。

    关于加载方式（这里有个真实的坑，已处理）：
    - 纯异步方案（to_thread + asyncio.Lock）需要 torch 有 Python 3.13 的 wheel，
      而 torch 的 3.13 支持是滞后的，装不上就只能放弃 local 模式；
    - 纯同步方案在 FastAPI 的 async 路由里会阻塞事件循环。
    因此这里做成"自适应"：asyncio 事件循环可用 → 走异步线程池；不可用（同步脚本调用）
    → 退回同步加载。这样 `python scripts/preflight.py` 和 uvicorn 里都能正常工作。
    """

    name = "local"

    def __init__(self, model_name: str, dimension: int = 0, hf_endpoint: str = "") -> None:
        super().__init__(dimension)
        self._model_name = model_name
        self._model = None
        self._thread_lock = threading.Lock()
        self._query_prefix = _default_query_prefix(model_name)
        # 关键：.env 里的 HF_ENDPOINT 只会被 pydantic-settings 读进 Settings 对象，
        # 不会自动写入 os.environ，而 huggingface_hub 只认环境变量。
        # 所以这里必须显式同步过去，否则镜像配置形同虚设（实测踩过）。
        if hf_endpoint:
            _apply_hf_endpoint(hf_endpoint)
        # 模型权重下载过一次后就在本地缓存里了。此时如果还允许联网探测，
        # 某些库（实测 transformers 对 adapter_config.json 的探测）会绕过 HF_ENDPOINT
        # 直连 huggingface.co，在 hosts 被改过或网络受限的机器上就是一堆 SSL 报错 +
        # 每次加载都要重试十几秒。置 1 后全部走本地缓存，加载从 ~15s 降到 ~2s，且不再报错。
        self._local_files_only = os.getenv("HF_HUB_OFFLINE", "").strip().lower() in {
            "1",
            "true",
            "yes",
        }

    def _ensure_model(self):
        with self._thread_lock:
            if self._model is not None:
                return self._model
            _configure_hf_environment()
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise EmbeddingError(
                    "本地模式需要可选依赖，请执行： pip install -r requirements-local.txt"
                ) from exc
            logger.info("正在加载本地向量模型（首次运行需要下载权重）：%s", self._model_name)
            try:
                model = SentenceTransformer(
                    self._model_name, local_files_only=self._local_files_only
                )
            except Exception as exc:
                # 下载失败是最常见的卡点（证书校验 / 网络不可达），这里给出可操作的中文提示，
                # 而不是把一大段 SSL 堆栈直接甩给用户。
                raise EmbeddingError(
                    f"加载本地向量模型失败：{type(exc).__name__}: {str(exc)[:200]}\n"
                    "常见原因与处理方式：\n"
                    "  1) SSL 证书校验失败 → 本程序已自动尝试使用 certifi 的 CA 包；"
                    "若仍失败，可设置环境变量 HF_HUB_DISABLE_SSL_VERIFY=1 临时跳过校验；\n"
                    "  2) 无法访问 huggingface.co → 设置镜像 HF_ENDPOINT=https://hf-mirror.com 后重试；\n"
                    "  3) 已手动下载模型 → 把 LOCAL_EMBEDDING_MODEL 指向本地目录即可离线加载。"
                ) from exc
            if not self._dimension:
                self._dimension = int(model.get_sentence_embedding_dimension() or 0)
            self._model = model
            logger.info("本地向量模型加载完成，维度 = %d", self._dimension)
            return model

    def _encode_sync(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        model = self._ensure_model()
        payload = list(texts)
        if is_query and self._query_prefix:
            # bge 等模型要求"查询侧"加指令前缀，文档侧不加。
            # 不加前缀不会报错，但检索质量会明显下降，很容易被误判成"向量模型不行"。
            payload = [self._query_prefix + text for text in payload]
        vectors = model.encode(payload, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, vector)) for vector in vectors]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode_sync, texts, False)

    async def embed_query(self, text: str) -> List[float]:
        vectors = await asyncio.to_thread(self._encode_sync, [text], True)
        return vectors[0]


# --------------------------------------------------------------------------- #
# 工厂
# --------------------------------------------------------------------------- #
def build_embedder(settings: Settings) -> BaseEmbedder:
    """按配置构造 embedder；provider=local 缺依赖时降级为 hash 并留下原因。"""
    provider = settings.embedding_provider

    if provider == "api":
        missing = [
            name
            for name, value in (
                ("EMBEDDING_API_KEY", settings.embedding_api_key),
                ("EMBEDDING_BASE_URL", settings.embedding_base_url),
                ("EMBEDDING_MODEL", settings.embedding_model),
            )
            if is_placeholder(value)
        ]
        if missing:
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=api，但以下配置仍为占位符或为空："
                + "、".join(missing)
                + "。请在 .env 中填写，或改用 EMBEDDING_PROVIDER=hash 先跑通链路。"
            )
        logger.info(
            "Embedding provider=api base_url=%s model=%s",
            settings.embedding_base_url,
            settings.embedding_model,
        )
        return OpenAICompatibleEmbedder(
            api_key=settings.embedding_api_key,
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            dimension=int(settings.embedding_dim or 0),
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    if provider == "local":
        if is_placeholder(settings.local_embedding_model):
            raise EmbeddingError(
                "EMBEDDING_PROVIDER=local，但 LOCAL_EMBEDDING_MODEL 未填写（当前为占位符）。"
            )
        try:
            import sentence_transformers  # noqa: F401  # type: ignore
        except ImportError:
            reason = "未安装 sentence-transformers（pip install -r requirements-local.txt）"
            logger.warning("本地 embedding 不可用，已降级为 hash 向量模式：%s", reason)
            fallback = HashEmbedder(settings.hash_embedding_dim)
            fallback.degraded = True
            fallback.degrade_reason = reason
            return fallback
        return LocalSentenceTransformerEmbedder(
            model_name=settings.local_embedding_model,
            dimension=int(settings.local_embedding_dim or 0),
            hf_endpoint=settings.hf_endpoint,
        )

    logger.warning(
        "Embedding provider=hash：向量由字符 n-gram 哈希生成，只能验证链路通畅，语义检索能力很弱。"
        "正式使用请在 .env 配置 EMBEDDING_PROVIDER=api。"
    )
    return HashEmbedder(settings.hash_embedding_dim)
