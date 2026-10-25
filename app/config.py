"""项目配置：全部来自环境变量 / .env，代码里不硬编码任何密钥。

优先级：真实环境变量 > .env 文件 > 下面的默认值。
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# 项目根目录（本文件位于 <root>/app/config.py）
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载 .env（不存在也不报错）
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _env_str(key: str, default: str = "") -> str:
    value = os.getenv(key)
    return default if value is None else value.strip()


def _env_int(key: str, default: int) -> int:
    raw = _env_str(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = _env_str(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_float_or_none(key: str) -> Optional[float]:
    """读一个可选的浮点环境变量；未配置或非法时返回 None（区别于「配了 0」）。"""
    raw = _env_str(key)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _env_bool(key: str, default: bool) -> bool:
    raw = _env_str(key).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def _env_optional_int(key: str) -> Optional[int]:
    raw = _env_str(key)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _env_list(key: str, default: Optional[List[str]] = None) -> List[str]:
    raw = _env_str(key)
    if not raw:
        return list(default or [])
    return [item.strip() for item in raw.split(",") if item.strip()]


def is_placeholder(value: str) -> bool:
    """判断一个配置值是否仍是占位符（[待补充] / 空 / 常见示例值）。"""
    if value is None:
        return True
    text = value.strip()
    if not text:
        return True
    return text.startswith("[待补充") or text in {"your-api-key", "sk-xxx", "changeme"}


def _resolve_path(raw: str, default: str) -> Path:
    """相对路径一律相对项目根目录解析，便于本地与容器内行为一致。"""
    text = raw.strip() or default
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


# 合法取值（写错时会被纠正为默认值并在 /health 的 config_warnings 里提示，而不是直接崩）
VALID_EMBEDDING_PROVIDERS = ("api", "local", "hash")
VALID_RETRIEVAL_MODES = ("vector", "keyword", "hybrid")


class Settings(BaseModel):
    # ---- LLM ----
    llm_api_key: str = Field(default_factory=lambda: _env_str("LLM_API_KEY"))
    llm_base_url: str = Field(
        default_factory=lambda: _env_str("LLM_BASE_URL", "https://api.deepseek.com/v1")
    )
    llm_model: str = Field(default_factory=lambda: _env_str("LLM_MODEL", "deepseek-chat"))
    llm_timeout_seconds: float = Field(
        default_factory=lambda: _env_float("LLM_TIMEOUT_SECONDS", 60.0)
    )
    llm_max_retries: int = Field(default_factory=lambda: _env_int("LLM_MAX_RETRIES", 3))
    llm_temperature: float = Field(
        default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0),
        description=(
            "LLM 采样温度。事实型 RAG 默认 0：实测（12.26，3 轮×20 条）把"
            " faithfulness 的极差从 0.13 收窄到 0.018，评测与线上回答都更稳定。"
        ),
    )
    llm_max_tokens: int = Field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 1024))

    # ---- Embedding ----
    # 注意：这两个字段故意用 str 而不是 Literal。用 Literal 时，如果 .env 里把值写错，
    # pydantic 会在导入期直接抛 ValidationError，用户只能看到一个启动崩溃；
    # 这里改成"接受任意字符串 + normalize() 纠正 + config_warnings 提示"，排错体验更好。
    embedding_provider: str = Field(
        default_factory=lambda: (_env_str("EMBEDDING_PROVIDER", "hash").lower() or "hash")
    )
    embedding_api_key: str = Field(default_factory=lambda: _env_str("EMBEDDING_API_KEY"))
    embedding_base_url: str = Field(default_factory=lambda: _env_str("EMBEDDING_BASE_URL"))
    embedding_model: str = Field(default_factory=lambda: _env_str("EMBEDDING_MODEL"))
    embedding_dim: Optional[int] = Field(default_factory=lambda: _env_optional_int("EMBEDDING_DIM"))
    local_embedding_model: str = Field(default_factory=lambda: _env_str("LOCAL_EMBEDDING_MODEL"))
    local_embedding_dim: Optional[int] = Field(
        default_factory=lambda: _env_optional_int("LOCAL_EMBEDDING_DIM")
    )
    hash_embedding_dim: int = Field(default_factory=lambda: _env_int("HASH_EMBEDDING_DIM", 1024))

    # 本地向量模型下载端点。留空时由 app/core/embedding.py 自动判断：
    # 若 huggingface.co 被 hosts/DNS 指向本机，会自动切到 hf-mirror.com。
    hf_endpoint: str = Field(default_factory=lambda: _env_str("HF_ENDPOINT"))

    # ---- 路径 ----
    documents_dir_raw: str = Field(
        default_factory=lambda: _env_str("DOCUMENTS_DIR", "./data/documents")
    )
    chroma_dir_raw: str = Field(default_factory=lambda: _env_str("CHROMA_DIR", "./data/chroma"))
    upload_dir_raw: str = Field(default_factory=lambda: _env_str("UPLOAD_DIR", "./data/uploads"))
    registry_path_raw: str = Field(
        default_factory=lambda: _env_str("REGISTRY_PATH", "./data/registry.json")
    )
    bm25_index_path_raw: str = Field(
        default_factory=lambda: _env_str("BM25_INDEX_PATH", "./data/bm25_index.json")
    )
    chroma_collection: str = Field(
        default_factory=lambda: _env_str("CHROMA_COLLECTION", "auto_after_sales")
    )
    guard_db_path_raw: str = Field(
        default_factory=lambda: _env_str("GUARD_DB_PATH", "./data/claims.db")
    )
    # ---- Agent（LangGraph）----
    agent_kb_path_raw: str = Field(
        default_factory=lambda: _env_str("AGENT_KB_PATH", "./data/agent_kb.db")
    )
    agent_checkpoint_path_raw: str = Field(
        default_factory=lambda: _env_str("AGENT_CHECKPOINT_PATH", "./data/agent_checkpoints.db")
    )
    agent_max_iterations: int = Field(
        default_factory=lambda: _env_int("AGENT_MAX_ITERATIONS", 6)
    )

    # ---- 切分 ----
    chunk_size: int = Field(default_factory=lambda: _env_int("CHUNK_SIZE", 600))
    chunk_overlap: int = Field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 80))
    chunk_min_chars: int = Field(default_factory=lambda: _env_int("CHUNK_MIN_CHARS", 80))

    # ---- 多轮会话（L4）----
    # Agent 链路早就有 session_id + checkpoint，主 /chat 链路的 session_id
    # 一直是预留字段。这里补上：历史轮数上限 + 是否做查询改写。
    session_max_turns: int = Field(
        default_factory=lambda: _env_int("SESSION_MAX_TURNS", 6),
        description="每个会话最多保留的问答轮数，旧轮次直接丢弃（不摘要）",
    )
    session_max_sessions: int = Field(
        default_factory=lambda: _env_int("SESSION_MAX_SESSIONS", 512),
        description=(
            "最多同时保留多少个会话的历史，超过后淘汰最早创建的会话。"
            "会话历史是常驻进程里的内存结构，没有上限就是慢泄漏。"
        ),
    )

    # ---- 查询缓存（L5）----
    # 缓存的正确性边界见 app/services/query_cache.py：
    # 键里带索引版本（入库即失效）；多轮与 debug 请求不缓存。
    query_cache_enabled: bool = Field(
        default_factory=lambda: _env_bool("QUERY_CACHE_ENABLED", True),
        description="是否启用查询缓存（同样的问题直接返回上次回答）",
    )
    query_cache_ttl_seconds: int = Field(
        default_factory=lambda: _env_int("QUERY_CACHE_TTL_SECONDS", 3600),
        description="缓存条目的存活时间（秒）。索引变更会立即失效，TTL 只是兜底",
    )
    query_cache_max_entries: int = Field(
        default_factory=lambda: _env_int("QUERY_CACHE_MAX_ENTRIES", 256),
        description="缓存最大条目数（LRU 淘汰）",
    )

    # ---- 成本估算（L5）：价格是会变的商业数据，默认不配置就不估算 ----
    llm_price_prompt_per_m: Optional[float] = Field(
        default_factory=lambda: _env_float_or_none("LLM_PRICE_PROMPT_PER_M"),
        description="每百万 prompt token 单价（元）；配置后 /stats 才输出成本估算",
    )
    llm_price_completion_per_m: Optional[float] = Field(
        default_factory=lambda: _env_float_or_none("LLM_PRICE_COMPLETION_PER_M"),
        description="每百万 completion token 单价（元）",
    )
    query_rewrite_enabled: bool = Field(
        default_factory=lambda: _env_bool("QUERY_REWRITE_ENABLED", True),
        description=(
            "有历史时是否用 LLM 把追问改写成独立问题（指代消解）。"
            "改写失败会回退为原问题，不影响可用性。"
        ),
    )

    # ---- 检索 ----
    retrieval_mode: str = Field(
        default_factory=lambda: (_env_str("RETRIEVAL_MODE", "hybrid").lower() or "hybrid")
    )
    rrf_k: int = Field(default_factory=lambda: _env_int("RRF_K", 60))
    vector_weight: float = Field(default_factory=lambda: _env_float("VECTOR_WEIGHT", 1.0))
    keyword_weight: float = Field(default_factory=lambda: _env_float("KEYWORD_WEIGHT", 0.8))
    top_k: int = Field(default_factory=lambda: _env_int("TOP_K", 5))
    candidate_k: int = Field(default_factory=lambda: _env_int("CANDIDATE_K", 20))
    min_score: float = Field(default_factory=lambda: _env_float("MIN_SCORE", 0.05))
    refuse_when_empty: bool = Field(default_factory=lambda: _env_bool("REFUSE_WHEN_EMPTY", True))

    # ---- 拒答加固（不依赖模型是否"听话"）----
    # 背景：实测 6 条负样本的 refused 全是 false —— 检索非空，没走"检索为空就拒答"那条硬规则，
    # 是模型自己遵守提示词说了"未找到"。这意味着换模型/改提示词，拒答可能悄悄失效而指标看不出来。
    # 下面两个阈值给拒答加两道服务端兜底，与模型无关。
    refusal_evidence_floor: float = Field(
        default_factory=lambda: _env_float("REFUSAL_EVIDENCE_FLOOR", 0.01),
        description=(
            "最高召回分低于此值即视为无证据，不调用模型直接拒答。"
            "0 表示关闭。hybrid 下的融合分是 0.0x 量级，与 vector 模式的余弦不同量纲，故单独配置。"
        ),
    )
    refusal_unsupported_ratio: float = Field(
        default_factory=lambda: _env_float("REFUSAL_UNSUPPORTED_RATIO", 0.5),
        description=(
            "生成后若「找不到依据的句子」占比达到此值（且至少 2 句），判定为编造并把回答改为拒答。"
            "0 表示关闭。只看 unsupported，不看 no_citation —— 后者是格式问题不是幻觉。"
        ),
    )

    # ---- 精排（cross-encoder，可选）----
    # 启发式重排已到极限（见 app/core/retriever.py 的注释与实测记录），
    # 精排用 (query, passage) 联合编码判断语义相关性，能纠正"字面匹配但语义无关"的排序错误。
    # 默认关闭：它会显著增加单次检索耗时（CPU 上 20 个候选约 0.5~2 秒）。
    rerank_enabled: bool = Field(default_factory=lambda: _env_bool("RERANK_ENABLED", False))
    rerank_model: str = Field(
        default_factory=lambda: _env_str("RERANK_MODEL", "BAAI/bge-reranker-base")
    )
    rerank_top_n: int = Field(default_factory=lambda: _env_int("RERANK_TOP_N", 20))
    rerank_device: str = Field(default_factory=lambda: _env_str("RERANK_DEVICE", "cpu"))
    rerank_quantize: bool = Field(
        default_factory=lambda: _env_bool("RERANK_QUANTIZE", True),
        description=(
            "对精排模型做 int8 动态量化（Linear 层）。实测：CPU 上 2491ms → 885ms（-64%），"
            "top-8 成员集与 fp32 100% 一致、分数平均偏差 0.010，检索指标在噪声内。"
            "关闭可回到 fp32 全精度。"
        ),
    )
    rerank_max_length: int = Field(
        default_factory=lambda: _env_int("RERANK_MAX_LENGTH", 512)
    )
    rerank_batch_size: int = Field(
        default_factory=lambda: _env_int("RERANK_BATCH_SIZE", 16)
    )

    # ---- 逐句归因校验（防幻觉）----
    # 生成后把回答拆成句子，逐句检查"关键信息是否真在被引用的片段里"，
    # 专门抓「引用编号合法、但内容并不在片段中」这类编号校验看不出的幻觉。
    # 只报告不拦截：结果写进响应的 attribution 字段 + warnings。
    attribution_enabled: bool = Field(
        default_factory=lambda: _env_bool("ATTRIBUTION_ENABLED", True)
    )
    # 覆盖率 >= 此值判为"有依据"
    attribution_weak_threshold: float = Field(
        default_factory=lambda: _env_float("ATTRIBUTION_WEAK_THRESHOLD", 0.35)
    )
    # 覆盖率低于此值判为"疑似无依据"
    attribution_unsupported_threshold: float = Field(
        default_factory=lambda: _env_float("ATTRIBUTION_UNSUPPORTED_THRESHOLD", 0.12)
    )

    # ---- Agent 自检（反思节点）----
    # Agent 给出答案后，用逐句归因校验自检；发现"无依据断言"就带着反馈回到
    # agent 节点重答一轮。判据是确定性的，不额外调用 LLM 自评。
    agent_reflection_enabled: bool = Field(
        default_factory=lambda: _env_bool("AGENT_REFLECTION_ENABLED", True)
    )
    # 最多重答几轮。设 1 是刻意的：重答本身有成本，且多数问题一次修正就够。
    agent_max_reflections: int = Field(
        default_factory=lambda: _env_int("AGENT_MAX_REFLECTIONS", 1)
    )

    # ---- 服务 ----
    api_host: str = Field(default_factory=lambda: _env_str("API_HOST", "0.0.0.0"))
    api_port: int = Field(default_factory=lambda: _env_int("API_PORT", 8000))
    api_key: str = Field(default_factory=lambda: _env_str("API_KEY"))
    log_level: str = Field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO").upper())
    cors_origins: List[str] = Field(default_factory=lambda: _env_list("CORS_ORIGINS", ["*"]))
    max_upload_mb: int = Field(default_factory=lambda: _env_int("MAX_UPLOAD_MB", 20))

    # ---- 派生路径 ----
    @property
    def documents_dir(self) -> Path:
        return _resolve_path(self.documents_dir_raw, "./data/documents")

    @property
    def chroma_dir(self) -> Path:
        return _resolve_path(self.chroma_dir_raw, "./data/chroma")

    @property
    def upload_dir(self) -> Path:
        return _resolve_path(self.upload_dir_raw, "./data/uploads")

    @property
    def registry_path(self) -> Path:
        return _resolve_path(self.registry_path_raw, "./data/registry.json")

    @property
    def bm25_index_path(self) -> Path:
        return _resolve_path(self.bm25_index_path_raw, "./data/bm25_index.json")

    @property
    def guard_db_path(self) -> Path:
        """FreshGuard 声明库（SQLite）路径。"""
        return _resolve_path(self.guard_db_path_raw, "./data/claims.db")

    @property
    def agent_kb_path(self) -> Path:
        """Agent 业务工具的结构化知识库（故障码 / 保养周期）。"""
        return _resolve_path(self.agent_kb_path_raw, "./data/agent_kb.db")

    @property
    def agent_checkpoint_path(self) -> Path:
        """Agent 对话 checkpoint（LangGraph SqliteSaver）。

        刻意与业务库分开：SqliteSaver 自己管理连接，
        与业务查询共用同一连接会互相干扰（详见 app/agent/service.py 注释）。
        """
        return _resolve_path(self.agent_checkpoint_path_raw, "./data/agent_checkpoints.db")

    @property
    def chunk_overlap_effective(self) -> int:
        """防御：重叠必须小于块大小，否则切分会死循环。"""
        return max(0, min(self.chunk_overlap, max(0, self.chunk_size - 1)))

    @property
    def max_upload_bytes(self) -> int:
        return max(1, self.max_upload_mb) * 1024 * 1024

    # ---- 启动前自检 ----
    def normalize(self) -> List[str]:
        """纠正非法枚举值，返回需要提醒用户的说明列表。"""
        warnings: List[str] = []

        if self.embedding_provider not in VALID_EMBEDDING_PROVIDERS:
            warnings.append(
                f"EMBEDDING_PROVIDER='{self.embedding_provider}' 不是合法值，"
                f"已回退为 'hash'（可选：{'/'.join(VALID_EMBEDDING_PROVIDERS)}）"
            )
            self.embedding_provider = "hash"

        if self.retrieval_mode not in VALID_RETRIEVAL_MODES:
            warnings.append(
                f"RETRIEVAL_MODE='{self.retrieval_mode}' 不是合法值，"
                f"已回退为 'hybrid'（可选：{'/'.join(VALID_RETRIEVAL_MODES)}）"
            )
            self.retrieval_mode = "hybrid"

        if self.chunk_size < 120:
            warnings.append(
                f"CHUNK_SIZE={self.chunk_size} 小于建议下限，已按 120 处理"
                "（过小的片段会丢上下文，检索质量通常更差）"
            )
            self.chunk_size = 120

        if self.candidate_k < self.top_k:
            warnings.append(
                f"CANDIDATE_K={self.candidate_k} 小于 TOP_K={self.top_k}，"
                f"已自动提升 CANDIDATE_K 到 {self.top_k}"
            )
            self.candidate_k = self.top_k

        if self.rerank_top_n < self.top_k:
            warnings.append(
                f"RERANK_TOP_N={self.rerank_top_n} 小于 TOP_K={self.top_k}，"
                f"已自动提升到 {self.top_k}"
                "（精排候选不足时，排在后面的正确答案没有机会被救回来）"
            )
            self.rerank_top_n = self.top_k

        return warnings

    def missing_required(self) -> List[str]:
        """返回当前配置下仍然缺失的必填项（人类可读字符串列表）。"""
        missing: List[str] = []

        if self.embedding_provider == "api":
            if is_placeholder(self.embedding_api_key):
                missing.append("EMBEDDING_API_KEY")
            if is_placeholder(self.embedding_base_url):
                missing.append("EMBEDDING_BASE_URL")
            if is_placeholder(self.embedding_model):
                missing.append("EMBEDDING_MODEL")
            if not self.embedding_dim:
                missing.append("EMBEDDING_DIM")
        elif self.embedding_provider == "local":
            if is_placeholder(self.local_embedding_model):
                missing.append("LOCAL_EMBEDDING_MODEL")

        return missing

    def ensure_directories(self) -> None:
        for path in (
            self.documents_dir,
            self.chroma_dir,
            self.upload_dir,
            self.registry_path.parent,
            self.bm25_index_path.parent,
            self.guard_db_path.parent,
            self.agent_kb_path.parent,
            self.agent_checkpoint_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def public_summary(self) -> dict:
        """给 /health、/api/v1/stats 用的安全摘要：绝不含密钥明文。"""
        return {
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_configured": not is_placeholder(self.llm_api_key),
            "embedding_provider": self.embedding_provider,
            "embedding_model": (
                self.embedding_model
                if self.embedding_provider == "api"
                else self.local_embedding_model
                if self.embedding_provider == "local"
                else f"hash-{self.hash_embedding_dim}"
            ),
            "embedding_dim": self.embedding_dim_config,
            "retrieval_mode": self.retrieval_mode,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "rerank_enabled": self.rerank_enabled,
            "rerank_model": self.rerank_model if self.rerank_enabled else None,
            "rerank_top_n": self.rerank_top_n if self.rerank_enabled else None,
            "attribution_enabled": self.attribution_enabled,
            "agent_reflection_enabled": self.agent_reflection_enabled,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap_effective,
            "collection": self.chroma_collection,
            "documents_dir": str(self.documents_dir),
            "chroma_dir": str(self.chroma_dir),
            "api_auth_enabled": not is_placeholder(self.api_key),
            # 多轮会话与拒答兜底：换模型/改配置后，/health 一眼能看出这些开关的状态
            "session_max_turns": self.session_max_turns,
            "session_max_sessions": self.session_max_sessions,
            "query_rewrite_enabled": self.query_rewrite_enabled,
            "query_cache_enabled": self.query_cache_enabled,
            "query_cache_ttl_seconds": self.query_cache_ttl_seconds,
            "query_cache_max_entries": self.query_cache_max_entries,
            "cost_estimation_enabled": (
                self.llm_price_prompt_per_m is not None
                and self.llm_price_completion_per_m is not None
            ),
            "refusal_evidence_floor": self.refusal_evidence_floor,
            "refusal_unsupported_ratio": self.refusal_unsupported_ratio,
            "missing_required": self.missing_required(),
        }

    @property
    def embedding_dim_config(self) -> int:
        if self.embedding_provider == "api":
            return int(self.embedding_dim or 0)
        if self.embedding_provider == "local":
            return int(self.local_embedding_dim or 0)
        return int(self.hash_embedding_dim)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    instance = Settings()
    for message in instance.normalize():
        # 这里用 logging 而不是 print，保证和全局日志格式一致
        logging.getLogger(__name__).warning("配置纠正：%s", message)
    return instance


settings = get_settings()
