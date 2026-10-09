#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""启动前自检（preflight）：一条命令把所有可能出问题的地方查一遍。

用法（在项目根目录 AutoRAG 下执行）：
    python scripts/preflight.py
    python scripts/preflight.py --ping-llm          # 额外真实调用一次 LLM（消耗约 1 token）
    python scripts/preflight.py --no-write-report   # 不落盘报告，只打印

它会依次检查：
  1. Python 版本、项目结构、核心依赖是否装齐（含版本号）
  2. .env 是否存在、哪些项还是 [待补充]、配置值是否合法
  3. 所有 app 模块能否导入（等价于一次编译级检查）
  4. 切分器、BM25、hash 向量、引用校验这几个纯逻辑组件能否工作
  5. embedder 能否真的产出向量、维度是多少
  6. Chroma 能否打开持久化目录、集合里有多少条、向量维度是否一致
  7. LLM 是否配置、base_url 是否形如 .../v1
  8. eval/eval_set.json 是否是合法 JSON、20 条里还有多少字段是 [待补充]

报告会同时打印到屏幕并写入 data/preflight_report.txt，
把它整段复制给我，我就能定位问题，不需要你的任何密钥。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台默认不是 UTF-8，中文会变乱码；这里强制切到 UTF-8 输出。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # pragma: no cover - 非 tty 或旧版本 python
        pass

# --------------------------------------------------------------------------- #
# 输出工具：同时写屏幕和内存，最后一次性落盘
# --------------------------------------------------------------------------- #
_LINES: List[str] = []


def emit(text: str = "") -> None:
    print(text)
    _LINES.append(text)


def section(title: str) -> None:
    emit("")
    emit("=" * 70)
    emit(title)
    emit("=" * 70)


_PASS = "  [PASS] "
_FAIL = "  [FAIL] "
_WARN = "  [WARN] "
_INFO = "  [INFO] "


# --------------------------------------------------------------------------- #
# .env 解析（不污染 os.environ，只读文件）
# --------------------------------------------------------------------------- #
def parse_env_file(path) -> dict:
    """解析 .env 为 dict；接受 str 或 Path。"""
    path = Path(path)
    values: dict = {}
    if not path.exists():
        return values
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-8", errors="replace")
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def is_placeholder(value: Optional[str]) -> bool:
    if value is None:
        return True
    text = value.strip()
    if not text:
        return True
    return text.startswith("[待补充") or text in {"your-api-key", "sk-xxx", "changeme"}


class Report:
    def __init__(self) -> None:
        self.blocking: List[str] = []
        self.problems: List[str] = []
        self.warnings: List[str] = []

    def fail(self, message: str, blocking: bool = False) -> None:
        emit(_FAIL + message)
        if blocking:
            self.blocking.append(message)
        else:
            self.problems.append(message)

    def warn(self, message: str) -> None:
        emit(_WARN + message)
        self.warnings.append(message)

    def ok(self, message: str) -> None:
        emit(_PASS + message)

    def info(self, message: str) -> None:
        emit(_INFO + message)


# --------------------------------------------------------------------------- #
# 1. 环境与依赖
# --------------------------------------------------------------------------- #
def check_python(report: Report) -> None:
    section("1. Python 与依赖")
    version = sys.version_info
    text = f"{version.major}.{version.minor}.{version.micro}"
    if version < (3, 10):
        report.fail(f"Python {text} 过低，本项目需要 3.10+", blocking=True)
    else:
        report.ok(f"Python {text}（{sys.executable}）")

    emit(_INFO + f"工作目录：{os.getcwd()}")
    emit(_INFO + f"项目根目录：{PROJECT_ROOT}")
    if PROJECT_ROOT != Path(os.getcwd()).resolve():
        report.warn(
            "当前工作目录不是项目根目录。请先 cd 到项目根目录再跑，"
            "否则 uvicorn app.main:app 会因为找不到 app 包而报 ModuleNotFoundError"
        )

    # 依赖检查：fastapi 单独判版本，因为 Body(default_factory=...) 需要 >= 0.115
    dependencies = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("pydantic_settings", "pydantic-settings"),
        ("dotenv", "python-dotenv"),
        ("httpx", "httpx"),
        ("chromadb", "chromadb"),
        ("multipart", "python-multipart"),
        # Agent 编排依赖（缺失时只有 /agent/* 接口不可用，其余功能不受影响）
        ("langgraph", "langgraph"),
        ("aiosqlite", "aiosqlite"),
    ]
    for module_name, package_name in dependencies:
        try:
            module = __import__(module_name)
            installed = getattr(module, "__version__", "未知")
            emit(_PASS + f"{package_name:<18} {installed}")
        except ImportError:
            report.fail(
                f"缺少依赖 {package_name}（import {module_name} 失败）→ "
                f"请执行： pip install -r requirements.txt",
                blocking=True,
            )

    try:
        import fastapi

        raw = getattr(fastapi, "__version__", "0")
        parts = []
        for piece in str(raw).split(".")[:2]:
            digits = "".join(ch for ch in piece if ch.isdigit())
            parts.append(int(digits or 0))
        while len(parts) < 2:
            parts.append(0)
        if tuple(parts) < (0, 115):
            report.fail(
                f"FastAPI {raw} 低于 0.115，Body(default_factory=...) 会 TypeError → "
                f"请执行： pip install -U \"fastapi>=0.115\"",
                blocking=True,
            )
        else:
            report.ok(f"FastAPI 版本 {raw} 支持 Body(default_factory=...)")
    except Exception as exc:  # pragma: no cover
        report.warn(f"无法判断 FastAPI 版本：{exc}")

    # 可选依赖
    optional = [("pypdf", "解析 PDF"), ("docx", "解析 DOCX"), ("sentence_transformers", "本地向量模型")]
    for module_name, purpose in optional:
        try:
            __import__(module_name)
            emit(_PASS + f"可选依赖 {module_name} 已安装（{purpose}）")
        except ImportError:
            emit(_INFO + f"可选依赖 {module_name} 未安装（{purpose}，不装也能跑）")


# --------------------------------------------------------------------------- #
# 2. .env
# --------------------------------------------------------------------------- #
def check_env(report: Report) -> dict:
    section("2. .env 配置")
    env_path = PROJECT_ROOT / ".env"
    example_path = PROJECT_ROOT / ".env.example"
    values = parse_env_file(env_path)

    if not env_path.exists():
        report.fail(
            ".env 不存在 → 请执行 copy .env.example .env（Windows）或 cp .env.example .env",
            blocking=True,
        )
    else:
        report.ok(f".env 存在：{env_path}（{env_path.stat().st_size} 字节）")

    if not example_path.exists():
        report.warn(".env.example 缺失（不影响运行，但建议保留作为模板）")

    if not values:
        report.warn(".env 里没有解析到任何 KEY=VALUE，请检查文件是不是空的或格式不对")
        return values

    provider = (values.get("EMBEDDING_PROVIDER") or "hash").strip().lower()
    if provider not in {"api", "local", "hash"}:
        report.fail(
            f"EMBEDDING_PROVIDER='{provider}' 不是合法值（可选 api/local/hash）；"
            "服务启动时会自动回退成 hash 并在日志里提示"
        )

    mode = (values.get("RETRIEVAL_MODE") or "hybrid").strip().lower()
    if mode not in {"vector", "keyword", "hybrid"}:
        report.fail(f"RETRIEVAL_MODE='{mode}' 不是合法值（可选 vector/keyword/hybrid）")

    emit("")
    emit(_INFO + f"EMBEDDING_PROVIDER = {provider}")
    emit(_INFO + f"RETRIEVAL_MODE     = {mode}")

    # LLM
    llm_key = values.get("LLM_API_KEY")
    llm_base = (values.get("LLM_BASE_URL") or "").strip()
    llm_model = (values.get("LLM_MODEL") or "").strip()
    if is_placeholder(llm_key):
        report.fail(
            "LLM_API_KEY 仍是空或 [待补充] → 问答接口会返回 [生成失败]，"
            "请填写后重启服务（注意 .env 只在进程启动时读取一次）"
        )
    else:
        report.ok(f"LLM_API_KEY 已填写（长度 {len(llm_key.strip())}，内容不会打印）")
    if is_placeholder(llm_base):
        report.fail("LLM_BASE_URL 为空 → 需填写形如 https://api.deepseek.com/v1 的地址")
    elif not llm_base.rstrip("/").endswith("/v1") and "/v1" not in llm_base:
        report.warn(
            f"LLM_BASE_URL='{llm_base}' 看起来不含 /v1，"
            "多数 OpenAI 兼容服务需要 .../v1，否则会 404"
        )
    else:
        report.ok(f"LLM_BASE_URL 形如：{llm_base}")
    if is_placeholder(llm_model):
        report.fail("LLM_MODEL 为空 → 需填写模型名，例如 deepseek-chat")
    else:
        report.ok(f"LLM_MODEL = {llm_model}")

    # Embedding
    if provider == "api":
        for key_name in ("EMBEDDING_API_KEY", "EMBEDDING_BASE_URL", "EMBEDDING_MODEL", "EMBEDDING_DIM"):
            if is_placeholder(values.get(key_name)):
                report.fail(
                    f"EMBEDDING_PROVIDER=api 但 {key_name} 仍为空或 [待补充] → "
                    "服务启动时 embedder 会构造失败，检索接口返回 503",
                    blocking=False,
                )
        dim_raw = (values.get("EMBEDDING_DIM") or "").strip()
        if dim_raw and not dim_raw.isdigit():
            report.fail(f"EMBEDDING_DIM='{dim_raw}' 不是整数 → 必须是模型文档里的真实维度")
        if not is_placeholder(values.get("EMBEDDING_API_KEY")):
            report.ok("EMBEDDING_API_KEY 已填写（内容不会打印）")
        if not is_placeholder(values.get("EMBEDDING_BASE_URL")):
            report.ok(f"EMBEDDING_BASE_URL = {values.get('EMBEDDING_BASE_URL')}")
        if not is_placeholder(values.get("EMBEDDING_MODEL")):
            report.ok(f"EMBEDDING_MODEL = {values.get('EMBEDDING_MODEL')}")
        if dim_raw:
            report.ok(f"EMBEDDING_DIM = {dim_raw}")
    elif provider == "local":
        if is_placeholder(values.get("LOCAL_EMBEDDING_MODEL")):
            report.fail("EMBEDDING_PROVIDER=local 但 LOCAL_EMBEDDING_MODEL 未填写")
        try:
            import sentence_transformers  # noqa: F401
            report.ok("sentence-transformers 已安装，可使用 local 模式")
        except ImportError:
            report.warn(
                "EMBEDDING_PROVIDER=local 但未安装 sentence-transformers → "
                "服务会自动降级为 hash 模式；请执行 pip install -r requirements-local.txt 或改用 api"
            )
    else:
        report.warn(
            "EMBEDDING_PROVIDER=hash：向量由字符 n-gram 哈希生成，只能验证链路，"
            "语义检索能力很弱（问法与原文不一致就召回不到）。正式使用请改 api"
        )

    # 路径
    emit("")
    for key_name, default in (
        ("DOCUMENTS_DIR", "./data/documents"),
        ("CHROMA_DIR", "./data/chroma"),
        ("UPLOAD_DIR", "./data/uploads"),
        ("REGISTRY_PATH", "./data/registry.json"),
        ("BM25_INDEX_PATH", "./data/bm25_index.json"),
    ):
        emit(_INFO + f"{key_name:<18} = {values.get(key_name) or default + '（默认）'}")

    return values


# --------------------------------------------------------------------------- #
# 3. 模块导入
# --------------------------------------------------------------------------- #
def check_imports(report: Report) -> bool:
    section("3. 模块导入检查（等价于编译级检查）")
    modules = [
        "app.config",
        "app.models",
        "app.logging_conf",
        "app.core.text_utils",
        "app.core.chunker",
        "app.core.loader",
        "app.core.embedding",
        "app.core.vector_store",
        "app.core.bm25",
        "app.core.retriever",
        "app.core.llm_client",
        "app.services.container",
        "app.services.ingestion_service",
        "app.services.rag_pipeline",
        "app.services.eval_service",
        "app.api.deps",
        "app.api.routes_system",
        "app.api.routes_ingest",
        "app.api.routes_qa",
        "app.api.routes_eval",
        "app.main",
    ]
    failed = 0
    for name in modules:
        try:
            __import__(name)
            emit(_PASS + name)
        except Exception as exc:
            failed += 1
            report.fail(f"导入 {name} 失败：{type(exc).__name__}: {exc}", blocking=True)
            emit("         " + traceback.format_exc().strip().splitlines()[-1])
    if failed == 0:
        report.ok(f"全部 {len(modules)} 个模块导入成功")
    return failed == 0


# --------------------------------------------------------------------------- #
# 4. 纯逻辑组件
# --------------------------------------------------------------------------- #
def check_logic(report: Report) -> None:
    section("4. 纯逻辑组件（切分 / BM25 / hash 向量 / 引用校验）")
    try:
        from app.config import Settings
        from app.core.bm25 import BM25Index
        from app.core.chunker import chunk_document
        from app.core.embedding import HashEmbedder
        from app.models import Citation, SourceDocument
        from app.services.rag_pipeline import QAEngine
    except Exception as exc:
        report.fail(f"组件导入失败，跳过本节：{exc}", blocking=True)
        return

    import asyncio

    sample = (
        "# 保养手册\n\n"
        "## 首保\n\n"
        "首保应在 5000 公里或 6 个月内完成，以先到者为准。"
        "首保由授权服务站免费执行，建议提前预约以免等待。"
        "首保完成后服务顾问会在系统内登记保养记录。\n\n"
        "## 机油\n\n"
        "推荐使用 5W-30 黏度等级的机油。更换机油时应同时更换机油滤清器。"
        "机油加注量以机油尺刻度为准，不要过量加注。"
        "不同年款车型的加注量存在差异，请以随车手册为准。\n"
    )
    # 注意：这里必须显式传入全部切分参数。Settings 的其它字段会从 .env 读取，
    # 如果 .env 里的 CHUNK_SIZE/CHUNK_MIN_CHARS 与本处不同，会干扰这段自检
    # （典型现象：CHUNK_MIN_CHARS 很大时，碎块合并会把两段重新并回一段，
    #   从而误判成"切分器坏了"）。显式传参的优先级高于环境变量，因此这里安全。
    settings = Settings(
        chunk_size=120,
        chunk_overlap=20,
        chunk_min_chars=10,
        embedding_provider="hash",
    )
    document = SourceDocument(
        doc_id="doc_preflight", source="preflight.md", title="保养手册", text=sample, checksum="x" * 8
    )
    chunks = chunk_document(document, settings)
    sections = [chunk.section for chunk in chunks]
    if len(chunks) < 2:
        report.fail(
            f"切分器未切出多个片段：chunks={len(chunks)}（chunk_size={settings.chunk_size}，"
            f"正文约 {len(sample)} 字符，按此参数应至少 2 段）"
        )
    elif not any("首保" in section or "机油" in section for section in sections):
        report.fail(f"切分器没有识别出章节路径：sections={sections}")
    else:
        report.ok(f"切分器正常：{len(chunks)} 个片段，章节路径 = {sections}")
        # 重叠校验：后一块开头应包含前一块的尾部若干字符
        overlap_hit = any(
            chunks[index].text[:10] and chunks[index].text[:10] in chunks[index - 1].text[-40:]
            for index in range(1, len(chunks))
        )
        if overlap_hit or settings.chunk_overlap == 0:
            report.ok("相邻片段重叠生效")
        else:
            report.warn(
                "未观察到相邻片段重叠（不影响检索，但可能是 chunk_overlap 与文本长度不匹配）"
            )

    index = BM25Index(persist_path=None)
    index.build(
        [
            {"chunk_id": "c1", "text": "首保应在 5000 公里或 6 个月内完成", "source": "a.md", "position": 0},
            {"chunk_id": "c2", "text": "故障码 P0420 表示催化器效率低", "source": "b.md", "position": 1},
        ]
    )
    hits = index.search("P0420 是什么故障码", top_k=2)
    if hits and hits[0][0]["chunk_id"] == "c2":
        report.ok("BM25 正常：中文与英文数字混合查询都能命中")
    else:
        report.fail(f"BM25 召回异常：{hits}")

    embedder = HashEmbedder(64)

    async def run_embed():
        return await embedder.embed_documents(["测试文本"])

    try:
        vectors = asyncio.run(run_embed())
        if vectors and len(vectors[0]) == 64:
            report.ok("hash embedder 正常：产出 64 维向量")
        else:
            report.fail(f"hash embedder 维度异常：{len(vectors[0]) if vectors else 0}")
    except Exception as exc:
        report.fail(f"hash embedder 抛异常：{type(exc).__name__}: {exc}")

    used, warnings = QAEngine.verify_citations(
        "结论[1]与[9]。",
        [Citation(index=1, chunk_id="c1", source="a.md", position=0)],
    )
    if len(warnings) == 1 and "9" in warnings[0]:
        report.ok("引用越界校验正常：能识别出 [9] 这类不存在的编号")
    else:
        report.fail(f"引用校验异常：warnings={warnings}")


# --------------------------------------------------------------------------- #
# 5. embedder 真实探测
# --------------------------------------------------------------------------- #
def check_embedder(report: Report, values: dict) -> None:
    section("5. Embedder 真实探测（会调用一次 embedding 接口）")
    try:
        from app.config import get_settings
        from app.core.embedding import build_embedder
    except Exception as exc:
        report.fail(f"无法导入 embedder 工厂：{exc}", blocking=True)
        return

    settings = get_settings()
    try:
        embedder = build_embedder(settings)
    except Exception as exc:
        report.fail(f"embedder 构造失败：{type(exc).__name__}: {exc}", blocking=True)
        return

    info = embedder.describe()
    emit(_INFO + f"provider={info['provider']} 声明维度={info['dimension']} degraded={info['degraded']}")
    if info.get("degrade_reason"):
        report.warn(f"降级原因：{info['degrade_reason']}")

    import asyncio

    async def probe():
        return await embedder.embed_documents(["保养周期与机油规格"])

    try:
        vectors = asyncio.run(probe())
    except Exception as exc:
        report.fail(
            f"embedding 实际调用失败：{type(exc).__name__}: {str(exc)[:300]}",
            blocking=True,
        )
        return

    if not vectors or not vectors[0]:
        report.fail("embedding 返回空向量", blocking=True)
        return

    actual_dim = len(vectors[0])
    report.ok(f"embedding 调用成功，实际维度 = {actual_dim}")

    declared = int(info.get("dimension") or 0)
    if declared and declared != actual_dim:
        report.fail(
            f"维度不一致：.env 里配置的是 {declared}，接口实际返回 {actual_dim} → "
            "请把 EMBEDDING_DIM 改成实际值，并重建向量库",
            blocking=True,
        )
    elif not declared:
        report.warn("未配置 EMBEDDING_DIM，已按接口返回的维度推断；建议在 .env 里显式写死")


# --------------------------------------------------------------------------- #
# 6. Chroma
# --------------------------------------------------------------------------- #
def check_chroma(report: Report) -> None:
    section("6. Chroma 向量库")
    try:
        from app.config import get_settings
        from app.core.embedding import build_embedder
        from app.core.vector_store import VectorStore
    except Exception as exc:
        report.fail(f"无法导入向量库组件：{exc}", blocking=True)
        return

    settings = get_settings()
    emit(_INFO + f"CHROMA_DIR = {settings.chroma_dir}")
    try:
        settings.ensure_directories()
        report.ok("数据目录创建/检查成功")
    except Exception as exc:
        report.fail(f"创建数据目录失败：{type(exc).__name__}: {exc}", blocking=True)
        return

    try:
        embedder = build_embedder(settings)
    except Exception as exc:
        report.fail(f"embedder 不可用，跳过向量库检查：{exc}", blocking=True)
        return

    try:
        store = VectorStore(
            persist_dir=str(settings.chroma_dir),
            collection_name=settings.chroma_collection,
            embedder=embedder,
        )
    except Exception as exc:
        report.fail(
            f"打开 Chroma 失败：{type(exc).__name__}: {exc} → "
            "常见原因：目录被占用、data/chroma 里有上一个版本的不兼容文件"
            "（可备份后删除 data/chroma 重试）",
            blocking=True,
        )
        return

    count = store.count()
    report.ok(f"集合 '{store.collection_name}' 打开成功，当前片段数 = {count}")

    if count == 0:
        report.warn(
            "向量库是空的 → 请把文档放进 data/documents 后执行 python scripts/ingest.py --rebuild"
        )
        return

    sources = store.sources()
    emit(_INFO + f"库中来源文件 {len(sources)} 个：" + "、".join(sources[:10]) + ("…" if len(sources) > 10 else ""))

    # 记录里存了向量维度，用来判断"换了 embedding 但没重建"
    try:
        record = store._collection.get(limit=1, include=["embeddings"])  # noqa: SLF001
        embeddings = record.get("embeddings")
        stored_dim = 0
        if embeddings is not None and len(embeddings) > 0:
            stored_dim = len(embeddings[0])
        declared_dim = int(embedder.dimension or 0)
        if stored_dim and declared_dim:
            emit(_INFO + f"库中向量维度 = {stored_dim}，当前 embedder 维度 = {declared_dim}")
            if stored_dim != declared_dim:
                report.fail(
                    "库里存的向量维度与当前 embedding 配置不一致 → "
                    "说明换过 embedding 模型但没重建，请执行： python scripts/ingest.py --rebuild",
                    blocking=True,
                )
            else:
                report.ok("库中向量维度与当前配置一致")
    except Exception as exc:
        report.warn(f"读取库中向量维度失败（不影响使用）：{type(exc).__name__}: {str(exc)[:160]}")


# --------------------------------------------------------------------------- #
# 7. LLM
# --------------------------------------------------------------------------- #
def check_llm(report: Report, ping: bool) -> None:
    section("7. LLM 连通性")
    try:
        from app.config import get_settings
        from app.core.llm_client import LLMClient
    except Exception as exc:
        report.fail(f"无法导入 LLM 客户端：{exc}", blocking=True)
        return

    import asyncio

    settings = get_settings()
    client = LLMClient(settings)
    emit(_INFO + f"请求地址 = {client.url}")
    emit(_INFO + f"模型 = {client.model}")

    if not client.configured:
        report.fail("LLM_API_KEY 未配置 → 问答接口会直接返回 [生成失败] 提示")
        return

    if not ping:
        report.ok("LLM_API_KEY 已配置（未做真实调用；加 --ping-llm 可真实探测一次）")
        return

    async def probe():
        return await client.health_check()

    try:
        result = asyncio.run(probe())
    except Exception as exc:
        report.fail(f"LLM 探测异常：{type(exc).__name__}: {exc}")
        return

    if result.get("ok"):
        report.ok(f"LLM 真实调用成功：{result.get('detail')}")
    else:
        report.fail(f"LLM 真实调用失败：{result.get('detail')}")


# --------------------------------------------------------------------------- #
# 8. 评测集与文档
# --------------------------------------------------------------------------- #
def check_data(report: Report) -> None:
    section("8. 知识库文档与评测集")
    try:
        from app.config import get_settings

        settings = get_settings()
        documents_dir = settings.documents_dir
    except Exception:
        documents_dir = PROJECT_ROOT / "data" / "documents"

    if not documents_dir.exists():
        report.fail(f"知识库目录不存在：{documents_dir}")
    else:
        files = [
            path
            for path in documents_dir.rglob("*")
            if path.is_file() and not path.name.startswith(".")
        ]
        emit(_INFO + f"DOCUMENTS_DIR = {documents_dir}")
        if not files:
            report.fail("知识库目录里没有任何文档 → 入库后 collection_count 会是 0")
        else:
            report.ok(f"发现 {len(files)} 个文件：")
            for path in files[:15]:
                emit("         - " + path.relative_to(documents_dir).as_posix())
            if len(files) > 15:
                emit(f"         …… 其余 {len(files) - 15} 个省略")
            placeholder_files = [path for path in files if "待补充" in path.name]
            if placeholder_files:
                report.warn(
                    "目录里还有我生成的占位示例文档（文件名含「待补充」），"
                    "记得换成你的真实资料后重新入库"
                )

    dataset_path = PROJECT_ROOT / "eval" / "eval_set.json"
    if not dataset_path.exists():
        report.fail(f"评测集不存在：{dataset_path}")
        return
    try:
        dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.fail(
            f"评测集 JSON 解析失败：第 {exc.lineno} 行第 {exc.colno} 列 {exc.msg} → "
            "常见原因是值里出现了未转义的半角双引号"
        )
        return

    items = dataset.get("items") or []
    emit(_INFO + f"评测集：{dataset_path.name}，共 {len(items)} 条")
    incomplete = []
    for item in items:
        missing = []
        for field in ("ground_truth", "expected_sources", "expected_keywords"):
            value = item.get(field)
            if isinstance(value, list):
                if not value or all(is_placeholder(str(entry)) for entry in value):
                    missing.append(field)
            elif is_placeholder(value):
                missing.append(field)
        if missing:
            incomplete.append((item.get("id"), missing))

    if incomplete:
        report.warn(
            f"{len(incomplete)}/{len(items)} 条仍是 [待补充]（只影响人工复核口径，不影响脚本能否运行）"
        )
        for item_id, fields in incomplete[:5]:
            emit(f"         - {item_id}: {', '.join(fields)}")
        if len(incomplete) > 5:
            emit(f"         …… 其余 {len(incomplete) - 5} 条省略")
    else:
        report.ok("20 条评测集字段已全部填齐")


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="AutoRAG 启动前自检")
    parser.add_argument("--ping-llm", action="store_true", help="真实调用一次 LLM（约 1 token）")
    parser.add_argument("--no-write-report", action="store_true", help="不写报告文件")
    args = parser.parse_args()

    emit("AutoRAG 启动前自检（preflight）")
    emit(f"时间：{__import__('time').strftime('%Y-%m-%d %H:%M:%S')}")

    report = Report()
    check_python(report)
    values = check_env(report)
    imports_ok = check_imports(report)
    if imports_ok:
        check_logic(report)
        check_embedder(report, values)
        check_chroma(report)
        check_llm(report, args.ping_llm)
    else:
        section("4~7. 已跳过")
        emit(_WARN + "模块导入失败，后续检查无法进行，请先解决第 3 节的导入错误")

    check_data(report)

    section("汇总")
    emit(f"阻塞项（必须先修）：{len(report.blocking)}")
    for message in report.blocking:
        emit("  - " + message)
    emit(f"问题项：{len(report.problems)}")
    for message in report.problems:
        emit("  - " + message)
    emit(f"提示项：{len(report.warnings)}")
    for message in report.warnings:
        emit("  - " + message)

    if not report.blocking and not report.problems:
        emit("")
        emit("结论：可以启动服务。建议顺序：")
        emit("  1) uvicorn app.main:app --reload --port 8000")
        emit("  2) python scripts/ingest.py --rebuild")
        emit("  3) curl -X POST http://127.0.0.1:8000/api/v1/retrieve -H \"Content-Type: application/json\" -d \"{\\\"query\\\":\\\"你的测试问题\\\"}\"")
        emit("  4) curl -X POST http://127.0.0.1:8000/api/v1/chat -H \"Content-Type: application/json\" -d \"{\\\"question\\\":\\\"你的测试问题\\\"}\"")
        emit("  5) Agent（自主调用工具）:")
        emit("     curl http://127.0.0.1:8000/api/v1/agent/status   # 看可用工具与业务库规模")
        emit("     curl -X POST http://127.0.0.1:8000/api/v1/agent/chat -H \"Content-Type: application/json\" -d \"{\\\"session_id\\\":\\\"demo\\\",\\\"message\\\":\\\"P0195 故障码什么意思\\\"}\"")
        emit("     python tests/agent_smoke_test.py                     # Agent 行为测试（确定性，不联网）")
    else:
        emit("")
        emit("结论：存在需要处理的项目，请先按上面提示修复后重新运行本脚本。")

    if not args.no_write_report:
        report_path = PROJECT_ROOT / "data" / "preflight_report.txt"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text("\n".join(_LINES), encoding="utf-8")
            emit("")
            emit(f"报告已保存：{report_path}")
            emit("把上面这段输出（或该文件内容）整段发给我即可，里面不含任何密钥。")
        except OSError as exc:
            emit(f"报告保存失败（不影响结论）：{exc}")

    if report.blocking:
        return 2
    return 1 if report.problems else 0


if __name__ == "__main__":
    sys.exit(main())
