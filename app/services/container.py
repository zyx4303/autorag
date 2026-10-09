"""服务装配：把 embedding / 向量库 / BM25 / 检索器 / LLM / 问答链路拼起来。

启动阶段遵循"能起来就先起来"的原则：
- LLM 客户端永远可构造（只在使用时校验 Key）；
- 向量库或 embedder 初始化失败时，把原因记进 container.init_errors，
  服务照常提供 /health 与 /api/v1/stats，业务接口返回 503 并附带原因，
  方便你本地排错，而不是看到一个没有任何信息的启动崩溃。
"""
from __future__ import annotations

from app.api.deps import ServiceContainer
from app.config import Settings
from app.core.bm25 import BM25Index
from app.core.embedding import build_embedder
from app.core.llm_client import LLMClient
from app.core.retriever import HybridRetriever
from app.core.vector_store import VectorStore
from app.logging_conf import get_logger
from app.services.ingestion_service import IngestionService
from app.services.eval_service import EvaluationService
from app.services.rag_pipeline import QAEngine

logger = get_logger(__name__)


def build_container(settings: Settings) -> ServiceContainer:
    settings.ensure_directories()

    container = ServiceContainer(settings=settings, llm=LLMClient(settings))

    # ---- Embedding ----
    try:
        container.embedder = build_embedder(settings)
    except Exception as exc:
        message = f"Embedding 初始化失败：{exc}"
        logger.error(message)
        container.init_errors.append(message)

    # ---- 向量库 ----
    if container.embedder is not None:
        try:
            container.vector_store = VectorStore(
                persist_dir=str(settings.chroma_dir),
                collection_name=settings.chroma_collection,
                embedder=container.embedder,
            )
        except Exception as exc:
            message = f"向量库初始化失败（{settings.chroma_dir}）：{exc}"
            logger.error(message)
            container.init_errors.append(message)

    # ---- BM25 ----
    container.bm25 = BM25Index(persist_path=settings.bm25_index_path)

    # ---- FreshGuard：声明库与变更影响分析 ----
    # 只依赖 LLM 客户端与切分器，不依赖向量库；
    # 因此即使向量库初始化失败，变更分析功能仍然可用。
    try:
        from app.guard.service import DriftService
        from app.guard.store import ClaimStore

        container.claim_store = ClaimStore(settings.guard_db_path)
        container.drift = DriftService(settings, container.llm, container.claim_store)
    except Exception as exc:
        message = f"变更分析模块初始化失败：{exc}"
        logger.error(message)
        container.init_errors.append(message)

    # ---- 检索器 / 入库 / 问答 ----
    if container.vector_store is not None:
        container.retriever = HybridRetriever(settings, container.vector_store, container.bm25)
        container.ingestion = IngestionService(settings, container.vector_store, container.bm25)
        container.qa = QAEngine(settings, container.retriever, container.llm)
        container.evaluation = EvaluationService(settings, container.qa)

        # BM25 索引：优先从磁盘加载，缺失或与向量库条数不符时重建
        loaded = container.bm25.load()
        collection_count = container.vector_store.count()
        if not loaded or container.bm25.size != collection_count:
            logger.info(
                "BM25 索引需要重建（磁盘 size=%d，向量库 count=%d）",
                container.bm25.size,
                collection_count,
            )
            container.ingestion.rebuild_bm25()

    # ---- Agent（LangGraph）：业务工具知识库 + 状态图 + checkpoint ----
    # 放在检索器之后，因为 search_kb 工具需要复用已有的检索链路
    try:
        from app.agent.kb import AgentKbStore
        from app.agent.service import AgentService

        container.agent_kb = AgentKbStore(settings.agent_kb_path)

        # checkpoint 交给 AgentService 延迟初始化：
        # 异步图要求 AsyncSqliteSaver，而它的初始化本身是异步的，
        # 在同步的容器构建阶段无法完成（实测报 NotImplementedError）。
        # 这里传 None，由服务在第一次请求的事件循环里建好 SqliteSaver 并缓存。
        container.agent = AgentService(
            settings=settings,
            llm=container.llm,
            retriever=container.retriever,
            kb_store=container.agent_kb,
            checkpointer=None,
        )
        # 惰性建库：库为空时才解析示例文档，避免每次启动都写库
        bootstrap = container.agent.bootstrap_kb(force=False)
        if not bootstrap.get("skipped"):
            logger.info(
                "Agent 业务知识库已建库：故障码 %d 条，保养项目 %d 条",
                bootstrap.get("dtc_inserted"),
                bootstrap.get("maintenance_inserted"),
            )
    except Exception as exc:
        message = f"Agent 模块初始化失败：{type(exc).__name__}: {exc}"
        logger.error(message)
        container.init_errors.append(message)

    for note in container.degraded_notes():
        logger.warning("配置提示：%s", note)

    return container
