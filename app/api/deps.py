"""FastAPI 依赖：服务容器、状态获取、可选的 API Key 校验。

设计取舍：本项目的定位是"面试可讲清、本地可跑通"的实习项目，
因此默认不启用鉴权（API_KEY 留空即关闭）；一旦在 .env 里填了 API_KEY，
所有业务接口都会要求请求头 X-API-Key，方便演示最小的访问控制。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

from fastapi import Depends, HTTPException, Request, status

from app.config import Settings
from app.core.bm25 import BM25Index
from app.core.embedding import BaseEmbedder
from app.core.llm_client import LLMClient
from app.core.retriever import HybridRetriever
from app.core.vector_store import VectorStore
from app.services.ingestion_service import IngestionService
from app.services.eval_service import EvaluationService
from app.services.rag_pipeline import QAEngine


@dataclass
class ServiceContainer:
    """把一次进程生命周期内的所有有状态组件放一起，便于依赖注入与测试替换。"""

    settings: Settings
    llm: LLMClient
    embedder: Optional[BaseEmbedder] = None
    vector_store: Optional[VectorStore] = None
    bm25: Optional[BM25Index] = None
    retriever: Optional[HybridRetriever] = None
    ingestion: Optional[IngestionService] = None
    qa: Optional[QAEngine] = None
    evaluation: Optional[EvaluationService] = None
    # FreshGuard：知识库变更影响分析（声明库 + 差分服务）
    claim_store: Optional[Any] = None
    drift: Optional[Any] = None
    init_errors: List[str] = field(default_factory=list)

    @property
    def vector_ready(self) -> bool:
        return self.vector_store is not None and self.retriever is not None

    @property
    def qa_ready(self) -> bool:
        return self.qa is not None and self.vector_store is not None

    @property
    def ingest_ready(self) -> bool:
        return self.ingestion is not None

    def degraded_notes(self) -> List[str]:
        """需要向调用方明示的降级/风险提示（不隐藏、不美化）。"""
        notes: List[str] = []
        if self.embedder is not None and getattr(self.embedder, "degraded", False):
            notes.append(
                f"Embedding 已降级为 hash 模式：{getattr(self.embedder, 'degrade_reason', '')}"
            )
        elif self.settings.embedding_provider == "hash":
            notes.append(
                "当前 EMBEDDING_PROVIDER=hash：向量由字符 n-gram 哈希生成，"
                "不具备真正的语义检索能力，仅用于验证链路。正式使用请配置 api。"
            )
        if not self.llm.configured:
            notes.append("LLM_API_KEY 未配置，问答接口会直接返回配置错误提示。")
        notes.extend(self.init_errors)
        return notes


def get_container(request: Request) -> ServiceContainer:
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - 正常启动流程不会走到
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="服务尚未初始化完成"
        )
    return container


def get_qa(container: ServiceContainer = Depends(get_container)) -> QAEngine:
    if not container.qa_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="问答链路未就绪：" + "；".join(container.init_errors or ["向量库初始化失败"]),
        )
    assert container.qa is not None
    return container.qa


def get_ingestion(container: ServiceContainer = Depends(get_container)) -> IngestionService:
    if not container.ingest_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="入库链路未就绪：" + "；".join(container.init_errors or ["向量库初始化失败"]),
        )
    assert container.ingestion is not None
    return container.ingestion


def get_vector_store(container: ServiceContainer = Depends(get_container)) -> VectorStore:
    if container.vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="向量库未就绪：" + "；".join(container.init_errors or ["初始化失败"]),
        )
    return container.vector_store


def get_eval_service(container: ServiceContainer = Depends(get_container)) -> EvaluationService:
    if container.evaluation is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="评测服务未就绪：" + "；".join(container.init_errors or ["问答链路未初始化"]),
        )
    return container.evaluation


async def verify_api_key(request: Request) -> None:
    """可选的 API Key 校验；未配置 API_KEY 时直接放行。"""
    container: ServiceContainer = get_container(request)
    expected = container.settings.api_key
    if not expected or expected.startswith("[待补充"):
        return
    provided = request.headers.get("X-API-Key") or request.headers.get("x-api-key") or ""
    if provided != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key 缺失或不正确（当前实例已在 .env 中启用 API_KEY 校验）",
        )
