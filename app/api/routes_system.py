"""健康检查与统计。

这里的每个布尔量都来自真实的运行时检查：
- llm_ready         : /health?probe_llm=true 时才会真正发一次极短请求（会消耗 token）
- embedding_ready   : embedder 是否构造成功
- vector_store_ready: 能否读到 Chroma 集合的 count
"""
from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.api.deps import ServiceContainer, get_container, get_vector_store, verify_api_key
from app.config import PROJECT_ROOT
from app.core.loader import scan_documents
from app.core.text_utils import sha256_text
from app.core.vector_store import VectorStore
from app.models import DeleteResponse, HealthResponse, StatsResponse

router = APIRouter(tags=["system"])

API_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse, summary="健康检查")
async def health(
    request: Request,
    probe_llm: bool = Query(
        default=False,
        description="true 时会真实调用一次 LLM（max_tokens=1）验证连通性，会产生极少量 token 消耗",
    ),
) -> HealthResponse:
    container: ServiceContainer = get_container(request)
    settings = container.settings

    llm_detail = (
        {"ok": True, "detail": "未探测（probe_llm=false），仅校验密钥是否已配置"}
        if not probe_llm
        else await container.llm.health_check()
    )
    if not probe_llm:
        llm_detail["ok"] = container.llm.configured
        if not container.llm.configured:
            llm_detail["detail"] = "LLM_API_KEY 未配置（空或 [待补充]）"

    vector_ready = container.vector_store.health() if container.vector_store else False
    collection_count = container.vector_store.count() if container.vector_store else 0

    status_value = "ok"
    if not vector_ready or not container.embedder:
        status_value = "degraded"
    if not container.llm.configured:
        status_value = "degraded"

    return HealthResponse(
        status=status_value,
        version=API_VERSION,
        llm_ready=bool(llm_detail.get("ok")),
        embedding_ready=container.embedder is not None,
        vector_store_ready=vector_ready,
        collection_count=collection_count,
        detail={
            "llm": llm_detail,
            "embedding": container.embedder.describe() if container.embedder else None,
            "bm25": container.bm25.describe() if container.bm25 else None,
            "config": settings.public_summary(),
            "notes": container.degraded_notes(),
            "project_root": str(PROJECT_ROOT),
        },
    )


@router.get("/stats", response_model=StatsResponse, summary="知识库统计")
async def stats(
    request: Request,
    _: None = Depends(verify_api_key),
) -> StatsResponse:
    container: ServiceContainer = get_container(request)
    settings = container.settings
    vector_store = container.vector_store
    if vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="向量库未就绪：" + "；".join(container.init_errors or ["初始化失败"]),
        )

    registry_summary = (
        container.ingestion.registry_summary() if container.ingestion else {"indexed_documents": 0}
    )
    files, _skipped = scan_documents(settings.documents_dir)

    return StatsResponse(
        collection_count=vector_store.count(),
        indexed_documents=int(registry_summary.get("indexed_documents") or 0),
        total_chunks_expected=int(registry_summary.get("total_chunks_expected") or 0),
        bm25_docs=container.bm25.size if container.bm25 else 0,
        retrieval_mode=settings.retrieval_mode,
        embedding_provider=settings.embedding_provider,
        embedding_dim=container.embedder.dimension if container.embedder else 0,
        llm_model=settings.llm_model,
        documents_dir=str(settings.documents_dir),
        documents_on_disk=len(files),
        config_warnings=container.degraded_notes(),
        last_ingest=(container.ingestion.last_result if container.ingestion else None)
        or registry_summary.get("last_run"),
    )


@router.get("/sources", summary="列出向量库中已有的来源文件")
async def sources(
    vector_store: VectorStore = Depends(get_vector_store),
    _: None = Depends(verify_api_key),
) -> dict:
    sources_list = vector_store.sources()
    return {"count": len(sources_list), "sources": sources_list}


@router.delete("/collection", response_model=DeleteResponse, summary="清空向量库与关键词索引")
async def clear_collection(
    request: Request,
    vector_store: VectorStore = Depends(get_vector_store),
    _: None = Depends(verify_api_key),
) -> DeleteResponse:
    container: ServiceContainer = get_container(request)
    deleted = await asyncio.to_thread(vector_store.reset)
    if container.bm25:
        container.bm25.clear()
    return DeleteResponse(
        collection=vector_store.collection_name,
        deleted_vectors=deleted,
        cleared_bm25=True,
        message=(
            f"已清空集合 {vector_store.collection_name}（删除 {deleted} 条向量）并重置 BM25 索引。"
            "注意：data/registry.json 仍保留增量记录，如需完全重来请调用 "
            "POST /api/v1/ingest 并带 rebuild=true、reset_registry=true。"
        ),
    )


@router.get("/version", summary="版本与构建信息")
async def version(request: Request) -> dict:
    container: ServiceContainer = get_container(request)
    return {
        "version": API_VERSION,
        "started_config_fingerprint": sha256_text(
            str(container.settings.public_summary())
        )[:12],
        "timestamp": int(time.time()),
    }
