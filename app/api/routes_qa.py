"""检索与问答接口：/retrieve（只召回）、/chat（完整回答）、/chat/stream（流式）。

设计要点：
- /retrieve 不调用大模型，方便你单独评估"召回质量"这一环；
- /chat 返回 citations，且对模型给出的 [n] 角标做越界校验（见 QAEngine.verify_citations）；
- /chat/stream 用 SSE 先推 meta（含 citations）再推 token；流式模式无法在服务端
  实时校验引用，因此需由前端拼接后再做一次校验（README 有说明）。
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator, List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from app.api.deps import ServiceContainer, get_container, get_qa, get_vector_store, verify_api_key
from app.core.retriever import VALID_MODES, HybridRetriever
from app.core.vector_store import VectorStore
from app.models import (
    ChatRequest,
    ChatResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from app.services.rag_pipeline import QAEngine

router = APIRouter(tags=["qa"])


def _validate_mode(mode: str | None) -> None:
    if mode and mode.lower() not in VALID_MODES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"retrieval_mode 只支持 {sorted(VALID_MODES)}，收到：{mode}",
        )


@router.post("/retrieve", response_model=RetrieveResponse, summary="只做检索，不调用大模型")
async def retrieve(
    payload: RetrieveRequest,
    qa: QAEngine = Depends(get_qa),
    _: None = Depends(verify_api_key),
) -> RetrieveResponse:
    _validate_mode(payload.retrieval_mode)
    results, mode, elapsed_ms = await qa.retrieve_only(
        payload.query, top_k=payload.top_k, mode=payload.retrieval_mode
    )
    return RetrieveResponse(
        query=payload.query,
        retrieval_mode=mode,
        latency_ms=elapsed_ms,
        results=HybridRetriever.to_debug(results),
    )


@router.post("/chat", response_model=ChatResponse, summary="RAG 问答（带引用）")
async def chat(
    payload: ChatRequest,
    qa: QAEngine = Depends(get_qa),
    _: None = Depends(verify_api_key),
) -> ChatResponse:
    _validate_mode(payload.retrieval_mode)
    outcome = await qa.answer(
        payload.question,
        top_k=payload.top_k,
        mode=payload.retrieval_mode,
        debug=payload.debug,
    )
    # retrieved_raw 是内部字段（RetrievedChunk 列表），不进 HTTP 响应
    outcome.pop("retrieved_raw", None)
    return ChatResponse(**outcome)


@router.post("/chat/stream", summary="RAG 问答（SSE 流式）")
async def chat_stream(
    payload: ChatRequest,
    qa: QAEngine = Depends(get_qa),
    _: None = Depends(verify_api_key),
) -> StreamingResponse:
    _validate_mode(payload.retrieval_mode)
    meta, iterator = await qa.stream_answer(
        payload.question,
        top_k=payload.top_k,
        mode=payload.retrieval_mode,
        debug=payload.debug,
    )

    async def event_source() -> AsyncIterator[str]:
        # 1) 先发元信息（引用来源、召回明细、模式），前端可以立刻渲染"参考来源"
        yield f"event: meta\ndata: {json.dumps(meta, ensure_ascii=False)}\n\n"
        started = time.perf_counter()
        collected: List[str] = []
        try:
            async for piece in iterator:
                collected.append(piece)
                yield f"event: token\ndata: {json.dumps({'text': piece}, ensure_ascii=False)}\n\n"
        except Exception as exc:  # 流中断时如实告知，不静默失败
            error_payload = {"message": f"{type(exc).__name__}: {exc}"}
            yield f"event: error\ndata: {json.dumps(error_payload, ensure_ascii=False)}\n\n"

        # 注意：done 事件不能放在 finally 里 yield。
        # 客户端断开时 Starlette 会 aclose 这个异步生成器，GeneratorExit 恰好会在
        # yield 处抛出，此时再 yield 就会触发 "async generator ignored GeneratorExit"。
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        done_payload = {
            "generate_ms": elapsed_ms,
            "chars": sum(len(item) for item in collected),
            "note": (
                "流式模式未做服务端引用越界校验；若需要校验后的 citations，"
                "请改用 POST /api/v1/chat。"
            ),
        }
        yield f"event: done\ndata: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 反向代理下关闭缓冲，保证逐字输出
        },
    )


@router.get("/chunks/preview", summary="预览向量库中的少量片段（人工抽查用）")
async def preview_chunks(
    vector_store: VectorStore = Depends(get_vector_store),
    container: ServiceContainer = Depends(get_container),
    limit: int = Query(default=5, ge=1, le=50),
    _: None = Depends(verify_api_key),
) -> dict:
    return {
        "collection": vector_store.collection_name,
        "count": vector_store.count(),
        "items": await vector_store.peek_async(limit),
        "notes": container.degraded_notes(),
    }
