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
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.deps import (
    ServiceContainer,
    get_container,
    get_metrics,
    get_qa,
    get_session_store,
    get_vector_store,
    verify_api_key,
)
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
    metrics: Any = Depends(get_metrics),
    _: None = Depends(verify_api_key),
) -> RetrieveResponse:
    _validate_mode(payload.retrieval_mode)
    results, mode, elapsed_ms = await qa.retrieve_only(
        payload.query, top_k=payload.top_k, mode=payload.retrieval_mode
    )
    if metrics is not None:
        try:
            metrics.record(endpoint="retrieve", total_ms=elapsed_ms)
        except Exception:  # noqa: BLE001 - 观测不影响主流程
            pass
    return RetrieveResponse(
        query=payload.query,
        retrieval_mode=mode,
        latency_ms=elapsed_ms,
        results=HybridRetriever.to_debug(results, include_text=True),
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
        session_id=payload.session_id,
    )
    # retrieved_raw 是内部字段（RetrievedChunk 列表），不进 HTTP 响应
    outcome.pop("retrieved_raw", None)
    return ChatResponse(**outcome)


class SessionClearRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "要清除的会话 ID；留空表示清空**全部**会话。"
            "评测脚本在每轮开始前会调用它，否则第二轮会带着第一轮的会话历史，"
            "测出来的就不是同一条件。"
        ),
    )


@router.post("/session/clear", summary="清除会话历史（多轮评测可重复的前提）")
async def clear_session(
    payload: SessionClearRequest,
    store: Any = Depends(get_session_store),
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    if store is None:
        return {"cleared_sessions": 0, "cleared_turns": 0, "note": "会话存储未装配"}
    if payload.session_id:
        cleared_turns = store.clear(payload.session_id)
        return {"cleared_sessions": 1 if cleared_turns else 0, "cleared_turns": cleared_turns}
    cleared_sessions = 0
    cleared_turns = 0
    for session_id in store.session_ids():
        cleared_turns += store.clear(session_id)
        cleared_sessions += 1
    return {"cleared_sessions": cleared_sessions, "cleared_turns": cleared_turns}


@router.post("/cache/clear", summary="清空查询缓存（含统计复位）")
async def clear_cache(
    container: ServiceContainer = Depends(get_container),
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    """运维/测试用。正常情况下缓存由索引版本机制自动失效（入库即失效）。"""
    cache = container.query_cache
    if cache is None:
        return {"cleared_entries": 0, "note": "查询缓存未启用"}
    return {"cleared_entries": cache.clear(), "stats": cache.stats()}


@router.get("/cache/stats", summary="查询缓存统计（命中/未命中/条目数）")
async def cache_stats(
    container: ServiceContainer = Depends(get_container),
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    cache = container.query_cache
    if cache is None:
        return {"enabled": False}
    return {"enabled": True, **cache.stats()}


@router.post("/chat/stream", summary="RAG 问答（SSE 流式）")
async def chat_stream(
    payload: ChatRequest,
    qa: QAEngine = Depends(get_qa),
    metrics_collector: Any = Depends(get_metrics),
    _: None = Depends(verify_api_key),
) -> StreamingResponse:
    _validate_mode(payload.retrieval_mode)
    meta, iterator = await qa.stream_answer(
        payload.question,
        top_k=payload.top_k,
        mode=payload.retrieval_mode,
        session_id=payload.session_id,
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
            if metrics_collector is not None:
                try:
                    metrics_collector.record(
                        endpoint="chat_stream",
                        total_ms=int((time.perf_counter() - started) * 1000),
                        error=True,
                    )
                except Exception:  # noqa: BLE001
                    pass
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
        if metrics_collector is not None:
            try:
                metrics_collector.record(
                    endpoint="chat_stream",
                    total_ms=int((time.perf_counter() - started) * 1000),
                    generate_ms=elapsed_ms,
                )
            except Exception:  # noqa: BLE001
                pass
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
