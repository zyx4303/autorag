"""Agent 接口：基于 LangGraph 的自主工具调用问答。

接口一览（前缀 /api/v1/agent）：
  POST /agent/chat     一次性返回 {answer, citations, tool_trace, ...}
  POST /agent/chat/stream  SSE 流式推送 agent 思考与工具调用过程
  GET  /agent/tools    列出可用工具及其 JSON schema（便于前端展示与调试）
  GET  /agent/status    Agent 运行状态（可用工具、迭代上限、业务库规模）
  POST /agent/kb/rebuild  从 data/documents 重建业务工具知识库

设计说明：
- `tool_trace` 是刻意暴露的：Agent 的自主性必须可观测，否则出问题无法定位，
  也没法做评测。每一步都记录工具名、入参、是否成功、耗时、输出摘要。
- 会话用 `session_id` 关联 LangGraph checkpoint，因此支持"先追问、用户补充后继续"。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.service import AgentService, format_sse
from app.api.deps import ServiceContainer, get_container, verify_api_key
from app.logging_conf import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


# --------------------------------------------------------------------------- #
# 请求 / 响应模型
# --------------------------------------------------------------------------- #
class AgentChatRequest(BaseModel):
    session_id: str = Field(
        default="default",
        min_length=1,
        max_length=128,
        description="会话 ID。同一 ID 的多次请求共享 LangGraph checkpoint，因此支持多轮追问。",
    )
    message: str = Field(min_length=1, max_length=2000, description="用户这句话")


class ToolTraceStep(BaseModel):
    step: int
    tool: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    ok: bool
    duration_ms: int = 0
    error: str = ""
    result_count: Optional[int] = None
    output_preview: str = ""
    questions: Optional[List[str]] = None


class AgentChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    tool_trace: List[ToolTraceStep] = Field(default_factory=list)
    iterations: int = 0
    max_iterations: int = 6
    stop_reason: str = ""
    needs_user_input: bool = False
    user_questions: List[str] = Field(default_factory=list)
    latency_ms: int = 0
    tool_counts: Dict[str, int] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
def get_agent(container: ServiceContainer = Depends(get_container)) -> AgentService:
    service = container.agent
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent 未就绪：" + "；".join(container.init_errors or ["初始化失败"]),
        )
    return service


@router.post("/chat", response_model=AgentChatResponse, summary="Agent 问答（自主调用工具）")
async def agent_chat(
    payload: AgentChatRequest,
    agent: AgentService = Depends(get_agent),
    _: None = Depends(verify_api_key),
) -> AgentChatResponse:
    result = await agent.run(payload.session_id, payload.message)
    return AgentChatResponse(
        session_id=result.get("session_id", payload.session_id),
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        tool_trace=result.get("tool_trace", []),
        iterations=result.get("iterations", 0),
        max_iterations=result.get("max_iterations", 6),
        stop_reason=result.get("stop_reason", ""),
        needs_user_input=result.get("needs_user_input", False),
        user_questions=result.get("user_questions", []),
        latency_ms=result.get("latency_ms", 0),
        tool_counts=result.get("tool_counts", {}),
    )


@router.post("/chat/stream", summary="Agent 问答（SSE 流式，含工具调用过程）")
async def agent_chat_stream(
    payload: AgentChatRequest,
    agent: AgentService = Depends(get_agent),
    _: None = Depends(verify_api_key),
) -> StreamingResponse:
    async def event_source():
        async for event in agent.run_stream(payload.session_id, payload.message):
            yield format_sse(event["event"], event["data"])

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/tools", summary="列出可用工具及其 JSON schema")
async def list_tools(
    agent: AgentService = Depends(get_agent),
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    from app.agent.tools import ALL_TOOL_SPECS

    return {
        "count": len(ALL_TOOL_SPECS),
        "tools": [
            {
                "name": spec["function"]["name"],
                "description": spec["function"]["description"],
                "parameters": spec["function"]["parameters"],
            }
            for spec in ALL_TOOL_SPECS
        ],
    }


@router.get("/status", summary="Agent 运行状态")
async def agent_status(
    container: ServiceContainer = Depends(get_container),
    agent: AgentService = Depends(get_agent),
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    return {
        "ready": True,
        "framework": "langgraph",
        "tools": agent.tool_names,
        "max_iterations": agent.max_iterations,
        "search_kb_available": agent.search_kb_available,
        "kb": agent.kb_store.stats(),
        "checkpoint_path": str(container.settings.agent_checkpoint_path),
        "init_errors": container.init_errors,
    }


@router.post("/kb/rebuild", summary="从 data/documents 重建 Agent 业务知识库")
async def rebuild_kb(
    force: bool = False,
    agent: AgentService = Depends(get_agent),
    _: None = Depends(verify_api_key),
) -> Dict[str, Any]:
    result = agent.bootstrap_kb(force=force)
    return result
