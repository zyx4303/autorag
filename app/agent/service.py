"""Agent 服务层：对接 FastAPI 与 SSE 流式输出。

对外两种调用方式：
  run()          一次性返回完整结果（含 tool_trace，供调试与评测）
  run_stream()   边执行边推送事件：node / tool_start / tool_end / answer / done

关于 checkpoint 的双通道设计（实测踩到的坑，写在这里避免以后误判）：
  · LangGraph 的 SqliteSaver 内部维护自己的连接，
    如果和业务查询共用同一个 sqlite3.Connection，首次执行会报
    "Cannot operate on a closed database" —— 因为 saver 会接管并关闭连接。
  · 因此这里给 checkpointer 单独开一个连接（独立文件句柄），
    业务侧的 AgentKbStore 继续用自己的连接，两者互不影响。
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from app.agent.graph import AGENT_MAX_ITERATIONS, AgentGraph
from app.agent.kb import AgentKbStore, bootstrap_from_documents
from app.agent.tools import ToolRegistry
from app.config import Settings
from app.core.llm_client import LLMClient
from app.core.retriever import HybridRetriever
from app.logging_conf import get_logger

logger = get_logger(__name__)


class AgentService:
    """把工具注册表 + 图 + checkpoint 组装好，并负责事件流。"""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        retriever: Optional[HybridRetriever],
        kb_store: AgentKbStore,
        checkpointer: Any = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._kb = kb_store
        # 检索工具依赖向量库；向量库不可用时仍允许 lookup_dtc / compute_maintenance，
        # 但要如实告知 search_kb 不可用，而不是静默降级
        self._retriever = retriever
        # checkpoint 采用「延迟初始化」：
        # LangGraph 的异步图要求异步 Saver（AsyncSqliteSaver），而它的初始化本身是异步的，
        # 无法在同步的容器构建阶段完成，因此放到第一次请求时在事件循环里建好并缓存。
        self._checkpointer = checkpointer
        self._saver_ctx: Any = None
        self._saver_lock = asyncio.Lock()
        self._graph: Optional[AgentGraph] = None
        self._tools = ToolRegistry(retriever, kb_store)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    async def _ensure_graph(self) -> AgentGraph:
        """确保图已构建。首次调用时初始化异步 checkpoint。"""
        if self._graph is not None:
            return self._graph
        async with self._saver_lock:
            if self._graph is not None:
                return self._graph
            checkpointer = self._checkpointer
            if checkpointer is None:
                try:
                    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

                    self._settings.agent_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    ctx = AsyncSqliteSaver.from_conn_string(str(self._settings.agent_checkpoint_path))
                    checkpointer = await ctx.__aenter__()
                    self._saver_ctx = ctx
                    logger.info("Agent checkpoint 使用 SQLite：%s", self._settings.agent_checkpoint_path)
                except Exception as exc:  # 退化为内存保存，但要如实记录
                    logger.warning("AsyncSqliteSaver 初始化失败，退化为内存保存：%s", exc)
                    from langgraph.checkpoint.memory import MemorySaver

                    checkpointer = MemorySaver()
            self._graph = AgentGraph(
                self._settings, self._llm, self._tools, checkpointer=checkpointer
            )
            return self._graph

    async def aclose(self) -> None:
        """释放 checkpoint 的异步连接（应用关闭时调用）。"""
        if self._saver_ctx is not None:
            try:
                await self._saver_ctx.__aexit__(None, None, None)
            except Exception:  # 关闭失败不影响退出
                logger.debug("关闭 AsyncSqliteSaver 时出现异常（已忽略）")
            self._saver_ctx = None
            self._graph = None

    # ------------------------------------------------------------------ #
    @property
    def kb_store(self) -> AgentKbStore:
        return self._kb

    @property
    def tool_names(self) -> List[str]:
        return self._tools.names

    @property
    def max_iterations(self) -> int:
        return AGENT_MAX_ITERATIONS

    @property
    def search_kb_available(self) -> bool:
        return self._retriever is not None

    @property
    def tool_counts(self) -> Dict[str, int]:
        return dict(self._tools.call_counts)

    def bootstrap_kb(self, force: bool = False) -> Dict[str, Any]:
        """从 data/documents 建业务工具知识库（幂等）。"""
        if not force and not self._kb.is_empty():
            return {"skipped": True, "stats": self._kb.stats()}
        result = bootstrap_from_documents(self._kb, self._settings.documents_dir)
        result["skipped"] = False
        return result

    # ------------------------------------------------------------------ #
    async def run(self, session_id: str, message: str) -> Dict[str, Any]:
        """执行一轮 Agent。返回结构化结果，供 /agent/chat 直接序列化。"""
        started = time.perf_counter()
        if not self.search_kb_available:
            logger.warning("向量检索不可用，search_kb 调用会返回可行动的错误提示")
        graph = await self._ensure_graph()
        result = await graph.ainvoke(message, session_id)
        result["session_id"] = session_id
        result["latency_ms"] = result.get("latency_ms") or int((time.perf_counter() - started) * 1000)
        result["max_iterations"] = AGENT_MAX_ITERATIONS
        return result

    # ------------------------------------------------------------------ #
    async def run_stream(self, session_id: str, message: str) -> AsyncIterator[Dict[str, Any]]:
        """SSE 事件流。

        事件类型：
          start       开始
          node        进入某个节点（agent / tools / finalize）
          tool_start  某个工具开始执行（含入参）
          tool_end    某个工具执行结束（含是否成功、耗时、结果条数）
          answer      最终答案（一次性推送，便于前端立即渲染）
          done        结束（含停止原因、迭代次数、耗时）
          error       出错
        """
        queue: asyncio.Queue = asyncio.Queue()
        started = time.perf_counter()

        yield {"event": "start", "data": {"session_id": session_id, "message": message,
                                          "max_iterations": AGENT_MAX_ITERATIONS,
                                          "tools": self._tools.names}}

        async def produce() -> None:
            try:
                graph = await self._ensure_graph()
                config = {"configurable": {"thread_id": session_id}}
                self._tools.reset_counts()
                async for chunk in graph.astream(message, config):
                    await queue.put(chunk)
            except Exception as exc:  # 采集失败也要让前端收到 error 事件
                logger.exception("SSE 图执行失败")
                await queue.put({"__error__": f"{type(exc).__name__}: {exc}"})
            finally:
                await queue.put(None)

        task = asyncio.create_task(produce())
        iteration = 0
        answered = False
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                if isinstance(event, dict) and "__error__" in event:
                    yield {"event": "error", "data": {"message": event["__error__"]}}
                    break
                name = event.get("event")
                data = event.get("data") or {}
                if name == "node" and data.get("node") == "agent":
                    iteration = int(data.get("iteration") or iteration)
                if name == "answer":
                    answered = True
                yield {"event": name, "data": data}
        finally:
            if not task.done():
                task.cancel()
            else:
                await task
            # 兜底：若 finalize 未推送 answer（例如图异常），补一个空 answer 事件，
            # 保证前端无论走哪条路径都能收到 answer 与 done
            if not answered:
                yield {
                    "event": "answer",
                    "data": {"answer": "", "stop_reason": "no_answer", "needs_user_input": False,
                             "user_questions": []},
                }
            yield {
                "event": "done",
                "data": {
                    "iterations": iteration,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "tool_counts": self.tools_counts_snapshot(),
                },
            }

    def tools_counts_snapshot(self) -> Dict[str, int]:
        return dict(self._tools.call_counts)


def format_sse(event: str, data: Dict[str, Any]) -> str:
    """把事件编码成 SSE 帧。JSON 里的换行会被 sse-starlette 处理，这里统一 ensure_ascii=False。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
