"""LangGraph 状态图：把固定 pipeline 的 RAG 升级为「LLM 在循环里自主决定下一步」的 Agent。

图结构（显式状态图，不用 LangChain 的链式抽象）：

        ┌──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
    ┌────────┐  模型要求调工具    ┌─────────┐           │
    │ agent  │ ────────────────▶ │  tools  │ ──────────┘
    │ (LLM)  │                   │ (执行)  │
    └────────┘                   └─────────┘
        │
        │ 模型直接给出最终答案 / 达到迭代上限 / 需要追问用户
        ▼
    ┌──────────┐
    │ finalize │ ──▶ END
    └──────────┘

停止条件（三道，任一触发即收尾）：
  1. 模型不再请求工具调用 → 直接进入 finalize；
  2. iteration_count 达到 AGENT_MAX_ITERATIONS（默认 6）→ 强制收尾并如实告知用户；
  3. 模型调用 ask_user → 进入 finalize 并把问题交回用户（等下一轮用户消息）。

关于 checkpoint：使用 LangGraph 的 SqliteSaver，按 thread_id（即 session_id）持久化。
用户追问后再次发消息，图会从上次状态继续，因此"补充信息后继续原来的任务"是天然支持的。
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Annotated, Any, AsyncIterator, Dict, List, Optional, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from app.agent.tools import ToolRegistry, ToolResult
from app.config import Settings
from app.core.llm_client import LLMClient, LLMError
from app.logging_conf import get_logger

logger = get_logger(__name__)

# 迭代上限：6 次（需求指定）。超过后强制收尾，避免无限调用工具烧 token。
AGENT_MAX_ITERATIONS = 6

AGENT_SYSTEM_PROMPT = """你是汽车售后诊断助手，通过调用工具获取事实依据，再给出可核对的答案。

## 工作方式
1. 先判断信息是否足够。若关键信息缺失（例如用户问"该保养了吗"却没说车型和里程、
   问"故障灯亮了怎么办"却没给故障码），**第一步必须调用 ask_user 追问**，不要猜。
2. 根据问题类型选择工具：
   - 知识型问题（保养周期、维修步骤、条款解释）→ search_kb
   - 明确的故障码（P0195 这类）→ 先 lookup_dtc 拿定义，**再用 search_kb 检索相关文档**
   - 按里程算保养清单 → compute_maintenance（必须同时有车型和里程）
3. **组合调用规则（很重要，直接用工具结果作答往往信息不够）**：
   - 用户问"XX 故障码什么意思"时：lookup_dtc 只给出码值定义与可能原因线索，
     用户想知道的通常还包括影响、检修方向、相关码。因此拿到定义后**必须再调用一次
     search_kb**（查询词用「码值 + 该码的描述关键词」，例如「P0195 机油温度传感器」），
     把码表文档中的解释性内容一并纳入答案。
   - 用 compute_maintenance 算出保养项目后，若用户还问了"为什么"或"要注意什么"，
     再用 search_kb 检索对应章节补充说明。
   - 工具返回明确提示"库中没有"时，按提示换参数或换工具，不要停在中间状态。
4. 工具返回后，判断信息是否足以回答。不足就换关键词/换工具再查；足够就直接给出最终答案。
   **收敛原则（重要）**：如果上一次 search_kb 返回的片段已经能直接回答问题，
   就不要再换近义词重复检索，也不要一次并行发起多个 search_kb——直接把答案写出来。
   判断标准是「现有片段是否已包含回答问题所需的事实」，而不是「是否还有更多相关内容」。
   重复检索只会增加耗时与 token，不会提升答案质量。

## 输出要求（重要）
- 只依据工具返回的内容作答。**工具没给的数据一律不许编造**，包括扭矩值、保养价格、
  未入库车型的参数。查不到就明说"知识库中没有相关依据"，并给出可行的确认渠道。
- 每条结论后面用 [序号] 标注来源，序号必须来自 search_kb / lookup_dtc 返回的 index。
- 涉及示例/虚构数据时必须点明，例如"（注：该车型为示例数据，不对应真实车型）"。
- 用中文回答，直接给结论，不要复述工具返回的原始 JSON。

## 边界
- 同一轮对话最多调用 6 次工具，请优先用最少的调用拿到需要的信息；
  但"少调用"不能以牺牲信息完整性为代价——该补检索时就要补。
- 如果工具报错，先看错误提示里的建议（换参数、换工具、或如实告知用户），不要重复同样的调用。"""

ASK_USER_MARKER = "__ASK_USER__"


class AgentState(TypedDict, total=False):
    """图状态。字段与需求一致，另加少量运行观测字段。

    messages 使用 add_messages reducer：节点返回的增量消息会自动追加到历史。
    """

    messages: Annotated[List[Any], add_messages]
    retrieved_docs: List[Dict[str, Any]]      # 本次会话累计检索到的片段（去重后）
    current_tool_calls: List[Dict[str, Any]]  # 本轮待执行的工具调用
    final_answer: str
    iteration_count: int
    tool_trace: List[Dict[str, Any]]          # 每步工具调用的入参出参（透明性）
    citations: List[Dict[str, Any]]           # 汇总引用
    needs_user_input: bool                    # 是否在等用户补充信息
    user_questions: List[str]                 # 要问用户的问题
    stop_reason: str                          # 收尾原因：answered / max_iterations / ask_user / error
    error: str


# --------------------------------------------------------------------------- #
# 工具调用解析（兼容不同网关的返回形态）
# --------------------------------------------------------------------------- #
def parse_tool_calls(raw_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把 OpenAI 格式的 tool_calls 规范成 [{id, name, arguments(dict)}]。

    容错点（实测不同兼容网关的差异）：
    - arguments 可能是 JSON 字符串，也可能是已经是 dict；
    - arguments 可能是非法 JSON（模型写坏了），此时退化为空参数并保留原文本，
      由工具层返回"参数不正确"的可行动提示，而不是让图崩掉。
    """
    parsed: List[Dict[str, Any]] = []
    for index, call in enumerate(raw_calls or []):
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        raw_args = function.get("arguments")
        arguments: Dict[str, Any] = {}
        parse_error = ""
        if isinstance(raw_args, dict):
            arguments = dict(raw_args)
        elif isinstance(raw_args, str) and raw_args.strip():
            try:
                loaded = json.loads(raw_args)
                if isinstance(loaded, dict):
                    arguments = loaded
                else:
                    parse_error = f"arguments 不是 JSON 对象：{raw_args[:120]}"
            except json.JSONDecodeError as exc:
                parse_error = f"arguments 不是合法 JSON（{exc}）：{raw_args[:120]}"
        parsed.append(
            {
                "id": str(call.get("id") or f"call_{index}"),
                "name": name,
                "arguments": arguments,
                "arguments_raw": raw_args if isinstance(raw_args, str) else json.dumps(raw_args or {}, ensure_ascii=False),
                "parse_error": parse_error,
            }
        )
    return parsed


# --------------------------------------------------------------------------- #
# Agent 图
# --------------------------------------------------------------------------- #
def _emit(writer: Any, payload: Dict[str, Any]) -> None:
    """向前端推送一个进度事件（tool_start / tool_result）。

    兼容三种情况（不同 LangGraph 版本的注入方式不同）：
      1. 节点签名里显式接收了 writer（LangGraph 0.6+ 支持的写法）；
      2. 用 get_stream_writer() 取到 writer（较早版本的写法）；
      3. 两者都没有（图不是以 custom 流模式运行）——静默跳过，
         绝不能因为"推不出事件"而让主流程报错。
    """
    target = writer
    if target is None:
        try:
            from langgraph.config import get_stream_writer

            target = get_stream_writer()
        except Exception:
            target = None
    if target is None:
        return
    try:
        target(payload)
    except Exception:  # 推送失败不影响主流程
        logger.debug("推送流式事件失败（已忽略）：%s", payload.get("kind"))


class AgentGraph:
    """把工具注册表 + LLM 组装成 LangGraph 状态图。"""

    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        tools: ToolRegistry,
        checkpointer: Optional[Any] = None,
        max_iterations: int = AGENT_MAX_ITERATIONS,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._tools = tools
        self._max_iterations = max(1, int(max_iterations))
        self._checkpointer = checkpointer or MemorySaver()
        self._graph = self._build()

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    @property
    def tool_names(self) -> List[str]:
        return self._tools.names

    # ------------------------------------------------------------------ #
    def _build(self):
        builder = StateGraph(AgentState)
        builder.add_node("agent", self._agent_node)
        builder.add_node("tools", self._tools_node)
        builder.add_node("finalize", self._finalize_node)

        builder.set_entry_point("agent")
        builder.add_conditional_edges(
            "agent",
            self._route_after_agent,
            {"tools": "tools", "finalize": "finalize"},
        )
        builder.add_edge("tools", "agent")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self._checkpointer)

    # ------------------------------------------------------------------ #
    # 节点 1：agent —— 调 LLM，决定"调工具"还是"直接回答"
    # ------------------------------------------------------------------ #
    async def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        iteration = int(state.get("iteration_count", 0)) + 1
        messages = list(state.get("messages") or [])

        # 为调用 LLM 准备消息：插一句系统提示词（不写进状态，避免历史里重复堆积）
        payload_messages: List[Dict[str, Any]] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        for message in messages:
            payload_messages.append(self._to_llm_message(message))

        # 接近上限时不再下发工具，逼模型用已有信息收尾
        remaining = self._max_iterations - int(state.get("iteration_count", 0))
        allow_tools = remaining > 1
        tools = self._tools.specs if allow_tools else None

        try:
            result = await self._llm.chat(payload_messages, tools=tools, temperature=0.0)
        except LLMError as exc:
            logger.warning("agent 节点 LLM 调用失败：%s", exc)
            return {
                "iteration_count": iteration,
                "final_answer": "",
                "error": f"LLM 调用失败：{exc}",
                "stop_reason": "error",
                "current_tool_calls": [],
            }

        calls = parse_tool_calls(result.tool_calls)
        if calls and allow_tools:
            return {
                "messages": [self._assistant_message(result)],
                "current_tool_calls": calls,
                "iteration_count": iteration,
            }

        # 模型给出最终答案（或已不允许调工具）
        answer = result.text.strip()
        if not answer and calls and not allow_tools:
            # 触到上限但模型仍想调工具：不再执行，交给 finalize 生成受限收尾说明
            answer = ""
        return {
            "messages": [self._assistant_message(result)] if result.text else [],
            "final_answer": answer,
            "current_tool_calls": [],
            "iteration_count": iteration,
        }

    def _route_after_agent(self, state: AgentState) -> str:
        if state.get("stop_reason") == "error":
            return "finalize"
        if state.get("current_tool_calls"):
            return "tools"
        return "finalize"

    # ------------------------------------------------------------------ #
    # 节点 2：tools —— 执行工具，把结果与 trace 写回状态
    # ------------------------------------------------------------------ #
    async def _tools_node(self, state: AgentState, writer: Any = None) -> Dict[str, Any]:
        """执行工具，并通过 StreamWriter 推送进度事件。

        为什么用 StreamWriter 而不是 yield：节点里 `yield` 出来的内容会被 LangGraph
        当作"状态增量"处理，非状态字段（如 tool_start）会被直接丢弃——
        实测 astream 只收到了 {'tools': {...状态字段...}}，中间事件全部丢失。
        正确做法是 writer(...) + stream_mode=["updates", "custom"]。
        """
        calls = list(state.get("current_tool_calls") or [])
        trace: List[Dict[str, Any]] = list(state.get("tool_trace") or [])
        retrieved: List[Dict[str, Any]] = list(state.get("retrieved_docs") or [])
        citations: List[Dict[str, Any]] = list(state.get("citations") or [])
        new_messages: List[Dict[str, Any]] = []
        user_questions: List[str] = []
        needs_user = False

        for call in calls:
            name = call["name"]
            arguments = dict(call.get("arguments") or {})

            _emit(writer, {
                    "kind": "tool_start",
                    "step": len(trace) + 1,
                    "tool": name,
                    "arguments": arguments,
                })

            if call.get("parse_error"):
                result = ToolResult(
                    ok=False,
                    content=(
                        f"参数解析失败：{call['parse_error']}。"
                        f"请重新发起一次调用，确保 arguments 是合法 JSON 对象。"
                    ),
                    error="bad_arguments",
                )
            else:
                result = await self._tools.execute(name, arguments)

            step: Dict[str, Any] = {
                "step": len(trace) + 1,
                "tool": name,
                "arguments": arguments,
                "ok": result.ok,
                "duration_ms": result.duration_ms,
                "error": result.error,
                "output_preview": result.content[:800],
            }

            if name == "search_kb":
                results = (result.data or {}).get("results") or []
                step["result_count"] = len(results)
                step["output"] = {"results": results}
                for item in results:
                    retrieved.append(item)
                for citation in result.citations:
                    if not any(c.get("chunk_id") == citation.get("chunk_id") for c in citations):
                        citations.append(citation)
            elif name == "lookup_dtc":
                step["output"] = result.data
                for citation in result.citations:
                    citations.append(citation)
            elif name == "compute_maintenance":
                data = result.data or {}
                step["output"] = {
                    "due_items": data.get("due_items"),
                    "upcoming_items": data.get("upcoming_items"),
                    "supported_models": data.get("supported_models"),
                }
                for citation in result.citations:
                    citations.append(citation)
            elif name == "ask_user":
                needs_user = True
                user_questions.extend((result.data or {}).get("questions") or [])
                step["questions"] = (result.data or {}).get("questions")
                step["output"] = result.data
            trace.append(step)

            _emit(writer, {
                    "kind": "tool_result",
                    "step": step["step"],
                    "tool": name,
                    "arguments": arguments,
                    "ok": result.ok,
                    "duration_ms": result.duration_ms,
                    "error": result.error,
                    "result_count": step.get("result_count"),
                    "questions": step.get("questions"),
                    "output_preview": step["output_preview"],
                })

            new_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": result.content,
                }
            )

        update: Dict[str, Any] = {
            "messages": new_messages,
            "tool_trace": trace,
            "retrieved_docs": self._dedupe_docs(retrieved),
            "citations": citations,
            "current_tool_calls": [],
        }
        if needs_user:
            update["needs_user_input"] = True
            update["user_questions"] = user_questions
            update["stop_reason"] = "ask_user"
        return update

    # ------------------------------------------------------------------ #
    # 节点 3：finalize —— 收尾（正常回答 / 追问 / 上限 / 报错）
    # ------------------------------------------------------------------ #
    async def _finalize_node(self, state: AgentState) -> Dict[str, Any]:
        already = (state.get("final_answer") or "").strip()
        reason = state.get("stop_reason") or ""
        questions = list(state.get("user_questions") or [])
        iteration = int(state.get("iteration_count", 0))

        # 需要用户补充信息：直接把问题作为回答返回，并标记等待
        if state.get("needs_user_input"):
            text = self._compose_ask_text(state, questions)
            return {
                "final_answer": text,
                "stop_reason": "ask_user",
                "needs_user_input": True,
                "user_questions": questions,
            }

        if reason == "error":
            error = state.get("error") or "未知错误"
            return {
                "final_answer": (
                    f"抱歉，本次请求在调用大模型时失败：{error}。\n"
                    f"已获取的工具结果仍然可用，你可以稍后重试或改用 /api/v1/chat 接口（不带 Agent 编排）。"
                ),
                "stop_reason": "error",
            }

        if already:
            return {"final_answer": already, "stop_reason": reason or "answered"}

        # 达到迭代上限且模型未给出文本：强制收尾，如实说明
        if iteration >= self._max_iterations:
            summary = self._summarize_tool_outputs(state)
            return {
                "final_answer": (
                    f"已达到本轮工具调用上限（{self._max_iterations} 次），为避免无意义消耗先停下。\n"
                    f"以下是已获得的核查结果，供你参考：\n\n{summary}\n\n"
                    f"如果还需要更准确的结论，可以把问题拆得更具体一些再问一次"
                    f"（例如直接给出故障码、车型与里程）。"
                ),
                "stop_reason": "max_iterations",
            }

        # 兜底：模型既没给文本也没要求工具
        return {
            "final_answer": "本次没有得到可用的回答，请把问题描述得更具体一些再试一次。",
            "stop_reason": "empty",
        }

    # ------------------------------------------------------------------ #
    # 辅助
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_llm_message(message: Any) -> Dict[str, Any]:
        """把状态里的消息转成可直接发给 OpenAI 兼容接口的格式。

        重要（实测踩坑）：LangGraph 的 add_messages reducer 会把我们写入的普通 dict
        自动转换成 LangChain 的 AIMessage / ToolMessage 对象。转发给 LLM 前必须做两件事：
          1. tool 消息必须带回 tool_call_id，否则 DeepSeek 等网关直接报 422
             "messages[i]: missing field `tool_call_id`"；
          2. AIMessage.tool_calls 用的是 {name, args, id} 结构，
             需转回 OpenAI 的 {"id","type":"function","function":{"name","arguments"(JSON 字符串)}}，
             否则模型无法把 assistant 的调用意图与 tool 结果对应起来。
        """
        if isinstance(message, dict):
            role = message.get("role", "user")
            out: Dict[str, Any] = {"role": role, "content": message.get("content", "")}
            if message.get("tool_calls"):
                out["tool_calls"] = message["tool_calls"]
            if message.get("tool_call_id"):
                out["tool_call_id"] = message["tool_call_id"]
            if message.get("name"):
                out["name"] = message["name"]
            return out

        # BaseMessage 形态（LangGraph 会把 dict 转成这些对象）
        role = getattr(message, "type", "user")
        role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
        out = {"role": role_map.get(role, role), "content": getattr(message, "content", "") or ""}

        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            out["tool_call_id"] = str(tool_call_id)
        name = getattr(message, "name", None)
        if name:
            out["name"] = str(name)

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            converted: List[Dict[str, Any]] = []
            for index, call in enumerate(tool_calls):
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if isinstance(function, dict):
                    # 已经是 OpenAI 形态
                    converted.append(
                        {
                            "id": str(call.get("id") or f"call_{index}"),
                            "type": "function",
                            "function": {
                                "name": function.get("name", ""),
                                "arguments": function.get("arguments")
                                if isinstance(function.get("arguments"), str)
                                else json.dumps(function.get("arguments") or {}, ensure_ascii=False),
                            },
                        }
                    )
                    continue
                # LangChain 形态：{name, args, id}
                converted.append(
                    {
                        "id": str(call.get("id") or f"call_{index}"),
                        "type": "function",
                        "function": {
                            "name": str(call.get("name") or ""),
                            "arguments": json.dumps(call.get("args") or {}, ensure_ascii=False),
                        },
                    }
                )
            if converted:
                out["tool_calls"] = converted
        return out

    @staticmethod
    def _assistant_message(result: Any) -> Dict[str, Any]:
        """构造要写回状态的 assistant 消息。

        必须带上 tool_calls 原文，否则下一轮 LLM 看不到自己刚才要求调用工具，
        会重复调用或直接放弃（这是 Function Calling 多轮的标准要求）。
        """
        message: Dict[str, Any] = {"role": "assistant", "content": result.text or ""}
        if result.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.get("id") or f"call_{index}",
                    "type": "function",
                    "function": call.get("function") or {},
                }
                for index, call in enumerate(result.tool_calls)
                if isinstance(call, dict)
            ]
        return message

    @staticmethod
    def _dedupe_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique: List[Dict[str, Any]] = []
        for doc in docs:
            key = (doc.get("source"), doc.get("position"), (doc.get("text") or "")[:60])
            if key in seen:
                continue
            seen.add(key)
            unique.append(doc)
        return unique

    @staticmethod
    def _summarize_tool_outputs(state: AgentState) -> str:
        """把已获得的工具结果整理成人可以读的摘要（用于上限收尾与错误兜底）。"""
        trace = state.get("tool_trace") or []
        if not trace:
            return "（本轮没有成功获取到工具结果）"
        lines: List[str] = []
        for step in trace:
            if step.get("tool") == "search_kb" and step.get("ok"):
                for item in (step.get("output") or {}).get("results") or []:
                    lines.append(
                        f"- [{item.get('index')}] {item.get('source')}"
                        f"（{item.get('section') or '无章节'}）：{(item.get('text') or '')[:160]}…"
                    )
            elif step.get("tool") == "lookup_dtc" and step.get("ok"):
                for item in (step.get("output") or {}).get("results") or []:
                    lines.append(f"- {item.get('code')}：{item.get('description')}（来源：{item.get('source')}）")
            elif step.get("tool") == "compute_maintenance" and step.get("ok"):
                items = (step.get("output") or {}).get("due_items") or []
                lines.append(f"- 按里程判定到期的保养项目：{len(items)} 项")
            elif not step.get("ok"):
                lines.append(f"- 工具 {step.get('tool')} 未成功：{step.get('error') or ''}")
        return "\n".join(lines) if lines else "（工具返回了结果，但未提取到可读摘要）"

    @staticmethod
    def _compose_ask_text(state: AgentState, questions: List[str]) -> str:
        reason = ""
        for step in state.get("tool_trace") or []:
            if step.get("tool") == "ask_user":
                reason = (step.get("output") or {}).get("reason") or ""
        lines = ["为了给你准确的答案，还需要确认以下信息："]
        for index, question in enumerate(questions, start=1):
            lines.append(f"{index}. {question}")
        if reason:
            lines.append(f"\n（为什么需要：{reason}）")
        lines.append("\n补充后我会继续为你查询。")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 对外入口
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # 流式执行（供 SSE 使用）
    # ------------------------------------------------------------------ #
    async def astream(self, message: str, config: Dict[str, Any]) -> AsyncIterator[Dict[str, Any]]:
        """边执行边产出事件，供上层转成 SSE。

        使用 stream_mode=["updates", "custom"]：
          · "updates" → 每个节点执行完的状态增量（用于 node 事件）
          · "custom"  → 节点内通过 StreamWriter 主动推送的进度事件（tool_start / tool_result）
        产出统一规范成 {"event": 事件名, "data": {...}}，上层不用再关心 LangGraph 的细节。
        """
        self._tools.reset_counts()
        initial: AgentState = {
            "messages": [{"role": "user", "content": message}],
            "iteration_count": 0,
            "tool_trace": [],
            "retrieved_docs": [],
            "citations": [],
            "current_tool_calls": [],
            "final_answer": "",
            "needs_user_input": False,
            "user_questions": [],
            "stop_reason": "",
            "error": "",
        }
        async for mode, chunk in self._graph.astream(
            initial, config=config, stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                if isinstance(chunk, dict):
                    kind = chunk.get("kind")
                    payload = {k: v for k, v in chunk.items() if k != "kind"}
                    if kind:
                        yield {"event": kind, "data": payload}
                continue
            # mode == "updates"
            if not isinstance(chunk, dict):
                continue
            for node, update in chunk.items():
                if not isinstance(update, dict):
                    continue
                if node == "agent":
                    yield {
                        "event": "node",
                        "data": {
                            "node": "agent",
                            "iteration": int(update.get("iteration_count") or 0),
                            "decided": "调用工具" if update.get("current_tool_calls") else "给出回答",
                            "planned_tools": [
                                c.get("name") for c in (update.get("current_tool_calls") or [])
                            ],
                        },
                    }
                elif node == "tools":
                    yield {
                        "event": "node",
                        "data": {
                            "node": "tools",
                            "executed": len(update.get("tool_trace") or []),
                        },
                    }
                elif node == "finalize":
                    yield {
                        "event": "answer",
                        "data": {
                            "answer": update.get("final_answer") or "",
                            "stop_reason": update.get("stop_reason") or "",
                            "needs_user_input": bool(update.get("needs_user_input")),
                            "user_questions": update.get("user_questions") or [],
                        },
                    }

    async def ainvoke(
        self,
        message: str,
        session_id: str,
    ) -> Dict[str, Any]:
        """跑一轮。session_id 即 LangGraph 的 thread_id，用于 checkpoint 续接。"""
        self._tools.reset_counts()
        started = time.perf_counter()
        config = {"configurable": {"thread_id": session_id}}
        initial: AgentState = {
            "messages": [{"role": "user", "content": message}],
            "iteration_count": 0,
            "tool_trace": [],
            "retrieved_docs": [],
            "citations": [],
            "current_tool_calls": [],
            "final_answer": "",
            "needs_user_input": False,
            "user_questions": [],
            "stop_reason": "",
            "error": "",
        }
        try:
            final_state = await self._graph.ainvoke(initial, config=config)
        except Exception as exc:  # 图执行异常也要给出结构化结果，不能把 500 抛给前端
            logger.exception("Agent 图执行失败")
            return {
                "answer": f"Agent 执行失败：{type(exc).__name__}: {exc}",
                "citations": [],
                "tool_trace": [],
                "iterations": 0,
                "stop_reason": "graph_error",
                "needs_user_input": False,
                "user_questions": [],
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "tool_counts": dict(self._tools.call_counts),
            }

        return {
            "answer": final_state.get("final_answer") or "",
            "citations": final_state.get("citations") or [],
            "tool_trace": final_state.get("tool_trace") or [],
            "retrieved_docs": final_state.get("retrieved_docs") or [],
            "iterations": int(final_state.get("iteration_count") or 0),
            "stop_reason": final_state.get("stop_reason") or "",
            "needs_user_input": bool(final_state.get("needs_user_input")),
            "user_questions": final_state.get("user_questions") or [],
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "tool_counts": dict(self._tools.call_counts),
        }
