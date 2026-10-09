"""Agent 可调用的工具集。

工具设计原则（参考 Anthropic《Building Effective Agents》的工具设计建议）：
1. **描述写清楚"什么时候用、什么时候不要用"**，而不是笼统的"帮你查资料"；
2. **参数少而明确**，能推算的参数不要暴露给模型（减少模型犯错的机会）；
3. **失败要给出可行动的信息**，例如"库里没有这个车型，可选值有 X/Y"，
   让模型能据此换策略（改参数重试 / 追问用户 / 改用 search_kb），而不是直接崩掉；
4. **返回值带上引用来源**，让最终答案可以标注 `[n]`。

四个工具：
  search_kb            知识库混合检索（保养、故障码、维修步骤等知识型问题）
  lookup_dtc           故障码精确/模糊查询（结构化表，比语义检索更准）
  compute_maintenance  按里程与车型计算应做的保养项目（纯计算，不调模型）
  ask_user            向用户追问缺失信息（虚拟工具，不产生副作用）
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional

from app.agent.kb import AgentKbStore, normalize_car_model
from app.core.retriever import HybridRetriever
from app.logging_conf import get_logger

logger = get_logger(__name__)

# 保养判定容差：里程在周期 ±10% 内即视为"该做了"
MILEAGE_TOLERANCE = 0.10


# --------------------------------------------------------------------------- #
# 工具返回值
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    """工具执行结果。ok=False 时不抛异常，而是把可行动的错误信息交给模型。"""

    ok: bool
    content: str
    data: Dict[str, Any] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    duration_ms: int = 0

    def to_message(self) -> str:
        """转成喂回模型的文本。错误也走这里，保证模型能看到失败原因。"""
        return self.content


# --------------------------------------------------------------------------- #
# 工具 JSON schema（OpenAI function calling 格式）
# --------------------------------------------------------------------------- #
SEARCH_KB_SPEC: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_kb",
        "description": (
            "在汽车售后知识库中做混合检索（向量召回 + BM25 关键词 + RRF 融合 + 重排），"
            "返回带来源与分值的原文片段。\n"
            "【适用】保养周期、保养项目、油液规格、故障码含义的解释性内容、维修与诊断步骤、"
            "保修与三包条款、故障现象的可能原因等知识型问题。\n"
            "【不适用】精确查询某个故障码的官方定义（用 lookup_dtc 更准）；"
            "按给定里程算保养清单（用 compute_maintenance 更准）；"
            "需要用户提供缺失信息时（用 ask_user）。\n"
            "【边界】只能检索到已入库文档中的内容。若返回 results 为空或分值都很低，"
            "说明知识库没有相关内容——此时**必须如实告知用户未找到依据，禁止编造**，"
            "不要凭常识补充库里没有的参数（如具体扭矩值、保养价格）。\n"
            "【返回】results 数组，每项含 text（原文片段）、source（文件名）、section（章节路径）、"
            "position（片段序号）、score（相关度）。引用时用 [序号] 标注。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "检索用的查询语句。建议用用户问题里的核心名词短语，"
                        "例如「机油 更换周期」「P0300 失火 原因」，不要带寒暄语。"
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回片段数量，默认 5，建议 3–8。取太大会引入不相关片段。",
                    "minimum": 1,
                    "maximum": 15,
                },
            },
            "required": ["query"],
        },
    },
}

LOOKUP_DTC_SPEC: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "lookup_dtc",
        "description": (
            "查询故障码（DTC）的结构化定义，数据来自本地码表数据库。\n"
            "【适用】用户给出明确码值（如 P0195、B1234、U0100），需要码值含义、"
            "所属系统/模块、可能原因；或用户只描述症状词（如「失火」「机油压力」）想找相关故障码。\n"
            "【不适用】解释故障码背后的检修步骤与判断逻辑——拿到定义后再调 search_kb 取维修内容。\n"
            "【边界】只覆盖已入库的码表。若 found=false，说明库里没有该码："
            "此时可以改用 keyword 参数按症状词模糊搜索，或改用 search_kb；"
            "**不要自行解释该故障码的含义**，也不要把范围码（如 P0300–P0399）当作具体码解释。\n"
            "【返回】found、results（每项含 code/description/system/possible_causes/model/source）。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dtc_code": {
                    "type": "string",
                    "description": (
                        "故障码，一位字母 + 四位数字，如 P0195。字母不区分大小写。"
                        "若用户没给码值但给了症状描述，把这个参数留空并改填 keyword。"
                    ),
                },
                "keyword": {
                    "type": "string",
                    "description": (
                        "症状/部件关键词，用于在故障码描述里做模糊匹配，"
                        "如「失火」「机油温度」「催化器」。仅在用户未给出具体码值时使用。"
                    ),
                },
            },
            "required": [],
        },
    },
}

COMPUTE_MAINTENANCE_SPEC: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "compute_maintenance",
        "description": (
            "根据车辆当前里程与车型，算出**现在该做哪些保养项目**（纯计算，不调用大模型）。\n"
            "【适用】用户问「该保养了吗」「这个里程要换什么」「保养清单」这类需要按里程判断的问题。\n"
            "【不适用】解释某个保养项目的做法或注意事项（用 search_kb）；查询故障码（用 lookup_dtc）。\n"
            "【边界】必须同时知道**车型**和**当前里程**才能计算。缺任一项时**不要瞎猜**："
            "先调 ask_user 向用户索要。若返回 supported=false，说明该车型不在库内，"
            "需如实告知用户并给出库里支持的车型列表，禁止用其它车型的周期冒充。\n"
            "【返回】supported、due_items（现在该做的项目，含周期原文与剩余里程）、"
            "upcoming_items（接近周期的项目）、supported_models。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mileage": {
                    "type": "integer",
                    "description": "车辆当前总里程，单位公里（km）。必须由用户提供，不要估算。",
                    "minimum": 0,
                    "maximum": 1000000,
                },
                "car_model": {
                    "type": "string",
                    "description": (
                        "车型名称，如「虚构车型A」。必须由用户提供；"
                        "若用户只说「我的车」而没有车型，先调 ask_user。"
                    ),
                },
                "months_in_use": {
                    "type": "integer",
                    "description": (
                        "车辆使用月数（可选）。用于判断按时间计周期的项目，如制动液每 2 年。"
                        "用户未提供时留空，此时按时间的项目会标注为「需自行核对」而不是猜一个值。"
                    ),
                    "minimum": 0,
                    "maximum": 600,
                },
            },
            "required": ["mileage", "car_model"],
        },
    },
}

ASK_USER_SPEC: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "向用户追问**继续完成任务所必需、但当前对话里没有**的信息。\n"
            "【适用】典型场景：用户问「我该保养了吗」但没说车型和里程；"
            "用户问「这个故障灯亮了怎么办」但没给故障码；"
            "用户描述的故障现象存在多种可能、必须知道车型才能给出准确内容。\n"
            "【不适用】信息已经足够时**禁止**调用它——不要为了显得严谨而反复追问；"
            "也不要问知识库里本来就能查到的通用知识。\n"
            "【边界】一次最多追问 3 个问题，且必须是**选择题或可由用户一句话回答**的问题，"
            "不要问需要用户查资料很久才能回答的问题。调用本工具后本轮对话会交给用户，"
            "等用户补充信息后再继续调用其它工具。\n"
            "【返回】questions 数组（照原样呈现给用户）与 missing_fields。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "要问用户的问题列表，1–3 条，每条一句话，直白不要绕。",
                    "minItems": 1,
                    "maxItems": 3,
                },
                "missing_fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "缺失的信息字段名，如 [\"车型\", \"当前里程\"]，用于前端高亮提示。",
                },
                "reason": {
                    "type": "string",
                    "description": "一句话说明为什么必须知道这些信息才能继续（会展示给用户，帮其理解）。",
                },
            },
            "required": ["questions"],
        },
    },
}

ALL_TOOL_SPECS = [SEARCH_KB_SPEC, LOOKUP_DTC_SPEC, COMPUTE_MAINTENANCE_SPEC, ASK_USER_SPEC]


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #
class ToolRegistry:
    """把工具实现与 schema 绑在一起。每个 ServiceContainer 持有一份。"""

    def __init__(self, retriever: HybridRetriever, kb_store: AgentKbStore) -> None:
        self._retriever = retriever
        self._kb = kb_store
        self._handlers: Dict[str, Callable[..., Awaitable[ToolResult]]] = {
            "search_kb": self.search_kb,
            "lookup_dtc": self.lookup_dtc,
            "compute_maintenance": self.compute_maintenance,
            "ask_user": self.ask_user,
        }
        # 调用计数：用于测试断言与运行观测（agent 行为测试里要用）
        self.call_counts: Dict[str, int] = {name: 0 for name in self._handlers}

    @property
    def specs(self) -> List[Dict[str, Any]]:
        return ALL_TOOL_SPECS

    @property
    def names(self) -> List[str]:
        return list(self._handlers)

    def reset_counts(self) -> None:
        for name in self.call_counts:
            self.call_counts[name] = 0

    async def execute(self, name: str, arguments: Dict[str, Any]) -> ToolResult:
        """执行工具。未知工具或参数错误都返回 ok=False 的结果，不抛异常。"""
        handler = self._handlers.get(name)
        started = time.perf_counter()
        if handler is None:
            return ToolResult(
                ok=False,
                content=f"工具 {name} 不存在。可用工具：{', '.join(self.names)}。"
                        f"请改用一个存在的工具，或直接基于已有信息回答。",
                error="unknown_tool",
            )
        self.call_counts[name] = self.call_counts.get(name, 0) + 1
        try:
            result = await handler(**(arguments or {}))
        except TypeError as exc:
            # 参数名/类型不对：把模型拉回正轨，而不是崩溃
            result = ToolResult(
                ok=False,
                content=(
                    f"调用 {name} 的参数不正确：{exc}。"
                    f"请检查参数名与类型后重试；若无法构造正确参数，改用其它工具或直接回答用户。"
                ),
                error="bad_arguments",
            )
        except Exception as exc:  # 兜底：任何工具内部异常都不该让 Agent 崩掉
            logger.exception("工具 %s 执行异常", name)
            result = ToolResult(
                ok=False,
                content=(
                    f"工具 {name} 执行失败：{type(exc).__name__}: {exc}。"
                    f"请换一种方式（换参数、换工具，或基于已有信息回答）后重试。"
                ),
                error="tool_exception",
            )
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    # ------------------------------------------------------------------ #
    async def search_kb(self, query: str = "", top_k: Optional[int] = None) -> ToolResult:
        """知识库混合检索。直接复用现有检索链路，不重写任何检索逻辑。"""
        query = (query or "").strip()
        if not query:
            return ToolResult(
                ok=False,
                content="search_kb 的 query 不能为空。请给出要检索的核心名词短语，例如「机油 更换周期」。",
                error="empty_query",
            )
        k = 5 if top_k is None else max(1, min(int(top_k), 15))
        chunks, mode = await self._retriever.retrieve(query, top_k=k)

        if not chunks:
            return ToolResult(
                ok=False,
                content=(
                    f"知识库中没有检索到与「{query}」相关的片段（检索模式 {mode}）。"
                    f"可以换更具体的关键词再试一次；若仍然为空，"
                    f"请直接告诉用户知识库中没有相关依据、不做推测性回答。"
                ),
                error="no_hit",
                data={"query": query, "mode": mode, "results": []},
            )

        results: List[Dict[str, Any]] = []
        citations: List[Dict[str, Any]] = []
        for index, chunk in enumerate(chunks, start=1):
            results.append(
                {
                    "index": index,
                    "text": chunk.text,
                    "source": chunk.source,
                    "section": chunk.section,
                    "position": chunk.position,
                    "score": round(float(chunk.score), 4),
                }
            )
            citations.append(
                {
                    "index": index,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "section": chunk.section,
                    "position": chunk.position,
                    "score": round(float(chunk.score), 4),
                }
            )

        lines = [f"检索到 {len(results)} 条片段（模式 {mode}）："]
        for item in results:
            lines.append(
                f"[{item['index']}] 来源={item['source']}｜章节={item['section'] or '（无）'}"
                f"｜片段#{item['position']}｜相关度={item['score']}\n{item['text']}"
            )
        return ToolResult(
            ok=True,
            content="\n\n".join(lines),
            data={"query": query, "mode": mode, "results": results},
            citations=citations,
        )

    # ------------------------------------------------------------------ #
    async def lookup_dtc(self, dtc_code: str = "", keyword: str = "") -> ToolResult:
        """故障码查询：优先精确匹配码值，未给码值或未命中时按症状词模糊匹配。"""
        code = (dtc_code or "").strip().upper()
        word = (keyword or "").strip()

        records = self._kb.find_dtc(code) if code else []
        mode = "code"

        if not records and word:
            records = self._kb.search_dtc(word)
            mode = "keyword"

        if not records:
            available = self._kb.stats().get("dtc_by_model", {})
            hint = ""
            if code:
                hint = (
                    f"码表库中没有 {code}。可能的解决方式："
                    f"① 若用户只记得症状（如失火、机油压力），用 keyword 参数重试；"
                    f"② 用 search_kb 检索码表文档中的解释性内容；"
                    f"③ 如实告知用户库中没有该码，不要自行解释其含义。"
                )
            else:
                hint = "未提供 dtc_code 且 keyword 也未命中。请补一个具体码值或换一个症状关键词。"
            return ToolResult(
                ok=False,
                content=f"{hint}（库内码表覆盖：{json.dumps(available, ensure_ascii=False)}）",
                error="not_found",
                data={"code": code, "keyword": word, "found": False, "results": []},
            )

        results = [
            {
                "code": record.code,
                "description": record.description,
                "system": record.system,
                "possible_causes": record.possible_causes,
                "model": record.model,
                "source": record.source,
            }
            for record in records
        ]
        citations = [
            {
                "index": index,
                "source": record.source,
                "section": f"{record.code} {record.description}",
                "position": 0,
                "score": 1.0,
            }
            for index, record in enumerate(records, start=1)
        ]

        header = f"故障码查询命中 {len(results)} 条（匹配方式：{'精确码值' if mode == 'code' else '症状关键词'}）："
        lines = [header]
        for item in results:
            part = [f"- {item['code']}：{item['description']}"]
            if item["system"]:
                part.append(f"  所属系统/模块：{item['system']}")
            if item["possible_causes"]:
                part.append(f"  可能原因线索：{item['possible_causes']}")
            part.append(f"  适用车型/来源：{item['model']}｜{item['source']}")
            lines.append("\n".join(part))
        lines.append(
            "提示：以上是码表里的定义与线索。若需要该码的检修步骤与判断逻辑，请再调用 search_kb。"
        )
        return ToolResult(
            ok=True,
            content="\n".join(lines),
            data={"code": code, "keyword": word, "found": True, "mode": mode, "results": results},
            citations=citations,
        )

    # ------------------------------------------------------------------ #
    async def compute_maintenance(
        self,
        mileage: Optional[int] = None,
        car_model: str = "",
        months_in_use: Optional[int] = None,
    ) -> ToolResult:
        """按里程与车型计算保养项目。

        判定规则（写进注释，便于面试解释为什么这么算）：
        - 以里程周期为主：当前里程落在「周期里程 × (1 ± 10%)」区间内即视为到期；
        - 已远超周期（里程 > 周期 × 1.1）的项目标注为"已超期"；
        - 只按时间计周期的项目（无里程数据，如制动液每 2 年）：
          用户给了 months_in_use 才判定，否则标注"需按时间自行核对"，
          **不猜**一个里程值出来。
        """
        if mileage is None:
            return ToolResult(
                ok=False,
                content=(
                    "缺少 mileage（当前里程），无法计算保养项目。"
                    "请先调用 ask_user 向用户索要当前总里程（公里）。"
                ),
                error="missing_mileage",
            )
        model = normalize_car_model(car_model)
        if not model:
            return ToolResult(
                ok=False,
                content=(
                    "缺少 car_model（车型），无法计算保养项目。"
                    "请先调用 ask_user 向用户索要车型。"
                ),
                error="missing_model",
            )

        info = self._kb.get_model(model)
        records = self._kb.maintenance_for(model)
        supported = self._kb.list_models()

        if info is None or not records:
            return ToolResult(
                ok=False,
                content=(
                    f"车型「{car_model}」不在保养数据库中，无法计算。"
                    f"当前支持的车型：{', '.join(supported) if supported else '（库为空）'}。"
                    f"请如实告知用户：该车型没有入库数据，不要套用其它车型的保养周期。"
                ),
                error="unsupported_model",
                data={"supported_models": supported, "car_model": car_model},
            )

        km = max(0, int(mileage))
        due: List[Dict[str, Any]] = []
        upcoming: List[Dict[str, Any]] = []
        time_only: List[Dict[str, Any]] = []

        for record in records:
            entry: Dict[str, Any] = {
                "item": record.item,
                "period_text": record.period_text,
                "period_km": record.period_km,
                "period_months": record.period_months,
                "source": record.source,
            }
            if record.period_km:
                ratio = km / record.period_km
                entry["ratio"] = round(ratio, 2)
                if ratio >= 1.0 - MILEAGE_TOLERANCE:
                    entry["status"] = "已超期" if ratio > 1.0 + MILEAGE_TOLERANCE else "到期"
                    due.append(entry)
                elif ratio >= 0.8:
                    entry["status"] = "即将到期"
                    entry["remaining_km"] = int(record.period_km * (1.0 - MILEAGE_TOLERANCE) - km)
                    upcoming.append(entry)
            else:
                entry["status"] = "需按时间核对"
                if record.period_months and months_in_use is not None:
                    entry["ratio"] = round(int(months_in_use) / record.period_months, 2)
                    if int(months_in_use) >= record.period_months * (1.0 - MILEAGE_TOLERANCE):
                        entry["status"] = "到期（按时间）"
                        due.append(entry)
                    else:
                        entry["remaining_months"] = int(
                            record.period_months * (1.0 - MILEAGE_TOLERANCE) - int(months_in_use)
                        )
                        upcoming.append(entry)
                else:
                    time_only.append(entry)

        data = {
            "car_model": model,
            "display_name": info["display_name"],
            "is_fictional": info["is_fictional"],
            "mileage": km,
            "months_in_use": months_in_use,
            "due_items": due,
            "upcoming_items": upcoming,
            "time_only_items": time_only,
            "supported_models": supported,
            "source": records[0].source if records else "",
        }

        lines = [
            f"车型：{info['display_name']}（{'示例/虚构数据' if info['is_fictional'] else '真实数据'}）"
            f"，当前里程 {km:,} km"
            + (f"，已使用 {months_in_use} 个月" if months_in_use is not None else ""),
            f"数据来源：{data['source']}",
        ]
        if due:
            lines.append(f"\n【现在该做】共 {len(due)} 项：")
            for entry in due:
                lines.append(f"- {entry['item']}（周期：{entry['period_text']}）→ {entry['status']}")
        else:
            lines.append("\n【现在该做】按里程周期判断，暂无到期项目。")
        if upcoming:
            lines.append(f"\n【即将到期】共 {len(upcoming)} 项：")
            for entry in upcoming:
                if "remaining_km" in entry:
                    lines.append(f"- {entry['item']}（周期：{entry['period_text']}）→ 约再行驶 {entry['remaining_km']:,} km")
                else:
                    lines.append(f"- {entry['item']}（周期：{entry['period_text']}）→ 约 {entry.get('remaining_months', '?')} 个月后")
        if time_only:
            lines.append("\n【按时间计周期、无法用里程判断】：")
            for entry in time_only:
                lines.append(f"- {entry['item']}（周期：{entry['period_text']}）→ 需用户提供使用月数或自行核对")
        lines.append(
            "\n提示：以上保养周期来自示例数据文档。回答用户时请标注来源，"
            "并说明数据为示例/虚构、不对应真实车型。"
        )
        return ToolResult(
            ok=True,
            content="\n".join(lines),
            data=data,
            citations=[
                {
                    "index": 1,
                    "source": data["source"],
                    "section": "常规保养周期表",
                    "position": 0,
                    "score": 1.0,
                }
            ],
        )

    # ------------------------------------------------------------------ #
    async def ask_user(
        self,
        questions: Optional[List[str]] = None,
        missing_fields: Optional[List[str]] = None,
        reason: str = "",
    ) -> ToolResult:
        """追问用户。不产生副作用：只是把问题整理好交给上层，由上层暂停图执行。"""
        items = [str(q).strip() for q in (questions or []) if str(q).strip()][:3]
        if not items:
            return ToolResult(
                ok=False,
                content="ask_user 至少需要 1 个问题。请给出具体要问用户的内容，或直接基于已有信息回答。",
                error="empty_questions",
            )
        fields = [str(f).strip() for f in (missing_fields or []) if str(f).strip()]
        data = {"questions": items, "missing_fields": fields, "reason": reason.strip()}
        lines = ["已向用户追问以下信息（等待用户补充后再继续）："]
        for index, question in enumerate(items, start=1):
            lines.append(f"{index}. {question}")
        if reason.strip():
            lines.append(f"追问原因：{reason.strip()}")
        return ToolResult(ok=True, content="\n".join(lines), data=data)
