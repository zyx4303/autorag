"""Agent 行为测试（确定性，不联网、不需要 API Key）。

覆盖需求里要求的 5 类场景：
  1. 纯知识问题 → 只调 search_kb 一次就回答
  2. 故障码问题 → 先 lookup_dtc 再 search_kb
  3. 信息不足 → 先 ask_user
  4. 迭代超过 6 次 → 优雅停止
  5. 工具报错 → 换策略而不是崩溃

实现方式：用"脚本化的假 LLM"精确控制每一步返回什么，
因此断言的是**图的行为**（路由、停止条件、trace 记录），而不是模型输出质量。
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.graph import AGENT_MAX_ITERATIONS, AgentGraph, parse_tool_calls  # noqa: E402
from app.agent.kb import (  # noqa: E402
    AgentKbStore,
    bootstrap_from_documents,
    iter_dtc_rows,
    iter_maintenance_rows,
    normalize_car_model,
    parse_period,
)
from app.agent.tools import ALL_TOOL_SPECS, ToolRegistry  # noqa: E402
from app.config import Settings  # noqa: E402
from app.core.llm_client import LLMResult  # noqa: E402

PASSED = 0
FAILED = 0
FAILURES: List[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        FAILURES.append(f"{name} :: {detail}")
        print(f"  [FAIL] {name}  {detail}")


# --------------------------------------------------------------------------- #
# 测试替身
# --------------------------------------------------------------------------- #
class FakeLLM:
    """脚本化 LLM：按调用次序返回预设结果，并记录收到的消息（便于断言上下文传递）。"""

    def __init__(self, script: List[LLMResult]) -> None:
        self._script = list(script)
        self.calls: List[List[Dict[str, Any]]] = []
        self.tools_seen: List[Optional[List[Dict[str, Any]]]] = []

    @property
    def model(self) -> str:
        return "fake-model"

    async def chat(self, messages, temperature=None, max_tokens=None, tools=None, tool_choice=None):
        self.calls.append(list(messages))
        self.tools_seen.append(tools)
        if self._script:
            return self._script.pop(0)
        return LLMResult(text="（脚本已用尽）", model="fake-model")


def tool_call(name: str, arguments: Dict[str, Any], call_id: str = "call_1") -> LLMResult:
    return LLMResult(
        text="",
        model="fake-model",
        finish_reason="tool_calls",
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
            }
        ],
        raw_message={"role": "assistant", "content": ""},
    )


def answer(text: str) -> LLMResult:
    return LLMResult(text=text, model="fake-model", finish_reason="stop")


class FakeChunk:
    def __init__(self, index: int, text: str, source: str = "示例文档.md", section: str = "一、保养"):
        self.chunk_id = f"c{index}"
        self.doc_id = "d1"
        self.source = source
        self.title = "示例"
        self.section = section
        self.position = index
        self.text = text
        self.score = 0.9 - index * 0.1
        self.vector_score = 0.8
        self.keyword_score = 1.2
        self.from_vector = True
        self.from_keyword = True


class FakeRetriever:
    """假检索器：返回固定片段，并可模拟"零命中"与"抛异常"。"""

    def __init__(self, chunks: Optional[List[FakeChunk]] = None, raise_error: bool = False):
        self._chunks = chunks if chunks is not None else [
            FakeChunk(1, "更换发动机机油：每 8,000 km 或每 8 个月（以先到者为准）。"),
            FakeChunk(2, "P0300：检测到随机或多缸失火，检查火花塞与点火线圈。"),
        ]
        self._raise = raise_error
        self.queries: List[str] = []

    async def retrieve(self, query: str, top_k: Optional[int] = None, mode: Optional[str] = None):
        self.queries.append(query)
        if self._raise:
            raise RuntimeError("模拟检索后端故障")
        return self._chunks, "hybrid"


SAMPLE_DTC_DOC = """# 示例码表

| 故障码 | 中文描述 |
|---|---|
| P0195 | 发动机机油温度传感器故障 |
| P0300 | 检测到随机或多缸失火 |
| P0420 | 催化器净化效率低于阈值（第 1 排） |
| P0300–P0399 | 点火或气缸失火 |
"""

SAMPLE_MAINT_DOC = """# 示例保养

## 一、常规保养周期表

| 保养项目 | 周期 |
|---|---|
| 更换发动机机油 | 每 8,000 km 或每 8 个月（以先到者为准） |
| 更换空气滤清器滤芯 | 每 16,000 km |
| 更换制动液 | 每 2 年 |
"""


def make_env(tmp_root: Path, retriever=None) -> tuple:
    """构造一套测试环境：临时文档目录 + 空库 + 工具注册表。"""
    docs = tmp_root / "data" / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "示例码表.md").write_text(SAMPLE_DTC_DOC, encoding="utf-8")
    (docs / "示例保养.md").write_text(SAMPLE_MAINT_DOC, encoding="utf-8")

    store = AgentKbStore(tmp_root / "agent_kb.db")
    # 建库只依赖文件内容，因此手工插入（避免依赖仓库里具体文件名）
    dtc_rows = iter_dtc_rows(SAMPLE_DTC_DOC, "示例码表.md")
    for record in dtc_rows:
        record.model = "通用"
    store.upsert_dtc(dtc_rows)
    maint_rows = iter_maintenance_rows(SAMPLE_MAINT_DOC, "示例保养.md", "虚构车型 A")
    store.upsert_maintenance(maint_rows)
    store.register_model("虚构车型 A", "虚构车型 A（示例数据）", "示例保养.md", is_fictional=True)

    settings = Settings()
    registry = ToolRegistry(retriever or FakeRetriever(), store)
    return settings, store, registry


# --------------------------------------------------------------------------- #
# [1] Markdown 解析与数据层
# --------------------------------------------------------------------------- #
def test_kb_parsing(tmp_root: Path) -> None:
    print("\n[1] 业务工具数据层（表格解析 / 周期归一化 / 车型归一化）")

    check("周期解析：里程+月份", parse_period("每 8,000 km 或每 8 个月（以先到者为准）") == (8000, 8))
    check("周期解析：仅年", parse_period("每 2 年") == (None, 24))
    check("周期解析：仅里程", parse_period("每 64,000 km") == (64000, None))
    check("周期解析：无法解析返回 None", parse_period("按需更换") == (None, None))

    rows = iter_dtc_rows(SAMPLE_DTC_DOC, "示例码表.md")
    codes = {r.code: r.description for r in rows}
    check("码表解析出具体码", {"P0195", "P0300", "P0420"} <= set(codes), str(sorted(codes)))
    check("码表跳过区间行（P0300–P0399 不是具体码）", "P0300–P0399" not in codes)

    maint = iter_maintenance_rows(SAMPLE_MAINT_DOC, "示例保养.md", "虚构车型 A")
    check("保养表解析出 3 项", len(maint) == 3, str([m.item for m in maint]))

    check("车型归一化：带空格", normalize_car_model("虚构车型A") == "虚构车型 A")
    check("车型归一化：未知车型原样返回", normalize_car_model("某不存在车型") == "某不存在车型")

    _settings, store, _registry = make_env(tmp_root)
    check("库内查到 P0195", [d.code for d in store.find_dtc("p0195")] == ["P0195"])
    check("症状词模糊查询可用", "P0300" in [d.code for d in store.search_dtc("失火")])
    check("未知码返回空", store.find_dtc("P9999") == [])
    check("保养项目按车型取回", len(store.maintenance_for("虚构车型 A")) == 3)


# --------------------------------------------------------------------------- #
# [2] 工具 JSON schema 与工具执行
# --------------------------------------------------------------------------- #
async def test_tools(tmp_root: Path) -> None:
    print("\n[2] 工具 schema 与执行行为")

    names = {spec["function"]["name"] for spec in ALL_TOOL_SPECS}
    check("四个工具都注册了 schema", names == {"search_kb", "lookup_dtc", "compute_maintenance", "ask_user"}, str(names))
    for spec in ALL_TOOL_SPECS:
        function = spec["function"]
        desc = function.get("description", "")
        check(f"{function['name']} 描述含适用边界", "【适用】" in desc and "【边界】" in desc)
        check(f"{function['name']} 描述够具体（>120 字）", len(desc) > 120, f"{len(desc)}")

    _settings, _store, registry = make_env(tmp_root)

    result = await registry.execute("search_kb", {"query": "机油更换周期"})
    check("search_kb 正常返回", result.ok and len(result.data["results"]) == 2, result.content[:80])
    check("search_kb 带来源引用", len(result.citations) == 2 and result.citations[0]["index"] == 1)

    result = await registry.execute("search_kb", {"query": ""})
    check("search_kb 空查询被拒并给出建议", not result.ok and "不能为空" in result.content)

    result = await registry.execute("lookup_dtc", {"dtc_code": "p0195"})
    check("lookup_dtc 大小写不敏感", result.ok and result.data["results"][0]["code"] == "P0195")

    result = await registry.execute("lookup_dtc", {"keyword": "失火"})
    check("lookup_dtc 症状词兜底", result.ok and result.data["mode"] == "keyword")

    result = await registry.execute("lookup_dtc", {"dtc_code": "P9999"})
    check("lookup_dtc 未命中时给出换策略建议", not result.ok and "keyword" in result.content)

    result = await registry.execute("lookup_dtc", {})
    check("lookup_dtc 无参数时提示补参数", not result.ok and "未提供" in result.content)

    result = await registry.execute("compute_maintenance", {"mileage": 8000, "car_model": "虚构车型A"})
    check("compute_maintenance 命中 8000km 到期项", result.ok and any("机油" in i["item"] for i in result.data["due_items"]), result.content[:120])
    check(
        "compute_maintenance 标出按时间核对项（制动液）",
        any("制动液" in i["item"] for i in result.data["time_only_items"]),
        str(result.data["time_only_items"]),
    )

    result = await registry.execute("compute_maintenance", {"car_model": "虚构车型A"})
    check("compute_maintenance 缺里程时要求追问", not result.ok and "ask_user" in result.content)

    result = await registry.execute("compute_maintenance", {"mileage": 10000, "car_model": "不存在的车"})
    check("compute_maintenance 车型不支持时给出可选车型", not result.ok and "支持的车型" in result.content, result.content[:100])

    result = await registry.execute("compute_maintenance", {"mileage": 2000, "car_model": "虚构车型A"})
    check("compute_maintenance 里程未到时无到期项", result.ok and result.data["due_items"] == [], result.content[:100])

    result = await registry.execute("ask_user", {"questions": ["你的车型是什么？", "当前总里程是多少？"]})
    check("ask_user 返回问题列表", result.ok and len(result.data["questions"]) == 2)

    result = await registry.execute("ask_user", {"questions": []})
    check("ask_user 空问题被拒", not result.ok)

    result = await registry.execute("不存在的工具", {})
    check("未知工具返回可行动提示而非抛错", not result.ok and "不存在" in result.content)

    result = await registry.execute("search_kb", {"错误参数名": 1})
    check("参数名错误时提示改参数", not result.ok and "参数不正确" in result.content, result.content[:100])

    # 工具内部异常不应让 Agent 崩掉
    _settings2, _store2, broken = make_env(tmp_root / "broken", retriever=FakeRetriever(raise_error=True))
    result = await broken.execute("search_kb", {"query": "机油"})
    check("工具内部异常被兜住并可换策略", not result.ok and "换一种方式" in result.content, result.content[:120])


# --------------------------------------------------------------------------- #
# [3] 工具调用解析容错
# --------------------------------------------------------------------------- #
def test_parse_tool_calls() -> None:
    print("\n[3] tool_calls 解析容错")

    calls = parse_tool_calls([
        {"id": "a", "function": {"name": "search_kb", "arguments": '{"query": "机油"}'}},
    ])
    check("标准 JSON 字符串参数可解析", calls[0]["arguments"] == {"query": "机油"})

    calls = parse_tool_calls([
        {"id": "b", "function": {"name": "search_kb", "arguments": {"query": "机油"}}},
    ])
    check("已是 dict 的参数直接使用", calls[0]["arguments"] == {"query": "机油"})

    calls = parse_tool_calls([
        {"id": "c", "function": {"name": "search_kb", "arguments": "{坏掉的 json"}},
    ])
    check("非法 JSON 不抛异常且标记 parse_error", calls[0]["arguments"] == {} and calls[0]["parse_error"] != "")

    calls = parse_tool_calls([{"id": "d", "function": {"arguments": "{}"}}])
    check("缺工具名时跳过该调用", calls == [])


# --------------------------------------------------------------------------- #
# [4] 图行为：单工具、多工具链、追问、上限、错误恢复
# --------------------------------------------------------------------------- #
async def test_graph_behaviors(tmp_root: Path) -> None:
    print("\n[4] LangGraph 状态图行为")

    # ---- 场景 1：纯知识问题只调 search_kb 一次就回答 ----
    settings, store, registry = make_env(tmp_root / "s1")
    llm = FakeLLM([
        tool_call("search_kb", {"query": "机油更换周期"}),
        answer("机油每 8,000 km 或 8 个月更换 [1]。"),
    ])
    graph = AgentGraph(settings, llm, registry)
    result = await graph.ainvoke("机油多久换一次？", "s1")
    check("场景1 只调了 search_kb", result["tool_counts"]["search_kb"] == 1, str(result["tool_counts"]))
    check("场景1 未调用其它工具", sum(v for k, v in result["tool_counts"].items() if k != "search_kb") == 0)
    check("场景1 迭代 2 次（工具+回答）", result["iterations"] == 2, str(result["iterations"]))
    check("场景1 给出答案且带引用", "[1]" in result["answer"] and len(result["citations"]) == 2)
    check("场景1 收尾原因 answered", result["stop_reason"] == "answered", result["stop_reason"])
    check("场景1 trace 记录了工具入参出参", result["tool_trace"][0]["tool"] == "search_kb" and result["tool_trace"][0]["result_count"] == 2)
    check("场景1 第二轮收到了工具结果消息", any(m.get("role") == "tool" for m in llm.calls[1]))

    # ---- 场景 2：故障码问题先 lookup_dtc 再 search_kb ----
    settings, store, registry = make_env(tmp_root / "s2")
    llm = FakeLLM([
        tool_call("lookup_dtc", {"dtc_code": "P0195"}, "c1"),
        tool_call("search_kb", {"query": "机油温度传感器 检修"}, "c2"),
        answer("P0195 表示发动机机油温度传感器故障 [1]，检修步骤见 [2]。"),
    ])
    graph = AgentGraph(settings, llm, registry)
    result = await graph.ainvoke("P0195 故障码什么意思", "s2")
    order = [step["tool"] for step in result["tool_trace"]]
    check("场景2 调用顺序为 lookup_dtc → search_kb", order == ["lookup_dtc", "search_kb"], str(order))
    check("场景2 迭代 3 次", result["iterations"] == 3, str(result["iterations"]))
    check("场景2 答案含引用来源", len(result["citations"]) >= 2)
    check("场景2 检索到片段被记录", len(result["retrieved_docs"]) == 2)

    # ---- 场景 3：信息不足时先 ask_user（虚拟工具，图应停在追问） ----
    settings, store, registry = make_env(tmp_root / "s3")
    llm = FakeLLM([
        tool_call("ask_user", {
            "questions": ["你的车型是什么？", "当前总里程是多少公里？"],
            "missing_fields": ["车型", "里程"],
            "reason": "计算保养项目需要知道车型与当前里程",
        }),
    ])
    graph = AgentGraph(settings, llm, registry)
    result = await graph.ainvoke("我该保养了吗", "s3")
    check("场景3 触发了 ask_user", result["tool_counts"]["ask_user"] == 1, str(result["tool_counts"]))
    check("场景3 标记为等待用户输入", result["needs_user_input"] is True)
    check("场景3 收尾原因 ask_user", result["stop_reason"] == "ask_user", result["stop_reason"])
    check("场景3 问题被呈现给用户", "车型" in result["answer"] and "里程" in result["answer"])
    check("场景3 追问后不再继续调工具", result["tool_counts"]["search_kb"] == 0)

    # ---- 场景 4：用户补充信息后继续（同一 session 续接 checkpoint） ----
    llm2 = FakeLLM([
        tool_call("compute_maintenance", {"mileage": 16000, "car_model": "虚构车型A"}, "c1"),
        answer("按 16,000 km 判断，应更换空气滤清器滤芯 [1]。"),
    ])
    graph.set_llm_for_test(llm2) if hasattr(graph, "set_llm_for_test") else None
    graph2 = AgentGraph(settings, llm2, registry)
    result2 = await graph2.ainvoke("我是虚构车型A，里程 16000 公里", "s3")
    check("场景4 补充信息后能继续计算保养", result2["tool_counts"]["compute_maintenance"] == 1, str(result2["tool_counts"]))
    check("场景4 给出答案", "空气滤清器" in result2["answer"], result2["answer"][:80])

    # ---- 场景 5：迭代超过上限能优雅停止 ----
    settings, store, registry = make_env(tmp_root / "s5")
    # 模型每轮都想调工具，永不回答：图必须在 6 次内停住
    llm = FakeLLM([tool_call("search_kb", {"query": f"查询{i}"}, f"c{i}") for i in range(20)])
    graph = AgentGraph(settings, llm, registry)
    result = await graph.ainvoke("一直查下去", "s5")
    check("场景5 迭代次数不超过上限", result["iterations"] <= AGENT_MAX_ITERATIONS, str(result["iterations"]))
    check("场景5 收尾原因 max_iterations", result["stop_reason"] == "max_iterations", result["stop_reason"])
    check("场景5 告知用户已到上限", "上限" in result["answer"], result["answer"][:80])
    check("场景5 仍给出已获得的结果摘要", "来源" in result["answer"] or "工具" in result["answer"])
    check("场景5 未无限循环（LLM 调用次数受限）", len(llm.calls) <= AGENT_MAX_ITERATIONS + 1, str(len(llm.calls)))

    # ---- 场景 6：工具报错时换策略而不是崩溃 ----
    settings, store, registry = make_env(tmp_root / "s6")
    llm = FakeLLM([
        tool_call("lookup_dtc", {"dtc_code": "P9999"}, "c1"),   # 库里没有 → 工具返回可行动提示
        tool_call("lookup_dtc", {"keyword": "失火"}, "c2"),      # 模型据此换策略
        answer("库中没有 P9999；与失火相关的码有 P0300 [1]。"),
    ])
    graph = AgentGraph(settings, llm, registry)
    result = await graph.ainvoke("P9999 是什么", "s6")
    check("场景6 换策略后成功", result["stop_reason"] == "answered", result["stop_reason"])
    check("场景6 trace 记录了第一次失败", result["tool_trace"][0]["ok"] is False and result["tool_trace"][0]["error"] == "not_found")
    check("场景6 第二次调用成功", result["tool_trace"][1]["ok"] is True)
    check("场景6 模型看到了失败原因", any("没有 P9999" in str(m.get("content", "")) for m in llm.calls[1]))

    # ---- 场景 7：检索后端异常 → 工具报错但图不崩 ----
    settings, store, registry = make_env(tmp_root / "s7", retriever=FakeRetriever(raise_error=True))
    llm = FakeLLM([
        tool_call("search_kb", {"query": "机油"}, "c1"),
        answer("检索服务暂时不可用，稍后可重试。"),
    ])
    graph = AgentGraph(settings, llm, registry)
    result = await graph.ainvoke("机油多久换", "s7")
    check("场景7 工具异常被记录在 trace", result["tool_trace"][0]["ok"] is False)
    check("场景7 图未崩溃且给出回答", result["stop_reason"] == "answered" and result["answer"])

    # ---- 场景 8：LLM 调用失败 → 结构化错误而不是 500 ----
    class BrokenLLM(FakeLLM):
        async def chat(self, *args, **kwargs):
            from app.core.llm_client import LLMError
            raise LLMError("模拟鉴权失败")

    settings, store, registry = make_env(tmp_root / "s8")
    graph = AgentGraph(settings, BrokenLLM([]), registry)
    result = await graph.ainvoke("任意问题", "s8")
    check("场景8 LLM 失败时返回结构化错误", result["stop_reason"] == "error" and "失败" in result["answer"])
    check("场景8 明确提示替代接口", "chat" in result["answer"])


# --------------------------------------------------------------------------- #
# [5] 系统提示词与迭代上限常量
# --------------------------------------------------------------------------- #
def test_prompt_and_limits() -> None:
    print("\n[5] 系统提示词与停止条件")

    from app.agent.graph import AGENT_SYSTEM_PROMPT
    check("提示词要求禁止编造", "不许编造" in AGENT_SYSTEM_PROMPT or "禁止编造" in AGENT_SYSTEM_PROMPT)
    check("提示词要求标注引用序号", "[序号]" in AGENT_SYSTEM_PROMPT)
    check("提示词要求信息不足先追问", "ask_user" in AGENT_SYSTEM_PROMPT)
    check("提示词写明工具调用上限", str(AGENT_MAX_ITERATIONS) in AGENT_SYSTEM_PROMPT)
    check("迭代上限为 6", AGENT_MAX_ITERATIONS == 6, str(AGENT_MAX_ITERATIONS))


# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 70)
    print("Agent 行为测试（脚本化 LLM，确定性，不联网、不需要 API Key）")
    print("=" * 70)

    tmp_root = PROJECT_ROOT / "data" / "_agent_test_tmp"
    if tmp_root.exists():
        import shutil
        shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    test_kb_parsing(tmp_root)
    asyncio.run(test_tools(tmp_root))
    test_parse_tool_calls()
    asyncio.run(test_graph_behaviors(tmp_root))
    test_prompt_and_limits()

    print("\n" + "=" * 70)
    print(f"结果：通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 70)
    if FAILURES:
        print("失败明细：")
        for item in FAILURES:
            print(f"  - {item}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
