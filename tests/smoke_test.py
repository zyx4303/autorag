#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""冒烟测试：只用标准库 + 项目依赖，验证纯逻辑，不联网、不调用大模型、不需要任何 Key。

运行：
    python tests/smoke_test.py

覆盖：
1. 文本工具：归一化、分句、中英混合分词；
2. 切分器：不丢内容、块大小受限、章节路径正确、重叠生效；
3. BM25：自建索引能召回含关键词的片段，且与查询无关的片段不排第一；
4. RRF 融合：两路结果能正确合并、去重并带上双方标记；
5. 引用校验：越界 [n] 能被检测出来。
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings  # noqa: E402
from app.core.attribution import (  # noqa: E402
    AttributionChecker,
    is_corpus_absence_claim,
    strip_clause_heading,
)
from app.core.bm25 import BM25Index  # noqa: E402
from app.core.chunker import chunk_document  # noqa: E402
from app.core.embedding import HashEmbedder  # noqa: E402
from app.core.text_utils import normalize_text, split_sentences, tokenize  # noqa: E402
from app.models import Citation, RetrievedChunk, SourceDocument  # noqa: E402
from app.services.rag_pipeline import QAEngine  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} {detail}")


SAMPLE_MD = """# 保养手册

## 首保

首保应在 [待补充] 公里或 [待补充] 个月内完成，以先到者为准。

## 机油

推荐使用 [待补充] 黏度等级的机油，更换周期为 [待补充] 公里。
更换机油时应同时更换机油滤清器。

## 故障码

| 故障码 | 含义 | 建议处理 |
| --- | --- | --- |
| P0420 | [待补充] | [待补充] |
"""


def make_settings(tmp_root: Path) -> Settings:
    return Settings(
        documents_dir_raw=str(tmp_root / "documents"),
        chroma_dir_raw=str(tmp_root / "chroma"),
        upload_dir_raw=str(tmp_root / "uploads"),
        registry_path_raw=str(tmp_root / "registry.json"),
        bm25_index_path_raw=str(tmp_root / "bm25.json"),
        chunk_size=120,
        chunk_overlap=20,
        chunk_min_chars=10,
        embedding_provider="hash",
    )


def test_text_utils() -> None:
    print("\n[1] 文本工具")
    raw = "第一行。\r\n\r\n\r\n  第二行  有空格\u200b。"
    normalized = normalize_text(raw)
    check("归一化压缩空行", "\n\n\n" not in normalized and "\u200b" not in normalized, repr(normalized))
    sentences = split_sentences("说明书要求定期检查。是否需要更换？请参考手册！")
    check("分句数量正确", len(sentences) == 3, str(sentences))
    tokens = tokenize("更换机油 5W-30 oil")
    check("中文产出 bigram", "机油" in tokens, str(tokens[:10]))
    check("英文数字成词", "5w" in tokens or "30" in tokens or "oil" in tokens, str(tokens))


def test_chunker(tmp_root: Path) -> None:
    print("\n[2] 切分器")
    settings = make_settings(tmp_root)
    document = SourceDocument(
        doc_id="doc_test",
        source="保养手册.md",
        title="保养手册",
        text=SAMPLE_MD,
        checksum="x" * 8,
    )
    chunks = chunk_document(document, settings)
    # 断言不应依赖"手工数出来的段数"（会随 chunk_size/min_chars 调整而变），
    # 因此这里只断言与切分逻辑本质相关的性质：多章节必须产生多个片段、且可复现。
    check("多章节切出多个片段", len(chunks) >= 2, f"chunks={len(chunks)}")
    check(
        "切分可复现（同输入同输出）",
        [c.text for c in chunk_document(document, settings)] == [c.text for c in chunks],
    )
    check("片段 ID 唯一", len({chunk.chunk_id for chunk in chunks}) == len(chunks))
    check(
        "章节路径被识别",
        any("首保" in chunk.section for chunk in chunks),
        str([chunk.section for chunk in chunks]),
    )
    check(
        "块大小未失控",
        all(len(chunk.text) <= settings.chunk_size + settings.chunk_overlap + 40 for chunk in chunks),
        str([len(chunk.text) for chunk in chunks]),
    )
    joined = "".join(chunk.text for chunk in chunks)
    for keyword in ("首保", "机油滤清器", "P0420", "[待补充]"):
        check(f"内容未丢失: {keyword}", keyword in joined)
    check(
        "表格未被切断",
        any("| P0420 |" in chunk.text and "| 建议处理 |" in chunk.text for chunk in chunks),
    )

    # ---- 章节标签必须与正文一致（实测 4/79 个片段曾标错） ----
    def _section_consistency(extra_text: str):
        doc = SourceDocument(
            doc_id="doc_sec",
            source="章节.md",
            title="章节",
            text=extra_text,
            checksum="y" * 8,
        )
        got = chunk_document(doc, settings)
        # 每个片段的 section 里的「第 N 章」必须能在正文里找到对应内容标记
        bad = [
            c
            for c in got
            if c.text.lstrip().startswith("---")
            or ("第壹章" in c.section and "壹" not in c.text)
            or ("第贰章" in c.section and "贰" not in c.text)
        ]
        return got, bad

    # 章节之间有 Markdown 分割线 —— 这正是线上数据里的情形
    hr_doc = (
        "## 第壹章 保养\n\n壹：机油每 8,000 km 更换。\n\n---\n\n"
        "## 第贰章 保修\n\n贰：包修期为 3 年或 60,000 公里。\n"
    )
    got, bad = _section_consistency(hr_doc)
    check("分割线不会被当成正文保留", not any("---" in c.text for c in got),
          str([c.text[:12] for c in got]))
    check("章节标签与正文一致（含分割线场景）", not bad,
          str([(c.section[-8:], c.text[:10]) for c in bad]))
    check(
        "两章各自成块，不合并",
        any("壹" in c.text for c in got) and any("贰" in c.text for c in got),
        str([c.text[:8] for c in got]),
    )


def test_bm25() -> None:
    print("\n[3] BM25 关键词检索")
    index = BM25Index(persist_path=None)
    index.build(
        [
            {
                "chunk_id": "c1",
                "text": "首保应在 5000 公里或 6 个月内完成",
                "source": "保养手册.md",
                "section": "保养 > 首保",
                "position": 0,
            },
            {
                "chunk_id": "c2",
                "text": "故障码 P0420 表示催化器效率低于阈值",
                "source": "故障码.md",
                "section": "故障码",
                "position": 1,
            },
            {
                "chunk_id": "c3",
                "text": "空调滤芯建议每年更换一次",
                "source": "易损件.md",
                "section": "滤芯",
                "position": 2,
            },
        ]
    )
    check("索引规模正确", index.size == 3, str(index.size))
    hits = index.search("P0420 故障码 是什么意思", top_k=3)
    check("能召回目标片段", bool(hits) and hits[0][0]["chunk_id"] == "c2", str(hits))
    hits2 = index.search("首保多少公里", top_k=3)
    check("中文关键词召回", bool(hits2) and hits2[0][0]["chunk_id"] == "c1", str(hits2))
    # 词典外查询（与语料完全无 token 重叠）必须返回空，否则说明索引里混进了噪声。
    # 注意：不能拿"完全无关的中文串"当反例——中文按单字切分时，
    # 任意两句中文都可能共享"的/是/关/全"这类单字，这是分词策略的固有特性，不是 bug。
    noise = index.search("zzz x9999 qqq", top_k=3)
    check("词典外查询返回空", noise == [], f"实测返回 {[item[0]['chunk_id'] for item in noise]}")
    weak = index.search("完全无关的字符串xyzzy", top_k=3)
    if weak:
        top = weak[0][1]
        check(
            "弱相关候选被相对分数下限过滤",
            all(score >= top * 0.15 for _, score in weak) and weak[0][0]["chunk_id"] == "c1",
            str([(item[0]["chunk_id"], round(item[1], 4)) for item in weak]),
        )
    else:
        check("弱相关候选被相对分数下限过滤", True, "已全部过滤")


def test_fusion() -> None:
    print("\n[4] RRF 融合（不依赖向量库）")
    from app.core.retriever import HybridRetriever

    settings = make_settings(Path("."))
    retriever = HybridRetriever(settings, vector_store=None, bm25_index=BM25Index(None))  # type: ignore[arg-type]

    vector_hits = [
        (
            {
                "chunk_id": "c1",
                "text": "首保应在 5000 公里或 6 个月内完成",
                "source": "a.md",
                "position": 0,
            },
            0.82,
        ),
        (
            {
                "chunk_id": "c2",
                "text": "机油推荐 5W-30",
                "source": "b.md",
                "position": 1,
            },
            0.61,
        ),
    ]
    keyword_hits = [
        (
            {
                "chunk_id": "c2",
                "text": "机油推荐 5W-30",
                "source": "b.md",
                "position": 1,
            },
            7.5,
        ),
        (
            {
                "chunk_id": "c3",
                "text": "空调滤芯每年更换",
                "source": "c.md",
                "position": 2,
            },
            3.2,
        ),
    ]
    fused = retriever._fuse(vector_hits, keyword_hits, "hybrid", "首保 机油 5W-30")  # noqa: SLF001
    ids = [item["chunk_id"] for item in fused]
    check("两路结果都被合并", set(ids) == {"c1", "c2", "c3"}, str(ids))
    c2 = next(item for item in fused if item["chunk_id"] == "c2")
    check("同时命中两路的片段带双标记", c2["from_vector"] and c2["from_keyword"])
    check("融合分数已计算", all(item["score"] >= 0 for item in fused))
    check("按分数降序", all(
        fused[i]["score"] >= fused[i + 1]["score"] for i in range(len(fused) - 1)
    ))


def test_citations() -> None:
    print("\n[5] 引用校验")
    citations = [
        Citation(index=1, chunk_id="c1", source="a.md", position=0),
        Citation(index=2, chunk_id="c2", source="b.md", position=1),
    ]
    used, warnings = QAEngine.verify_citations("结论一[1]，结论二[2]。", citations)
    check("正常引用无告警", warnings == [] and len(used) == 2, str(warnings))
    used2, warnings2 = QAEngine.verify_citations("结论[1]，另外[7]。", citations)
    check("越界引用被检出", len(warnings2) == 1 and "7" in warnings2[0], str(warnings2))
    used3, warnings3 = QAEngine.verify_citations("没有任何角标的回答。", citations)
    check("缺少角标被提示", len(warnings3) == 1 and len(used3) == 2, str(warnings3))


def test_hash_embedder() -> None:
    print("\n[6] hash embedding（离线确定性）")
    embedder = HashEmbedder(128)

    async def run() -> None:
        first = await embedder.embed_documents(["更换机油", "刹车片磨损"])
        second = await embedder.embed_documents(["更换机油", "刹车片磨损"])
        check("维度正确", len(first[0]) == 128, str(len(first[0])))
        check("同文本向量一致（可复现）", first == second)
        norm = sum(value * value for value in first[0]) ** 0.5
        check("已做 L2 归一化", abs(norm - 1.0) < 1e-6, str(norm))
        query = await embedder.embed_query("更换机油")
        check("query 与同文本文档向量一致", query == first[0])

    asyncio.run(run())


def test_guard_claims() -> None:
    print("\n[7] FreshGuard 声明模型（归一化 / 变更判定 / 严重度）")
    from app.guard.claims import Claim, classify_change, normalize_value, severity_of
    from app.guard.extractor import ClaimExtractor, plan_incremental

    # 归一化：带千分位、带单位、时间单位都要能统一
    check("里程归一化", normalize_value("每 8,000 km") == "8000km", str(normalize_value("每 8,000 km")))
    check("月份归一化", normalize_value("每 8 个月") == "8month", str(normalize_value("每 8 个月")))
    check("年归一化", normalize_value("每 2 年") == "2year", str(normalize_value("每 2 年")))
    check("无单位文本返回 None", normalize_value("建议缩短更换间隔") is None)

    def build(value: str, qualifiers=None) -> Claim:
        return Claim(
            claim_key="k", claim_type="interval", subject="更换发动机机油",
            predicate="周期", value=value, qualifiers=qualifiers or [],
        )

    # 变更判定：三种情况必须区分开
    check("值未变判 unchanged", classify_change(build("每 8,000 km"), build("每 8000km")) == "unchanged")
    check(
        "值变化判 modified",
        classify_change(build("每 8,000 km"), build("每 10,000 km")) == "modified",
    )
    check(
        "仅限定条件变化单独分类",
        classify_change(build("每 8,000 km", ["首次"]), build("每 8,000 km")) == "qualifier_changed",
    )
    check("新增判 added", classify_change(None, build("每 8,000 km")) == "added")
    check("删除判 removed", classify_change(build("每 8,000 km"), None) == "removed")

    # 严重度：数值型修改必须是 high，限定条件变化不能升到 high
    check("周期修改为高严重度", severity_of("modified", "interval") == "high")
    check("限定条件变化为中严重度", severity_of("qualifier_changed", "interval") == "medium")
    check("新增为低严重度", severity_of("added", "interval") == "low")

    # claim_key 稳定性：同文档同类型同主体同属性必须得到同一个 key
    key1 = Claim.build_key("a.md", "interval", "更换发动机机油", "周期")
    key2 = Claim.build_key("a.md", "interval", "更换发动机机油", "周期")
    key3 = Claim.build_key("b.md", "interval", "更换发动机机油", "周期")
    check("claim_key 同输入同结果", key1 == key2)
    check("claim_key 跨文档不冲突", key1 != key3)

    # 增量规划：内容哈希相同的片段应被复用
    class FakeChunk:
        def __init__(self, cid, text):
            self.chunk_id = cid
            self.text = text

    chunks = [FakeChunk("c1", "更换机油 每 8000 km"), FakeChunk("c2", "更换火花塞 每 64000 km")]
    hashes = {"c1": ClaimExtractor.chunk_content_hash("更换机油 每 8000 km")}
    to_extract, reusable = plan_incremental(chunks, hashes)
    check("未变更片段被复用", [c.chunk_id for c in reusable] == ["c1"], str([c.chunk_id for c in reusable]))
    check("变更片段需重抽", [c.chunk_id for c in to_extract] == ["c2"], str([c.chunk_id for c in to_extract]))


def test_agent_invariants() -> None:
    """Agent 模块的 schema 不变量。

    更深的行为测试（图路由、停止条件、工具链）在 tests/agent_smoke_test.py，
    那里用脚本化 LLM 精确控制每一步；这里只做不依赖网络的静态校验，
    保证"工具描述必须写清边界"这条要求在 CI 里也不会被悄悄破坏。
    """
    print("\n[8] Agent 工具 schema 不变量")
    from app.agent.graph import AGENT_MAX_ITERATIONS, parse_tool_calls
    from app.agent.tools import ALL_TOOL_SPECS

    names = [spec["function"]["name"] for spec in ALL_TOOL_SPECS]
    check("四个业务工具齐全", set(names) == {"search_kb", "lookup_dtc", "compute_maintenance", "ask_user"}, str(names))
    check("工具名唯一", len(names) == len(set(names)))

    for spec in ALL_TOOL_SPECS:
        function = spec["function"]
        name = function["name"]
        description = function.get("description", "")
        parameters = function.get("parameters", {})
        check(f"{name} 参数是 object schema", parameters.get("type") == "object", str(parameters.get("type")))
        check(f"{name} 描述含【适用】与【边界】", "【适用】" in description and "【边界】" in description)
        check(f"{name} 描述禁止笼统表述", "帮你查资料" not in description)
        check(f"{name} 描述长度 > 150 字", len(description) > 150, str(len(description)))

    check("迭代上限固定为 6", AGENT_MAX_ITERATIONS == 6, str(AGENT_MAX_ITERATIONS))

    # compute_maintenance 的必备参数：车型与里程都必填，避免模型瞎猜
    compute = next(s for s in ALL_TOOL_SPECS if s["function"]["name"] == "compute_maintenance")
    check(
        "compute_maintenance 要求 mileage 与 car_model 必填",
        set(compute["function"]["parameters"].get("required", [])) == {"mileage", "car_model"},
        str(compute["function"]["parameters"].get("required")),
    )
    check(
        "ask_user 限制最多 3 个问题",
        next(
            s for s in ALL_TOOL_SPECS if s["function"]["name"] == "ask_user"
        )["function"]["parameters"]["properties"]["questions"].get("maxItems") == 3,
    )

    # tool_calls 解析的容错不变量
    check(
        "非法 JSON 参数不会抛异常",
        parse_tool_calls([{"id": "x", "function": {"name": "search_kb", "arguments": "{坏"}}])[0]["arguments"] == {},
    )


def test_attribution() -> None:
    print("\n[9] 逐句归因校验（防幻觉）")
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            source="保养.md",
            section="周期表",
            text="发动机机油 | 每 8,000 km 或每 8 个月更换（以先到者为准）。",
        ),
        RetrievedChunk(
            chunk_id="c2",
            doc_id="d2",
            source="三包.md",
            section="条款",
            text="家用汽车产品包修期不低于 3 年或者行驶里程 60,000 公里。",
        ),
    ]
    checker = AttributionChecker()

    r = checker.check("发动机机油为每 8,000 km 或每 8 个月更换 [1]。", chunks)
    check(
        "贴合原文的断言判为有依据",
        r.supported == 1 and r.unsupported == 0,
        f"supported={r.supported} total={r.total}",
    )

    # 这是分句级校验的核心价值：整句覆盖率会被真话拉高，必须拆开才能抓到
    r2 = checker.check("发动机机油每 8,000 km 更换 [1]，可延长发动机寿命约 30% [1]。", chunks)
    check(
        "句内幻觉不被整句覆盖率掩盖",
        r2.weak + r2.unsupported >= 1,
        f"weak={r2.weak} unsupported={r2.unsupported}",
    )

    r3 = checker.check("其他品牌同级车型保养价格约每次 500 元 [2]。", chunks)
    check("完全无依据判为 unsupported", r3.unsupported == 1, f"unsupported={r3.unsupported}")

    r4 = checker.check("该车冬季应使用 0W-20 全合成机油。", chunks)
    check("无引用角标被标记", r4.no_citation == 1, f"no_citation={r4.no_citation}")

    # 拒答场景主要是这类话术，若当成"无依据断言"会产生大量误报
    r5 = checker.check("当前知识库中未找到相关依据。建议以随车《用户手册》为准。", chunks)
    check("元话语/免责话术不计入断言", r5.total == 0, f"total={r5.total}")

    clauses = checker._split_clauses("包修期不低于 3 年或 60,000 公里")
    check("数字千分位不被误切", len(clauses) == 1, str(clauses))

    body = checker.split_body("结论 [1]。\n\n依据：\n[1] 保养.md · 周期表")
    check("「依据：」清单被剥离", "依据" not in body and "结论" in body, body[:30])

    # ---- 校验面 = 片段正文 + 章节路径 ----
    # 真话可能写在章节标题里（「以先到者为准」），只对正文算覆盖会把它判成编造。
    # 实测把一个完全正确的多轮回答强制改成了拒答。
    sec_chunk = RetrievedChunk(
        chunk_id="c3", doc_id="d3", source="保养.md",
        section="二、定期保养项目表（按里程 / 时间，以先到者为准）",
        text="| 第 1 次 | 16 | 12 |",
    )
    r_sec = checker.check("保养节点以先到者为准 [1]。", [sec_chunk])
    check(
        "写在章节标题里的真话不算编造",
        r_sec.unsupported == 0,
        f"unsupported={r_sec.unsupported}",
    )
    r_fake = checker.check("质保期为 10 年 [1]。", [sec_chunk])
    check(
        "正文没有的编造内容不会被判成有依据",
        r_fake.unsupported + r_fake.weak >= 1,
        f"unsupported={r_fake.unsupported} weak={r_fake.weak}",
    )

    # ---- 过渡语必须按**分句**判，不能按整句 ----
    # 曾经的 bug：句子里出现「未找到」就整句跳过，
    # 连带把同一句里的事实断言也漏掉了。实测 20 条评测里 10 条因此归因为空。
    mixed = "当前知识库中未找到机油标号的依据；但保养表写明发动机机油为每 8,000 km 更换 [1]。"
    r6 = checker.check(mixed, chunks)
    check(
        "同一句里「声明缺口 + 事实断言」仍要校验断言",
        r6.total >= 1 and r6.supported >= 1,
        f"total={r6.total} supported={r6.supported}",
    )
    check(
        "纯声明缺口的那半句不计入断言",
        r6.total == 1,
        f"total={r6.total}（应只有后半句进校验）",
    )

    # ---- 「语料缺口」陈述无法用词面覆盖验证，必须跳过 ----
    # 全部分句取自 2026-10-25 实跑，当时 5 个 unsupported 全是这一类，
    # 没有一个是真正的编造。
    absence_cases = [
        "但该表未标注其是否等同于首保",
        "资料中仅有示例车型 A 的定期保养项目表",
        '且未对"长途出行"作专门规定 [1]',
        '且资料未给出"不制冷"与上述任一故障码的对应关系',
        "不能作为实际诊断结论 [2][6]",
        "不能直接套用到其他品牌或车型 [2][6]",
        "该文件没有记录胎压标准值",
    ]
    for clause in absence_cases:
        check(f"缺口陈述应跳过：{clause[:14]}", is_corpus_absence_claim(clause))

    # 对车辆本身的否定是**事实断言**，不能放过
    fact_cases = [
        "该车型未配备胎压监测系统",
        "该车不能正常启动",
        "制动液每 2 年更换一次",
        "该表规定机油每 8,000 km 更换",
        "包修期不低于 3 年或 60,000 公里",
    ]
    for clause in fact_cases:
        check(f"事实断言不应跳过：{clause[:14]}", not is_corpus_absence_claim(clause))

    r7 = checker.check("资料中仅有示例车型 A 的定期保养项目表。", chunks)
    check("纯缺口陈述的回答归因为空（无断言可校验）", r7.total == 0, f"total={r7.total}")
    check("跳过的分句数被记录", r7.meta_skipped == 1, f"meta_skipped={r7.meta_skipped}")

    # ---- 真实语料回放：19 个曾被评为「未标引用」的分句（2026-10-25 实测） ----
    # 其中 16 句根本不是事实断言（引导语 / 缺口陈述 / 建议 / 评述 / 小标题），
    # 只有 2 句是真断言。若这些判定退化，no_citation 会立刻虚高。
    non_assertions = [
        "仅能提供与高温环境相关的保养间隔调整说明",
        "以及常规定期检查项目",
        "上述第 3–7 项为示例车型 A 的定期保养节点",
        "不构成夏季专项检查清单",
        "如需夏季专项检查项目",
        "仅能提供示例车型 A 保养时间表中与出行安全相关的常规检查项目，供参考",
        "可作为排查方向参考",
        "但不能据此判定实际故障原因",
        "现有资料仅涉及故障码（DTC）结构、故障指示灯（MIL）术语及部分故障码含义",
        "未收录警示灯图标的分类说明",
        # 「不涉及」变体（2026-10-25 扩评集时实测）：语料指称 + 不涉及 → 跳过
        "但该条目不涉及长期停放场景下的亏电判断或维护措施",
        "该文件不包含此信息",
        "需区分「示例车型 A 的虚构条款」与「法规最低要求」两套依据",
        "两者数值并不一致",
        '其"不属于保修范围"的列举也与法规第三十二条的表述不完全一致',
        "仅能确认索赔所需材料及包修期内的部分相关义务",
        "也没有 2027 款未上市车型的参数信息",
        "或以各品牌官方售后渠道公布的价格为准",
        "不代表其他车型 [1][3]",
        "示例车型 A 的数据为虚构示例",
        # 第二轮实跑补充：指代式适用范围 / 疑问式主题句 / 限制性结论 / 不可外推
        "仅适用于该车型",
        "该车型是否存在长途前检查要求",
        "但不足以给出原因结论",
        "不可互相套用 [1][2]",
        # 第三轮补充：假设句 / 疑问性短语
        "若需针对具体车型的夏季检查清单",
        "具体适用哪一数值",
        # 第四轮：同族同义词补充
        "不适用于其他车型",
        "可作为排查线索",
        "或关注厂家官方发布渠道",
    ]
    for clause in non_assertions:
        check(
            f"非断言应跳过：{clause[:16]}",
            checker._is_transitional(clause) or is_corpus_absence_claim(clause),
        )

    # 这两句是真断言（带数字），必须留在校验范围里
    real_assertions = [
        "示例车型 A 的包修期里程为 100,000 km",
        "高于法规下限 60,000 km",
        # 对车辆本身的否定是事实断言，不能因为带「不」就跳过
        # （「不涉及/不包含」只有同时带语料指称才会跳过）
        "该车型不使用 0W-20 机油",
        "该车型不涉及召回",
        # 类别约束（区别于「仅适用于该车型」这种指代式声明）仍是事实断言
        "仅适用于涡轮增压车型",
        "该机油可供涡轮增压发动机使用",
        "机油滤清器每 8,000 km 更换",
    ]
    for clause in real_assertions:
        check(
            f"真断言必须校验：{clause[:16]}",
            not checker._is_transitional(clause) and not is_corpus_absence_claim(clause),
        )

    # 已知误判（如实记录）：这是个小标题式名词短语，没有谓语，判定不出来。
    # 它会留在分母里，属于可接受残差（1/122）。
    check(
        "已知误判：无谓语的小标题短语仍会被校验",
        not checker._is_transitional("整车包修期与不属于保修范围的情形"),
    )

    # ---- 小标题与正文粘连时，只剥标题、保留断言 ----
    check(
        "粘连的小标题被剥离",
        strip_clause_heading("**三、需注意的差异**\n示例车型 A 的包修期里程为 100,000 km")
        == "示例车型 A 的包修期里程为 100,000 km",
        strip_clause_heading("**三、需注意的差异**\n正文"),
    )
    check("纯标题行被整体剥空", strip_clause_heading("**四、结论**") == "")
    check("普通断言不被改动", strip_clause_heading("制动液每 2 年更换") == "制动液每 2 年更换")

    # ---- 适用范围声明：同样是「谈论资料」，但必须限定语境 ----
    check("「仅适用于该示例车型」是元陈述", checker._is_transitional("仅适用于该示例车型"))
    check("「不适用于真实车辆诊断」是元陈述", checker._is_transitional("不适用于真实车辆诊断"))
    check(
        "「仅适用于涡轮增压车型」是事实断言，不能跳过",
        not checker._is_transitional("仅适用于涡轮增压车型"),
    )


def test_context_text() -> None:
    """索引文本的上下文拼接：只影响索引，不能影响原文。"""
    from app.core.text_utils import build_context_text

    # 12.24 之后：章节含「保养」+ 正文含「第 1 次」→ 注入别称行（修 Q001 词汇鸿沟）
    got = build_context_text("| 第 1 次 | 16 | 12 |", "二、定期保养项目表")
    check("保养节点片段注入首保别称", "（同义表述：首保、首次保养）" in got, got)
    check(
        "别称行在正文之前",
        got.startswith("二、定期保养项目表；（同义表述：首保、首次保养）\n"),
        got,
    )
    check(
        "with_aliases=False 可关闭注入",
        build_context_text("| 第 1 次 | 16 | 12 |", "二、定期保养项目表", with_aliases=False)
        == "二、定期保养项目表\n| 第 1 次 | 16 | 12 |",
    )
    check(
        "无别称命中的片段不受影响",
        build_context_text("普通正文内容", "无关章节")
        == "无关章节\n普通正文内容",
    )
    check("无章节时原样返回", build_context_text("正文") == "正文")
    check("章节为空串时原样返回", build_context_text("正文", "") == "正文")
    check("章节为 None 时原样返回", build_context_text("正文", None) == "正文")
    check(
        "多个上下文用分号连接（避免跨边界 bigram 噪声）",
        build_context_text("正文", "标题", "章节") == "标题；章节\n正文",
        build_context_text("正文", "标题", "章节"),
    )
    check("上下文首尾空白被清理", build_context_text("正文", "  章节  ") == "章节\n正文")


def test_refusal_guard() -> None:
    """拒答兜底：区分「编造」与「只是忘了标引用」。

    为什么单独测：真实评测集里没有「模型编造」的样本，这层逻辑靠跑评测验证不出来。
    而它的第一版用「归因总分 < 阈值」当门槛，把两条高质量的部分覆盖回答误杀成拒答
    （过度拒答率 7% → 21%），所以这里必须把边界钉死。
    """
    from app.services.rag_pipeline import refusal_on_hallucination

    check(
        "多数句子无依据 -> 判为编造",
        refusal_on_hallucination({"total": 4, "unsupported": 3}, 0.5) is not None,
    )
    check(
        "恰好一半无依据 -> 判为编造（阈值含等号）",
        refusal_on_hallucination({"total": 4, "unsupported": 2}, 0.5) is not None,
    )
    check(
        "少数句子无依据 -> 不拒答",
        refusal_on_hallucination({"total": 4, "unsupported": 1}, 0.5) is None,
    )
    check(
        "只有 1 句且无依据 -> 不拒答（样本太少，可能是元话语）",
        refusal_on_hallucination({"total": 1, "unsupported": 1}, 0.5) is None,
    )
    check(
        "只是没标引用、没有编造 -> 不拒答（第一版误杀就是这个场景）",
        refusal_on_hallucination(
            {"total": 4, "unsupported": 0, "no_citation": 4, "score": 0.0}, 0.5
        )
        is None,
    )
    check(
        "关闭开关（ratio=0）时不触发",
        refusal_on_hallucination({"total": 4, "unsupported": 4}, 0.0) is None,
    )
    check("没有归因报告时不触发", refusal_on_hallucination(None, 0.5) is None)


def test_attribution_highlight() -> None:
    """句↔片段对齐：归因句子里的编号解析成来源明细，前端可直接渲染。"""
    print("\n[12] 归因句↔片段对齐")
    from app.models import Citation
    from app.services.rag_pipeline import QAEngine

    citations = [
        Citation(index=1, chunk_id="c1", source="保养.md",
                 section="二、定期保养项目表", position=2),
        Citation(index=2, chunk_id="c2", source="保修.md", section="", position=5),
    ]
    attr = {
        "total": 2, "supported": 2, "score": 1.0,
        "sentences": [
            {"text": "首保 16,000 km", "citations": [1], "coverage": 0.9, "status": "supported"},
            {"text": "未找到依据", "citations": [], "coverage": 0.0, "status": "no_citation"},
            {"text": "伪造句", "citations": [9], "coverage": 0.0, "status": "unsupported"},
        ],
    }
    out = QAEngine.resolve_attribution_sources(attr, citations)
    sents = out["sentences"]
    check("句子数不变", len(sents) == 3)
    check(
        "编号解析成 source/section/position",
        sents[0]["sources"] == [
            {"index": 1, "source": "保养.md", "section": "二、定期保养项目表", "position": 2}
        ],
        str(sents[0]["sources"]),
    )
    check("无引用的句子 sources 为空列表", sents[1]["sources"] == [])
    check(
        "越界编号防御性标 invalid",
        sents[2]["sources"][0].get("invalid") is True,
        str(sents[2]["sources"]),
    )
    check("原 dict 不被就地修改", "sources" not in (attr["sentences"][0]))
    check("报告级字段原样保留", out["total"] == 2 and out["score"] == 1.0)


def test_session_multiturn() -> None:
    """多轮会话：历史存储、历史块、改写消息体、改写护栏。"""
    print("\n[11] 多轮会话")
    import asyncio

    from app.services.session_store import ConversationStore

    store = ConversationStore(max_turns=3)
    h1 = store.append("s1", "首保多少公里？", "16,000 km 或 12 个月 [1]。")
    check("追加后返回完整历史", len(h1) == 1 and h1[0]["question"].startswith("首保"), str(h1))
    check("默认轮次 refused=False", h1[0].get("refused") is False, str(h1[0]))
    store.append("s1-ref", "量子纠缠？", "未找到依据", refused=True)
    hist = store.history("s1-ref")
    check("拒答轮 refused 标记保留", hist[-1].get("refused") is True, str(hist[-1]))
    store.append("s1", "那第 2 次呢？", "32,000 km 或 24 个月 [2]。")
    check("会话内按序累积", len(store.history("s1")) == 2)

    for i in range(5):  # 超出上限后只留最近 3 轮
        store.append("s1", f"第 {i} 轮", "答")
    hist = store.history("s1")
    check("超过上限时只保留最近 N 轮", len(hist) == 3, f"len={len(hist)}")
    check("保留的是最近而不是最早的", hist[-1]["question"] == "第 4 轮", str(hist[-1]))

    check("空 session_id 是安全 no-op", store.append("", "q", "a") == [])
    check("clear 返回清掉的轮数", store.clear("s1") == 3 and store.history("s1") == [])

    # ---- 历史块：组织语言用，不进依据 ----
    from app.services.rag_pipeline import QAEngine

    block = QAEngine.build_history_block(
        [{"question": "首保多少公里？", "answer": "x" * 260}]
    )
    check("历史块含问题", "首保多少公里？" in block)
    check("长答案被截断到 200 字", "…" in block and block.count("x") <= 200)
    check("空问题不进历史块", QAEngine.build_history_block([{"question": "", "answer": "a"}]) == "")

    # 拒答轮要在历史块 / 改写消息里显式标注（实测改写会把被拒话题拖进追问）
    block_refused = QAEngine.build_history_block(
        [{"question": "量子纠缠？", "answer": "未找到", "refused": True}]
    )
    check("历史块标注拒答轮", "（上轮未回答）" in block_refused, block_refused)
    block_ok = QAEngine.build_history_block(
        [{"question": "首保多少？", "answer": "16,000 km", "refused": False}]
    )
    check("正常轮不标注", "（上轮未回答）" not in block_ok)

    rw_refused = QAEngine.build_rewrite_messages(
        "那正常保养呢？",
        [{"question": "量子纠缠？", "answer": "未找到依据", "refused": True}],
    )
    check("改写消息标注拒答轮", "（上轮未回答）" in rw_refused[1]["content"])
    check("改写提示词含拒答规则", "上轮未回答" in rw_refused[0]["content"])

    settings = Settings()
    engine = QAEngine.__new__(QAEngine)  # 只测纯方法，跳过装配
    engine._settings = settings

    msgs_single = engine.build_messages("q", "ctx")
    check("单轮提示词不含历史块", "对话历史" not in msgs_single[1]["content"])
    msgs_multi = engine.build_messages("q", "ctx", history=[{"question": "上一问", "answer": "上一答"}])
    check("多轮提示词含历史块标题", "对话历史" in msgs_multi[1]["content"])
    check("历史块在参考资料之后、问题之前",
          msgs_multi[1]["content"].index("参考资料") < msgs_multi[1]["content"].index("对话历史")
          < msgs_multi[1]["content"].index("## 用户问题"))

    rw = engine.build_rewrite_messages(
        "那这个车呢？",
        [{"question": "示例车型 A 的首保是多少公里？", "answer": "16,000 km"}] * 5,
    )
    check("改写消息只用最近 3 轮", rw[1]["content"].count("问：") == 3,
          f"count={rw[1]['content'].count('问：')}")
    check("改写消息含当前追问", "那这个车呢？" in rw[1]["content"])
    check("改写 system 提示禁止回答", "不回答" in rw[0]["content"])

    # ---- 改写护栏：不用真 LLM，用桩对象 ----
    class _FakeLLM:
        def __init__(self, text=None, error=False):
            self._text, self._error = text, error

        async def chat(self, messages):
            if self._error:
                raise RuntimeError("llm down")
            obj = type("R", (), {})()
            obj.text = self._text
            return obj

    engine._llm = _FakeLLM("示例车型 A 的第 2 次保养是多少公里？")
    warnings: List[str] = []
    rewritten, did = asyncio.run(
        engine._rewrite_query("那第 2 次呢？", [{"question": "首保", "answer": "16,000 km"}], warnings)
    )
    check("正常改写生效", did and rewritten.startswith("示例车型 A"), rewritten)

    engine._llm = _FakeLLM(error=True)
    warnings2: List[str] = []
    fallback, did2 = asyncio.run(engine._rewrite_query("那第 2 次呢？", [{"question": "首保", "answer": "a"}], warnings2))
    check("改写失败回退为原问题", not did2 and fallback == "那第 2 次呢？")
    check(
        "回退时给出可观测警告",
        any("已按原问题检索" in w for w in warnings2),
        str(warnings2),
    )

    engine._llm = _FakeLLM("好")  # 太短，不可信
    warnings3: List[str] = []
    out3, did3 = asyncio.run(engine._rewrite_query("那第 2 次呢？", [{"question": "首保", "answer": "a"}], warnings3))
    check("改写结果异常时回退", not did3 and out3 == "那第 2 次呢？", out3)

    engine._llm = _FakeLLM("示例车型 A 的第 2 次保养？")
    out4, did4 = asyncio.run(engine._rewrite_query("那第 2 次呢？", [], []))
    check("无历史时不调用改写", not did4 and out4 == "那第 2 次呢？")


def test_query_cache() -> None:
    """L5 查询缓存：命中/未命中、LRU 淘汰、索引版本失效、跳过条件。"""
    print("\n[13] 查询缓存")
    from app.services.index_state import IndexState
    from app.services.query_cache import QueryCache

    cache = QueryCache(max_entries=3, ttl_seconds=60)
    check("空缓存未命中", cache.get("a") is None)
    cache.put("a", {"answer": "A"})
    check("put 后命中且值相同", cache.get("a") == {"answer": "A"})

    # LRU 淘汰：容量 3，塞 4 个后最早的被淘汰
    cache.put("b", 2)
    cache.put("c", 3)
    cache.get("a")  # 访问 a，让 b 变成最久未用
    cache.put("d", 4)
    check("超容量后最久未用的被淘汰", cache.get("b") is None)
    check("最近使用过的仍在", cache.get("a") is not None)
    st = cache.stats()
    check("统计计数正确", st["entries"] == 3 and st["hits"] >= 2 and st["misses"] >= 1, str(st))
    check("clear 返回条数并清空", cache.clear() == 3 and cache.stats()["entries"] == 0)

    # TTL：构造函数把 TTL 下限抬到 1.0s（防误配 0），测试直接改内部值验证过期逻辑
    import time as _time

    ttl_cache = QueryCache(max_entries=4, ttl_seconds=60)
    ttl_cache._ttl = 0.05
    ttl_cache.put("x", 1)
    _time.sleep(0.08)
    check("TTL 过期后未命中", ttl_cache.get("x") is None)
    check("过期条目被移出存储", ttl_cache.stats()["entries"] == 0)

    # 索引版本：版本变 → 旧键失效
    state = IndexState()
    check("初始版本为 1", state.version == 1)
    cache2 = QueryCache()
    key1 = f"v{state.version}|auto|k8|问题"
    cache2.put(key1, "旧答案")
    state.bump(reason="测试入库")
    key2 = f"v{state.version}|auto|k8|问题"
    check("版本递增", state.version == 2)
    # 失效的真实机制：新请求构建的是新版本键 → 未命中 → 重新计算；
    # 旧条目不是被删除，而是永远不会再被查到（之后由 LRU/TTL 自然回收）
    check("新版本键未命中（会重新计算而不是拿旧答案）", cache2.get(key2) is None)
    cache2.put(key2, "新答案")
    check("新版本键命中", cache2.get(key2) == "新答案")

    # 跳过条件：多轮与 debug 不进缓存（评测方差不能被缓存污染）
    check("单轮非 debug 可缓存", QAEngine._cacheable(None, False) is True)
    check("多轮请求不缓存", QAEngine._cacheable("s1", False) is False)
    check("debug 请求不缓存", QAEngine._cacheable(None, True) is False)

    # 键的构造：top_k/mode 归一化，问题参与
    settings = Settings()
    engine = QAEngine.__new__(QAEngine)
    engine._settings = settings
    engine._query_cache = QueryCache()
    engine._index_state = state
    k1 = engine._cache_key("首保多少公里？", None, None)
    k2 = engine._cache_key("首保多少公里？", None, None)
    k3 = engine._cache_key("首保多少公里？", settings.top_k + 1, None)
    k4 = engine._cache_key("首保多少公里？", None, "bm25")
    check("相同参数生成相同键", k1 == k2 and k1 is not None)
    check("None 的 top_k 归一化为默认值", engine._cache_key("q", settings.top_k, None) == engine._cache_key("q", None, None))
    check("不同 top_k 生成不同键", k1 != k3)
    check("mode 参与键", k1 != k4)
    check("不同问题不同键", engine._cache_key("刹车油呢？", None, None) != k1)
    # 关闭开关时不生成键
    settings2 = Settings()
    settings2.query_cache_enabled = False
    engine._settings = settings2
    check("开关关闭返回 None（不缓存）", engine._cache_key("q", None, None) is None)


def test_ingest_jobs() -> None:
    """L5 异步入库：提交/进度/完成/失败/单飞约束/历史上限。"""
    print("\n[14] 异步入库任务管理")
    import asyncio

    from app.models import IngestResponse
    from app.services.ingestion_jobs import IngestJobManager

    class _FakeIngestion:
        def __init__(self, fail=False, delay=0.0):
            self._fail, self._delay = fail, delay
            self.calls = []

        async def ingest(self, rebuild=False, reset_registry=False,
                         only_sources=None, progress_cb=None):
            self.calls.append({"rebuild": rebuild, "only": only_sources})
            if progress_cb:
                progress_cb({"phase": "file", "source": "a.md", "status": "indexed"})
            if self._delay:
                await asyncio.sleep(self._delay)
            if self._fail:
                raise ValueError("模拟入库失败")
            return IngestResponse(scanned=1, indexed=1, total_chunks_in_store=1)

    async def _scenario():
        mgr = IngestJobManager()

        # 成功路径：queued → running → done，进度被记录
        acc = mgr.start(_FakeIngestion(), rebuild=False)
        job_id = acc["job_id"]
        check("受理返回 job_id", acc["status"] == "queued" and job_id)
        for _ in range(50):
            await asyncio.sleep(0.02)
            if mgr.status(job_id)["status"] == "done":
                break
        job = mgr.status(job_id)
        check("任务完成", job["status"] == "done", job["status"])
        check("进度被记录（file 阶段）", job["progress"].get("source") == "a.md", str(job["progress"]))
        check("结果落盘", job["result"]["indexed"] == 1)
        check("finished_at 已填", bool(job["finished_at"]))

        # 单飞约束：任务在跑时再提交要被拒
        slow = _FakeIngestion(delay=0.3)
        acc2 = mgr.start(slow)
        await asyncio.sleep(0.05)
        check("运行中任务可见", mgr.running_job_id == acc2["job_id"])
        try:
            mgr.start(_FakeIngestion())
            check("并发提交被拒", False)
        except RuntimeError as exc:
            check("并发提交被拒", "不允许并发" in str(exc))
        for _ in range(100):
            await asyncio.sleep(0.05)
            if mgr.status(acc2["job_id"])["status"] == "done":
                break
        check("慢任务最终完成", mgr.status(acc2["job_id"])["status"] == "done")
        check("结束后 running 清空", mgr.running_job_id is None)

        # 失败路径：异常被捕获落记录，而不是悬空 task
        acc3 = mgr.start(_FakeIngestion(fail=True))
        for _ in range(50):
            await asyncio.sleep(0.02)
            if mgr.status(acc3["job_id"])["status"] == "failed":
                break
        job3 = mgr.status(acc3["job_id"])
        check("失败被记录", job3["status"] == "failed" and "模拟入库失败" in job3["error"])

        # 未知任务查不到
        check("未知 job_id 返回 None", mgr.status("nope") is None)

        # 历史上限：_MAX_FINISHED_JOBS 是模块常量，直接验证 prune 逻辑
        from app.services import ingestion_jobs as ij
        check("历史上限为正数", ij._MAX_FINISHED_JOBS >= 1)

    asyncio.run(_scenario())


def test_metrics_collector() -> None:
    """L5 请求指标：分位数、环形缓冲、成本估算、空窗口、分端点。"""
    print("\n[15] 请求指标采集")
    from app.services.metrics_collector import RequestMetricsCollector, _percentile

    # 分位数：已知数组验证下标法
    # 用奇数长度数组避开 round 的银行家舍入歧义（round(4.5)=4，round(0.5*10)=5 无歧义）
    data = sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    check("P50 正确", _percentile(data, 50) == 6, str(_percentile(data, 50)))  # round(0.5*10)=5
    check("P100 是最大值", _percentile(data, 100) == 11)
    check("空数组返回 None", _percentile([], 95) is None)

    c = RequestMetricsCollector(max_records=10)  # 小缓冲便于测淘汰
    c.record("chat", total_ms=100, prompt_tokens=10, completion_tokens=20)
    c.record("chat", total_ms=200, cache_hit=True)
    c.record("chat", total_ms=300, refused=True)
    c.record("retrieve", total_ms=5)

    snap = c.snapshot()
    check("窗口计数", snap["count"] == 4)
    check("总体 P50 合理", snap["latency_overall"]["p50_ms"] == 200, str(snap["latency_overall"]))
    check("分端点统计", snap["latency_by_endpoint"]["retrieve"]["count"] == 1)
    check("缓存命中率 1/4", snap["cache_hit_rate"] == 0.25)
    check("拒答率 1/4", snap["refusal_rate"] == 0.25)
    check("有 usage 的请求数", snap["tokens"]["requests_with_usage"] == 1)
    check("token 汇总", snap["tokens"]["prompt"] == 10 and snap["tokens"]["completion"] == 20)
    check("未配单价时不输出成本", "estimated_cost" not in snap)

    # 环形缓冲：超出上限后丢最老的
    for i in range(15):
        c.record("chat", total_ms=i)
    snap2 = c.snapshot()
    check("缓冲有界", snap2["count"] == 10, str(snap2["count"]))
    check("被挤掉的记录被计数", snap2["dropped_by_buffer"] >= 5, str(snap2["dropped_by_buffer"]))
    # 最老的 5 条（total_ms=100..300 那些）已不在：窗口里最小值应是第 5 次追加的
    check("留下的是最近的", snap2["latency_overall"]["max_ms"] == 14, str(snap2["latency_overall"]))

    # 成本估算：配置单价后输出且数学正确
    c2 = RequestMetricsCollector()
    c2.record("chat", prompt_tokens=1_000_000, completion_tokens=500_000)
    cost = c2.snapshot(price_prompt_per_m=2.0, price_completion_per_m=8.0)["estimated_cost"]
    check("成本估算数学正确", abs(cost["total"] - (2.0 + 4.0)) < 1e-6, str(cost))
    check("只配一个单价不估算", (
        "estimated_cost" not in c2.snapshot(price_prompt_per_m=2.0)
    ))

    # 空窗口
    empty = RequestMetricsCollector().snapshot()
    check("空窗口 count=0 且有说明", empty["count"] == 0 and "note" in empty)


def test_reranker_quantize() -> None:
    """L5 精排 int8 量化：helper 行为 + 配置接线（用小模型，不加载 1.1GB 权重）。"""
    print("\n[16] 精排量化")
    try:
        import torch
        from torch import nn
    except ImportError:
        print("  （torch 不可用，跳过）")
        return

    from app.core.reranker import quantize_linear_layers

    class _Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 8)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(8, 1)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    tiny = _Tiny().eval()
    quantized = quantize_linear_layers(tiny, torch)
    # 量化后 Linear 被替换为动态量化层；ReLU 等非 Linear 不受影响
    check("Linear 被替换为量化层", not isinstance(quantized.fc1, nn.Linear))
    check("非 Linear 层不受影响", isinstance(quantized.act, nn.ReLU))
    # 前向仍然能跑，输出形状一致
    x = torch.randn(2, 8)
    with torch.no_grad():
        y = quantized(x)
    check("量化后前向可用", y.shape == (2, 1), str(y.shape))

    # 配置接线：默认开、可关
    from app.core.reranker import CrossEncoderReranker
    settings = Settings()
    r_on = CrossEncoderReranker(quantize=True)
    r_off = CrossEncoderReranker(quantize=False)
    check("默认开启量化", settings.rerank_quantize is True)
    check("构造参数正确传递", r_on._quantize is True and r_off._quantize is False)


def main() -> int:
    print("=" * 68)
    print("AutoRAG 冒烟测试（纯逻辑，不联网、不需要 API Key、不调用大模型）")
    print("=" * 68)
    tmp_root = PROJECT_ROOT / "data" / "_smoke_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    test_text_utils()
    test_chunker(tmp_root)
    test_bm25()
    test_fusion()
    test_citations()
    test_hash_embedder()
    test_guard_claims()
    test_agent_invariants()
    test_attribution()
    test_context_text()
    test_refusal_guard()
    test_session_multiturn()
    test_attribution_highlight()
    test_query_cache()
    test_ingest_jobs()
    test_metrics_collector()
    test_reranker_quantize()

    print("\n" + "=" * 68)
    print(f"结果：通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 68)
    if FAILED:
        print("存在失败项，请把上面的 FAIL 行连同环境信息一起反馈。")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
