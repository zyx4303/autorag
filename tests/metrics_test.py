#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""指标模块单元测试：只用标准库，不联网、不调用模型、不需要任何 Key。

运行：
    python tests/metrics_test.py

为什么单独一个文件而不是并进 smoke_test.py：
metrics 是「所有效果结论的分母」，它的口径错一次，后面每一轮优化结论都是错的。
所以这里的用例比业务代码更细，重点打三类容易错的地方：
1. 数字写法差异（16,000 vs 16000 vs 全角）；
2. nDCG 的理想排序只在召回集内计算（口径错会导致 nDCG 虚高）；
3. 失败分型的判定优先级（顺序错会把检索问题误报成拒答问题）。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core import metrics as M  # noqa: E402

PASSED = 0
FAILED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  [FAIL] {name}" + (f" —— {detail}" if detail else ""))


def near(actual: float, expected: float, tol: float = 1e-3) -> bool:
    return actual is not None and abs(actual - expected) <= tol


# --------------------------------------------------------------------------- #
# 1. 文本规范化
# --------------------------------------------------------------------------- #
def test_normalize() -> None:
    print("[1] 文本规范化")
    check(
        "千分位被去掉",
        M.normalize_number_text("每 16,000 km") == "每 16000 km",
        M.normalize_number_text("每 16,000 km"),
    )
    check(
        "多级千分位一次去净",
        M.normalize_number_text("1,234,567") == "1234567",
        M.normalize_number_text("1,234,567"),
    )
    check(
        "全角数字折成半角",
        M.normalize_number_text("１６０００") == "16000",
        M.normalize_number_text("１６０００"),
    )
    check(
        "匹配形式去掉空白",
        M.normalize_for_match("12 个月") == "12个月",
        M.normalize_for_match("12 个月"),
    )
    check(
        "匹配形式不误删英文空格语义（仅去空白，不做词形处理）",
        M.normalize_for_match("OBD II") == "obdii",
        M.normalize_for_match("OBD II"),
    )


# --------------------------------------------------------------------------- #
# 2. 关键词覆盖（旧实现的主要漏判来源）
# --------------------------------------------------------------------------- #
def test_keyword() -> None:
    print("[2] 关键词覆盖")
    check(
        "千分位差异不应算漏答",
        M.keyword_coverage(["16,000"], "首保是 16000 公里") == 1.0,
        str(M.keyword_coverage(["16,000"], "首保是 16000 公里")),
    )
    check(
        "空格差异不应算漏答",
        M.keyword_coverage(["12 个月"], "或 12个月") == 1.0,
        str(M.keyword_coverage(["12 个月"], "或 12个月")),
    )
    check(
        "部分命中给比例",
        near(M.keyword_coverage(["8,000", "8 个月", "16,000"], "机油每 8000 公里或 8 个月更换"), 2 / 3),
        str(M.keyword_coverage(["8,000", "8 个月", "16,000"], "机油每 8000 公里或 8 个月更换")),
    )
    check("无期望词返回 0.0", M.keyword_coverage([], "任意答案") == 0.0)
    check(
        "真不匹配时确实不命中",
        M.keyword_coverage(["刹车片"], "机油需要更换") == 0.0,
    )
    hits, total = M.keyword_hits(["a", "b"], "只有 a")
    check("命中数与总数可分别取到", (hits, total) == (1, 2), f"{hits}/{total}")


# --------------------------------------------------------------------------- #
# 3. 检索侧：相关性标记与 recall / precision
# --------------------------------------------------------------------------- #
def _chunks(*sources: str) -> list:
    """构造最小可用的召回结果（dict 形式，兼容 pydantic 取值路径）。"""
    return [{"source": source} for source in sources]


def _chunks_with_text(pairs) -> list:
    """构造带正文的召回结果：(source, text) 元组序列。"""
    return [{"source": source, "text": text} for source, text in pairs]


def test_retrieval() -> None:
    print("[3] 检索侧指标")
    flags = M.relevance_flags(["a.md"], _chunks("x.md", "a.md", "b.md"))
    check("相关性按来源文件判定", flags == [0, 1, 0], str(flags))

    flags = M.relevance_flags(["docs/a.md"], _chunks("docs/a.md"))
    check("完整路径可匹配", flags == [1], str(flags))

    flags = M.relevance_flags(["a.md"], _chunks("data/documents/a.md"))
    check("允许只写文件名（basename 匹配）", flags == [1], str(flags))

    recall = M.source_recall_at_k(["a.md", "b.md"], _chunks("a.md", "x.md", "b.md"), k=3)
    check("两个期望来源都被覆盖时 recall=1.0", recall == 1.0, str(recall))

    recall = M.source_recall_at_k(["a.md", "b.md"], _chunks("a.md", "x.md", "b.md"), k=2)
    check("截断到 k 条时只算前 k", recall == 0.5, str(recall))

    recall = M.source_recall_at_k(["a.md"], _chunks("a.md"), k=5)
    check("k 大于返回条数不报错", recall == 1.0, str(recall))

    check(
        "无期望来源返回 None 而非 0.0",
        M.source_recall_at_k([], _chunks("a.md"), k=3) is None,
    )

    check(
        "precision 分母用实际返回条数",
        M.precision_at_k([1, 0, 0], k=5) == round(1 / 3, 4),
        str(M.precision_at_k([1, 0, 0], k=5)),
    )
    check("空结果 precision 为 None", M.precision_at_k([], k=5) is None)


# --------------------------------------------------------------------------- #
# 3b. 片段级相关性（关键短语标注）
# --------------------------------------------------------------------------- #
def test_snippet_relevance() -> None:
    print("[3b] 片段级相关性")
    snips = ["更换火花塞 | 每 64,000 km"]

    # 来源对、但正文不是那一段 —— 这正是文件级口径看不见的失效
    flags = M.relevance_flags(
        ["a.md"],
        _chunks_with_text([("a.md", "发动机机油每 8,000 km 更换")]),
        snips,
    )
    check("来源命中但片段不含关键句 -> 不相关", flags == [0], str(flags))

    flags = M.relevance_flags(
        ["a.md"],
        _chunks_with_text([("a.md", "| 更换火花塞 | 每 64,000 km |")]),
        snips,
    )
    check("来源命中且片段含关键句 -> 相关", flags == [1], str(flags))

    # 千分位/空格差异不应影响命中
    flags = M.relevance_flags(
        ["a.md"],
        _chunks_with_text([("a.md", "更换火花塞 每 64000 km")]),
        snips,
    )
    check("关键句匹配能容忍千分位与空格差异", flags == [1], str(flags))

    # 关键句出现在**别的**文档里，不能算相关
    flags = M.relevance_flags(
        ["a.md"],
        _chunks_with_text([("b.md", "| 更换火花塞 | 每 64,000 km |")]),
        snips,
    )
    check("关键句跨文档出现但来源不对 -> 不相关", flags == [0], str(flags))

    # 未标注时退回文件级
    flags = M.relevance_flags(["a.md"], _chunks_with_text([("a.md", "任意正文")]), [])
    check("未标注关键句时退回文件级", flags == [1], str(flags))

    # recall 在片段级下更严格
    retrieved = _chunks_with_text(
        [("a.md", "无关片段"), ("b.md", "| 更换火花塞 | 每 64,000 km |")]
    )
    file_level = M.source_recall_at_k(["a.md"], retrieved, k=2)
    snip_level = M.source_recall_at_k(["a.md"], retrieved, k=2, expected_snippets=snips)
    check("文件级认为已覆盖", file_level == 1.0, str(file_level))
    check("片段级认为未覆盖（关键段落在别的文件里）", snip_level == 0.0, str(snip_level))

    check("relevance_level 标注 snippet", M.relevance_level(snips) == "snippet")
    check("relevance_level 标注 file", M.relevance_level([]) == "file")
    check("relevance_level 对空串也算 file", M.relevance_level([""]) == "file")


# --------------------------------------------------------------------------- #
# 4. MRR 与 nDCG
# --------------------------------------------------------------------------- #
def test_ranking() -> None:
    print("[4] 排序指标")
    check("第 1 位命中 RR=1.0", M.reciprocal_rank([1, 0, 0]) == 1.0)
    check("第 3 位命中 RR=1/3", near(M.reciprocal_rank([0, 0, 1]), 1 / 3))
    check("无相关结果 RR=0.0", M.reciprocal_rank([0, 0, 0]) == 0.0)
    check("首个相关排名可取", M.first_relevant_rank([0, 0, 1]) == 3)
    check("无相关排名为 None", M.first_relevant_rank([0, 0]) is None)

    check("全部相关 nDCG=1.0", M.ndcg_at_k([1, 1, 0], k=3) == 1.0)
    check("第 1 位命中 nDCG=1.0", M.ndcg_at_k([1, 0, 0], k=3) == 1.0)
    check(
        "第 2 位命中 nDCG=1/log2(3)",
        near(M.ndcg_at_k([0, 1, 0], k=3), 1 / 1.5849625),
        str(M.ndcg_at_k([0, 1, 0], k=3)),
    )
    check(
        "第 3 位命中 nDCG=1/log2(4)=0.5",
        near(M.ndcg_at_k([0, 0, 1], k=3), 0.5),
        str(M.ndcg_at_k([0, 0, 1], k=3)),
    )
    check("无相关结果 nDCG=0.0", M.ndcg_at_k([0, 0, 0], k=3) == 0.0)
    check("空结果 nDCG 为 None", M.ndcg_at_k([], k=3) is None)

    # 这一条是本次改造的核心动机：旧的 source_hit 是布尔值，
    # 上面两种情况的命中率完全相同（都是「命中」），nDCG 才能区分排序优劣。
    top1 = M.ndcg_at_k([1, 0, 0], k=3)
    top3 = M.ndcg_at_k([0, 0, 1], k=3)
    check(
        "nDCG 能区分「第 1 位命中」与「第 3 位命中」（旧指标做不到）",
        top1 > top3,
        f"{top1} vs {top3}",
    )


# --------------------------------------------------------------------------- #
# 5. token F1
# --------------------------------------------------------------------------- #
def test_token_f1() -> None:
    print("[5] 词面一致性")
    check("完全相同 F1=1.0", M.token_f1("更换发动机机油", "更换发动机机油") == 1.0)
    check("毫无重合 F1=0.0", M.token_f1("刹车片磨损", "完全无关内容") == 0.0)
    check("空参考返回 None", M.token_f1("", "任意") is None)
    check("空答案返回 None", M.token_f1("任意", "") is None)
    mid = M.token_f1("更换发动机机油每 8,000 km", "发动机机油每 8000 km 更换")
    check("千分位与语序差异下仍有较高重合", mid is not None and mid > 0.5, str(mid))


# --------------------------------------------------------------------------- #
# 6. 安全比例
# --------------------------------------------------------------------------- #
def test_ratio() -> None:
    print("[6] 安全比例")
    check("分母为 0 返回 None", M.ratio(0, 0) is None)
    check("正常比例", M.ratio(1, 4) == 0.25)
    check("全中为 1.0", M.ratio(3, 3) == 1.0)


# --------------------------------------------------------------------------- #
# 7. 失败分型优先级
# --------------------------------------------------------------------------- #
def test_classify() -> None:
    print("[7] 失败分型")
    label = M.classify_failure
    check("异常最优先", label(error="TimeoutError") == M.ERROR)

    check(
        "负样本正确拒答算通过",
        label(must_refuse=True, refused=True) == M.OK,
    )
    check(
        "负样本正确拒答不因词面 F1 低而被判答案不符",
        label(must_refuse=True, refused=True, f1=0.02, keyword_cov=0.0) == M.OK,
    )
    check(
        "负样本正确拒答但引用越界仍要报格式错误",
        label(must_refuse=True, refused=True, out_of_range=[3]) == M.CITATION_OUT_OF_RANGE,
    )
    check(
        "负样本拿不到拒答结果 -> 未评测，不能算通过",
        label(must_refuse=True, refused=None) == M.NOT_EVALUATED,
    )
    check(
        "该拒却答 -> 漏拒答",
        label(must_refuse=True, refused=False) == M.MISSED_REFUSAL,
    )
    check(
        "有答案却拒 -> 过度拒答",
        label(must_refuse=False, refused=True, substantive=True) == M.WRONG_REFUSAL,
    )
    check(
        "无实质内容却被拒且本应有答案 -> 依据未找到（不是过度拒答）",
        label(
            must_refuse=False,
            refused=True,
            substantive=False,
            expected_sources=["a.md"],
        )
        == M.EVIDENCE_MISS,
    )
    check(
        "无实质内容、无期望来源却被拒 -> 过度拒答",
        label(must_refuse=False, refused=True, substantive=False) == M.WRONG_REFUSAL,
    )
    check(
        "拒答错误优先于检索判断（即使 recall=0）",
        label(must_refuse=True, refused=False, expected_sources=["a.md"], recall_at_k=0.0)
        == M.MISSED_REFUSAL,
    )
    check(
        "一条都没召回到 -> 检索未召回",
        label(expected_sources=["a.md"], recall_at_k=0.0, refused=False) == M.RETRIEVAL_MISS,
    )
    check(
        "召回到但排第 4 -> 排序靠后",
        label(expected_sources=["a.md"], recall_at_k=1.0, first_rank=4) == M.RANKING_LOW,
    )
    check(
        "排第 3 仍在可接受范围",
        label(expected_sources=["a.md"], recall_at_k=1.0, first_rank=3, keyword_cov=1.0) == M.OK,
    )
    check(
        "引用越界",
        label(expected_sources=["a.md"], recall_at_k=1.0, first_rank=1, out_of_range=[7])
        == M.CITATION_OUT_OF_RANGE,
    )
    # ---- 归因不足：只看无依据分句占比，不看总分 ----
    check(
        "无依据分句占比过半 -> 归因不足",
        label(
            expected_sources=["a.md"],
            recall_at_k=1.0,
            first_rank=1,
            attribution_total=5,
            attribution_unsupported=3,
        )
        == M.ATTRIBUTION_WEAK,
    )
    check(
        "恰好 40% 判归因不足（阈值含等号）",
        label(
            expected_sources=["a.md"],
            recall_at_k=1.0,
            first_rank=1,
            attribution_total=5,
            attribution_unsupported=2,
        )
        == M.ATTRIBUTION_WEAK,
    )
    check(
        "12 句里 2 句无依据（17%）不判归因不足",
        label(
            expected_sources=["a.md"],
            recall_at_k=1.0,
            first_rank=1,
            attribution_total=12,
            attribution_unsupported=2,
        )
        == M.OK,
    )
    check(
        "分句太少（2 句 1 句无依据）不做判定",
        label(
            expected_sources=["a.md"],
            recall_at_k=1.0,
            first_rank=1,
            attribution_total=2,
            attribution_unsupported=1,
        )
        == M.OK,
    )
    check(
        "只有 no_citation（忘了标引用）不算编造",
        label(
            expected_sources=["a.md"],
            recall_at_k=1.0,
            first_rank=1,
            attribution_total=10,
            attribution_unsupported=0,
        )
        == M.OK,
    )
    check(
        "检索与归因都正常但关键词覆盖率低 -> 答案不符",
        label(
            expected_sources=["a.md"],
            recall_at_k=1.0,
            first_rank=1,
            attribution_total=5,
            attribution_unsupported=0,
            keyword_cov=0.2,
        )
        == M.ANSWER_MISMATCH,
    )
    check(
        "全部正常 -> 通过",
        label(
            expected_sources=["a.md"],
            recall_at_k=1.0,
            first_rank=1,
            attribution_total=5,
            attribution_unsupported=0,
            keyword_cov=1.0,
            f1=0.8,
        )
        == M.OK,
    )
    check(
        "关键词覆盖率高时，F1 低不判错（F1 不参与分型）",
        label(expected_sources=["a.md"], recall_at_k=1.0, first_rank=1, keyword_cov=1.0, f1=0.05)
        == M.OK,
    )
    check(
        "无期望来源时也不靠 F1 判错",
        label(keyword_cov=1.0, f1=0.05) == M.OK,
    )

    dist = M.distribution([M.OK, M.OK, M.RETRIEVAL_MISS])
    check("分布按次数降序", list(dist.keys())[0] == "通过", str(dist))
    check("分布使用中文可读名", dist.get("检索未召回") == 1, str(dist))


# --------------------------------------------------------------------------- #
# 8. 拒答路径
# --------------------------------------------------------------------------- #
def test_refusal_path() -> None:
    print("[8] 拒答路径")
    check(
        "硬拒答固定话术可识别",
        M.is_refusal_text("当前知识库中未找到与该问题相关的依据，因此不做推测性回答。"),
    )
    check(
        "模型软拒答也能识别",
        M.is_refusal_text("当前知识库中未找到关于刹车片更换标准的相关依据。建议以《用户手册》为准。"),
    )
    check("正常回答不误判", not M.is_refusal_text("机油每 8,000 km 或 8 个月更换 [1]。"))
    check(
        "含「未找到」但属于正常建议时不误判",
        not M.is_refusal_text("若未找到故障码，请先检查蓄电池电压 [1]。"),
    )
    check("refused=True 判为 hard", M.refusal_path(True, "任意") == "hard")
    check(
        "refused=False 但话术是拒答 -> soft",
        M.refusal_path(False, "当前知识库中未找到相关依据。") == "soft",
    )
    check(
        "说明缺口后仍给出信息 -> partial（不算拒答）",
        M.refusal_path(False, "当前知识库中未找到相关依据，但常规保养为 8000 km。", True)
        == "partial",
    )
    check("普通回答 -> none", M.refusal_path(False, "机油每 8000 公里更换。") == "none")

    # 实质信息判定：区分「部分覆盖」与「纯拒答」的唯一可靠信号
    check(
        "有关键词命中即为有实质内容",
        M.has_substantive_answer(["8000"], "常规保养为 8000 km [1]。"),
    )
    check(
        "无关键词命中即无实质内容",
        not M.has_substantive_answer(["16,000"], "当前知识库中未找到首保依据。"),
    )
    check("无期望词时一律视为无实质内容", not M.has_substantive_answer([], "任意回答"))


# --------------------------------------------------------------------------- #
# 9. 拒答极性：负样本「声明缺口」就算履行义务，正样本要「纯拒答」才算没答
# --------------------------------------------------------------------------- #
def test_refusal_polarity() -> None:
    print("[9] 拒答极性判定")
    partial = "当前知识库中未找到长期停放亏电的依据，但保养表要求每 10,000 km 检查蓄电池电压。"
    pure = "当前知识库中未找到关于该问题的相关依据，建议以用户手册为准。"
    normal = "首保为 16,000 km 或 12 个月 [1]。"

    # 负样本：部分披露也算履行拒答义务
    check(
        "负样本 + 声明缺口 + 有实质信息 -> 有效拒答",
        M.effective_refusal(True, False, partial, substantive=True),
    )
    check(
        "负样本 + 纯拒答 -> 有效拒答",
        M.effective_refusal(True, False, pure, substantive=False),
    )
    check(
        "负样本 + 正常作答 -> 不算拒答（漏拒答）",
        not M.effective_refusal(True, False, normal, substantive=True),
    )

    # 正样本：只有纯拒答才算没答
    check(
        "正样本 + 部分披露 -> 不算拒答",
        not M.effective_refusal(False, False, partial, substantive=True),
    )
    check(
        "正样本 + 纯拒答 -> 算拒答",
        M.effective_refusal(False, False, pure, substantive=False),
    )
    check(
        "正样本 + 硬拒答 -> 算拒答（即使带了信息）",
        M.effective_refusal(False, True, partial, substantive=True),
    )

    # 同一段文本在两类样本里结论相反 —— 这正是需要极性的原因
    check(
        "同一文本极性相反：负样本算拒答、正样本不算",
        M.effective_refusal(True, False, partial, True)
        != M.effective_refusal(False, False, partial, True),
    )


# --------------------------------------------------------------------------- #
def main() -> int:
    test_normalize()
    test_keyword()
    test_retrieval()
    test_snippet_relevance()
    test_ranking()
    test_token_f1()
    test_ratio()
    test_classify()
    test_refusal_path()
    test_refusal_polarity()

    print()
    print("=" * 60)
    print(f"通过 {PASSED} 项，失败 {FAILED} 项")
    print("=" * 60)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
