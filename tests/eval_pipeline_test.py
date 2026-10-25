# -*- coding: utf-8 -*-
"""评测链路测试：用假 QA 引擎跑一遍 EvaluationService.run()，验证汇总数字自洽。

不联网、不调模型、不需要 Key。运行：
    python tests/eval_pipeline_test.py

它测的不是单条指标算法（那在 metrics_test.py），而是**组装后的行为**：
分组分母对不对、分型优先级在真实数据结构上是否成立、拒答两条路径是否被正确区分。
这类错误单测抓不到 —— 实测中「拒答成功率 100% 但分型里有 6 条漏拒答」
就是这么漏出来的。
"""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.models import Citation, RetrievedChunk
from app.services.eval_service import EvaluationService

MAINT = "示例-虚构车型A-保养与故障码.md"
DTC = "故障码表：示例车企B-DTC总列表（虚构）.md"
WARRANTY = "保修条款：汽车三包规定全文.md"


def chunk(source, text="片段正文", pos=1):
    return RetrievedChunk(
        chunk_id=f"{source}#{pos}", doc_id=source[:6], source=source,
        title=source, section="测试", position=pos, text=text, score=0.9,
    )


def good_chunk(item, pos=1):
    """生成「含该条关键短语」的召回片段。

    评测集已标注 expected_snippets，相关性判定升级为片段级：
    只有正文里真的出现关键短语，这条召回才算相关。
    所以 mock 不能再返回「片段正文」这种占位文本 —— 它会被判成不相关，这是对的。
    """
    source = (item.get("expected_sources") or [MAINT])[0]
    snippets = item.get("expected_snippets") or ["片段正文"]
    return chunk(source, text=snippets[0], pos=pos)


def make_outcome(item):
    """按条目 id 造不同场景，覆盖每一类分型。"""
    qid = item["id"]
    sources = item.get("expected_sources") or [MAINT]

    if qid == "Q001":  # 正常：关键片段第 1 位命中 + 关键词齐全
        return dict(answer="首保为 16000 公里或 12 个月 [1]。",
                    citations=[Citation(index=1, chunk_id="c1", source=sources[0])],
                    refused=False, retrieved_raw=[good_chunk(item)],
                    attribution={"total": 1, "score": 0.95, "unsupported": 0})
    if qid == "Q002":  # 检索未召回：召回到的全是无关来源
        return dict(answer="无法确定 [1]。",
                    citations=[Citation(index=1, chunk_id="c1", source=WARRANTY)],
                    refused=False, retrieved_raw=[chunk(WARRANTY)],
                    attribution={"total": 1, "score": 0.9, "unsupported": 0})
    if qid == "Q003":  # 负样本正确拒答
        return dict(answer="未找到相关依据，建议以用户手册为准。", citations=[],
                    refused=True, retrieved_raw=[chunk(WARRANTY)],
                    attribution={"total": 0, "score": 1.0, "unsupported": 0})
    if qid == "Q006":  # 负样本却作答 -> 漏拒答
        return dict(answer="刹车片磨损到 2mm 需更换 [1]。",
                    citations=[Citation(index=1, chunk_id="c1", source=MAINT)],
                    refused=False, retrieved_raw=[chunk(MAINT)],
                    attribution={"total": 1, "score": 0.9, "unsupported": 0})
    if qid == "Q008":  # 负样本正确拒答
        return dict(answer="资料中未找到该车型的冷却液补充说明。", citations=[],
                    refused=True, retrieved_raw=[],
                    attribution={"total": 0, "score": 1.0, "unsupported": 0})
    if qid == "Q014":  # 负样本正确拒答
        return dict(answer="未找到官方建议的低温启动步骤。", citations=[],
                    refused=True, retrieved_raw=[],
                    attribution={"total": 0, "score": 1.0, "unsupported": 0})
    if qid == "Q017":  # 负样本正确拒答
        return dict(answer="资料中没有各警示灯的完整含义表。", citations=[],
                    refused=True, retrieved_raw=[],
                    attribution={"total": 0, "score": 1.0, "unsupported": 0})
    if qid == "Q020":  # 负样本正确拒答
        return dict(answer="超出知识库范围，未找到相关内容。", citations=[],
                    refused=True, retrieved_raw=[],
                    attribution={"total": 0, "score": 1.0, "unsupported": 0})
    if qid == "Q011":  # 负样本：声明缺口 + 披露相关证据 -> 有效拒答（极性判定）
        return dict(answer=(
                        "当前知识库中未找到关于长期停放是否导致蓄电池亏电、以及如何维护的依据。"
                        "现有资料仅有：检查 12V 蓄电池电压，每 10,000 km 或每 1 年 [1]。"
                    ),
                    citations=[Citation(index=1, chunk_id="c1", source=MAINT)],
                    refused=False, retrieved_raw=[good_chunk(item)],
                    attribution={"total": 1, "score": 0.95, "unsupported": 0})
    if qid == "Q012":  # 同上：问的是胎压标准值，语料只有检查频次
        return dict(answer=(
                        "当前知识库中未找到胎压标准值（kPa / bar）的依据。"
                        "现有资料仅有：检查胎压与胎面磨损，每月 [1]。"
                    ),
                    citations=[Citation(index=1, chunk_id="c1", source=MAINT)],
                    refused=False, retrieved_raw=[good_chunk(item)],
                    attribution={"total": 1, "score": 0.95, "unsupported": 0})
    if qid == "Q004":  # 该答却拒，且答案没有实质内容 -> 依据未找到
        return dict(answer="未找到依据。", citations=[], refused=True,
                    retrieved_raw=[good_chunk(item)],
                    attribution={"total": 0, "score": 1.0, "unsupported": 0})
    if qid == "Q005":  # 说明缺口但给出可用信息 -> partial，不算拒答
        return dict(answer="当前知识库中未找到专项说明，但空气滤清器滤芯为每 16000 km 更换 [1]。",
                    citations=[Citation(index=1, chunk_id="c1", source=sources[0])],
                    refused=False, retrieved_raw=[good_chunk(item)],
                    attribution={"total": 2, "score": 0.95, "unsupported": 0})
    if qid == "Q010":  # 期望两个来源，只召回其中一个 -> recall=0.5
        return dict(answer="P0420 表示催化器效率低于阈值 [1]。",
                    citations=[Citation(index=1, chunk_id="c1", source=sources[0])],
                    refused=False,
                    retrieved_raw=[good_chunk(item, pos=1), chunk(MAINT, pos=2),
                                   chunk(MAINT, pos=3), chunk(DTC, pos=4)],
                    attribution={"total": 1, "score": 0.9, "unsupported": 0})
    if qid == "Q013":  # 关键片段排在第 4 位 -> 排序靠后
        return dict(answer="夏季需检查轮胎与空调 [1]。",
                    citations=[Citation(index=1, chunk_id="c1", source=sources[0])],
                    refused=False,
                    retrieved_raw=[chunk(WARRANTY, pos=1), chunk(DTC, pos=2),
                                   chunk(DTC, pos=3), good_chunk(item, pos=4),
                                   chunk(MAINT, pos=5)],
                    attribution={"total": 1, "score": 0.9, "unsupported": 0})
    if qid == "Q018":  # 归因不足：5 句里 3 句在片段中找不到依据（占比 60%）
        # 注意不能只用「2 句 1 句无依据」——分句少于 3 句时不做归因判定（统计不稳定）
        return dict(answer="整车质保 3 年或 6 万公里 [1]。延保可覆盖 5 年 [1]。易损件终身免费 [1]。",
                    citations=[Citation(index=1, chunk_id="c1", source=WARRANTY)],
                    refused=False,
                    retrieved_raw=[good_chunk(item, pos=1), chunk(WARRANTY, pos=2)],
                    attribution={"total": 5, "score": 0.4, "unsupported": 3})
    # 其余条目（评测集扩建后的 Q021+）：按条目极性给默认场景，
    # 断言全部从数据集动态推导（见 main），新增条目不需要改本文件
    if item.get("must_refuse"):
        # 负样本：正确拒答。有相关证据的走 Q011/Q012 模式（声明缺口+披露），
        # 纯无依据的走 Q003 模式（模型 soft 拒答）
        snippets = item.get("expected_snippets") or []
        if sources and snippets:
            return dict(
                answer=(
                    "当前知识库中未找到所问内容的直接依据。"
                    f"现有资料仅有：{snippets[0]} [1]。"
                ),
                citations=[Citation(index=1, chunk_id="c1", source=sources[0])],
                refused=False, retrieved_raw=[good_chunk(item)],
                attribution={"total": 1, "score": 0.9, "unsupported": 0},
            )
        return dict(answer="未找到相关依据，建议以用户手册为准。", citations=[],
                    refused=True, retrieved_raw=[chunk(WARRANTY)],
                    attribution={"total": 0, "score": 1.0, "unsupported": 0})
    keywords = item.get("expected_keywords") or []
    answer = f"参考保养表：{'、'.join(keywords)} [1]。" if keywords else "参考保养表，按周期更换 [1]。"
    return dict(answer=answer,
                citations=[Citation(index=1, chunk_id="c1", source=sources[0])],
                refused=False, retrieved_raw=[good_chunk(item)],
                attribution={"total": 1, "score": 0.9, "unsupported": 0})


class FakeQA:
    def __init__(self, dataset_items):
        self._items = {it["id"]: it for it in dataset_items}

    async def answer(self, question, top_k=None, mode=None, debug=False):
        qid = next((k for k, v in self._items.items() if v["question"] == question), None)
        item = self._items.get(qid) or {"id": "", "must_refuse": False}
        outcome = make_outcome(item)
        outcome.update(retrieval_mode="hybrid", latency={}, usage={}, warnings=[])
        return outcome


async def main():
    settings = Settings()
    svc = EvaluationService(settings, qa=None)
    path, dataset = svc.load_dataset()
    svc._qa = FakeQA([it.model_dump() for it in dataset.items])
    summary, report = await svc.run(top_k=5)

    print(f"总条数 {summary.total} / 完成 {summary.completed}")
    print(f"分组: {report['dataset']['groups']}")
    print()
    print(f"source_hit_rate      = {summary.source_hit_rate}")
    print(f"recall@5             = {summary.mean_recall_at_k}")
    print(f"mrr                  = {summary.mean_mrr}")
    print(f"ndcg@5               = {summary.mean_ndcg_at_k}")
    print(f"precision@5          = {summary.mean_precision_at_k}")
    print(f"faithfulness         = {summary.mean_faithfulness}")
    print(f"refusal_recall       = {summary.refusal_recall}")
    print(f"over_refusal_rate    = {summary.over_refusal_rate}")
    print(f"avg_keyword_coverage = {summary.avg_keyword_coverage}")
    print()
    print("失败分型分布:")
    for name, count in summary.failure_distribution.items():
        print(f"  {name}: {count}")
    print()
    print("逐条标签:")
    for row in report["items"]:
        print(f"  {row['id']} -> {row.get('failure_label_cn')}"
              f"  recall={row.get('retrieval_metrics', {}).get('recall_at_k')}"
              f"  rank={row.get('retrieval_metrics', {}).get('first_relevant_rank')}"
              f"  kw={row.get('generation_metrics', {}).get('keyword_coverage')}")

    # ---------------- 断言：数字必须自洽 ----------------
    groups = report["dataset"]["groups"]
    assert groups["refusal"] + groups["answerable"] == summary.total, "拒答/可答分组之和应等于总数"
    n_refusal = sum(1 for it in dataset.items if it.must_refuse)
    expected_retrieval = sum(1 for it in dataset.items if it.expected_sources)
    assert groups["retrieval"] == expected_retrieval, (
        f"有期望来源的条目应为 {expected_retrieval} 条，实际 {groups['retrieval']}"
    )

    dist_total = sum(summary.failure_distribution.values())
    assert dist_total == summary.total, f"分型总数 {dist_total} 应等于 {summary.total}"

    # 唯一漏拒是 Q006、唯一过度拒答是 Q004（mock 里固定构造），其余全对
    assert summary.refusal_recall == round((n_refusal - 1) / n_refusal, 4), (
        f"拒答召回应为 {(n_refusal - 1)}/{n_refusal}，实际 {summary.refusal_recall}"
    )
    answerable = summary.total - n_refusal
    assert summary.over_refusal_rate == round(1 / answerable, 4), (
        f"过度拒答应为 1/{answerable}，实际 {summary.over_refusal_rate}"
    )

    labels = {row["id"]: row.get("failure_label") for row in report["items"]}
    paths = {row["id"]: row.get("refusal_path") for row in report["items"]}
    assert labels["Q002"] == "retrieval_miss", labels["Q002"]
    assert labels["Q006"] == "missed_refusal", labels["Q006"]
    assert labels["Q004"] == "evidence_miss", labels["Q004"]
    assert labels["Q018"] == "attribution_weak", labels["Q018"]
    assert labels["Q013"] == "ranking_low", labels["Q013"]
    assert labels["Q001"] == "ok", labels["Q001"]
    # 说明缺口但给出信息：不应被当成拒答
    assert labels["Q005"] == "ok", labels["Q005"]
    assert paths["Q005"] == "partial", paths["Q005"]

    # 负样本正确拒答必须全部判通过（此处曾是 bug：被词面 F1 判成答案不符）
    for qid in ("Q003", "Q008", "Q014", "Q017", "Q020"):
        assert labels[qid] == "ok", f"{qid} 应为 ok，实际 {labels[qid]}"

    # 「声明缺口 + 披露相关证据」在负样本里是**合格**回答，不能被判漏拒答。
    # 这两个条目曾标成可答样本，导致核心问题没答却拿了满分（标注缺陷）。
    for qid in ("Q011", "Q012"):
        assert labels[qid] == "ok", f"{qid} 应为 ok，实际 {labels[qid]}"
        assert paths[qid] == "partial", f"{qid} 路径应为 partial，实际 {paths[qid]}"
        row = next(r for r in report["items"] if r["id"] == qid)
        assert row["refusal_declared"] is True, f"{qid} 应声明了证据不足"
        assert row["refused"] is True, f"{qid} 应按极性算作有效拒答"

    # 12.27 指标口径：关键词覆盖均值只统计「实际作答」的行。
    # 有效拒答的行（refused=True）不计入——负样本硬拒答本来就不该提到关键词，
    # 按旧口径记 0 分是在惩罚正确行为。
    kw_by_id = {it.id: (it.expected_keywords or []) for it in dataset.items}
    answered_rows = [
        row for row in report["items"]
        if kw_by_id.get(row["id"]) and not row["refused"]
    ]
    expected_avg = (
        round(
            sum(row["generation_metrics"]["keyword_coverage"] for row in answered_rows)
            / len(answered_rows),
            4,
        )
        if answered_rows
        else 0.0
    )
    assert summary.avg_keyword_coverage == expected_avg, (
        f"关键词覆盖均值应只统计实际作答行（{expected_avg}），实际 {summary.avg_keyword_coverage}"
    )

    # Q010 期望 2 个来源只召回 1 个 -> 文件级 recall 0.5
    q010 = next(r for r in report["items"] if r["id"] == "Q010")
    assert q010["retrieval_metrics"]["recall_at_k"] == 0.5, q010["retrieval_metrics"]

    print("\n>>> 全部断言通过，数字自洽")


if __name__ == "__main__":
    asyncio.run(main())
