"""离线评测服务：跑评测集 -> 规则打分 -> 产出可复核的报告。

必须诚实说明的三件事：
1. 这里用的是**规则打分**（来源命中、排序质量、关键词覆盖、引用越界、拒答正确性、
   逐句归因），不是 LLM-as-judge，更不是人工评测；
2. 报告保存的是"某次运行的真实结果"，字段里不含任何基准对比或编造的指标；
3. ground_truth 只参与 `answer_token_f1` 的词面重合计算，
   它是**下限检测**，不是准确率——同义改写会被判低分，报告里已标注。

指标口径全部集中在 `app/core/metrics.py`，本文件只负责取数和汇总，
不重复实现任何一条判定规则（此前 `scripts/run_eval.py` 曾另写一份，已统一）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.config import PROJECT_ROOT, Settings
from app.core import metrics as M
from app.logging_conf import get_logger
from app.models import (
    Citation,
    EvalDataset,
    EvalItem,
    EvalItemResult,
    EvalRunResponse,
    RetrievedChunk,
)
from app.services.rag_pipeline import CITATION_PATTERN, QAEngine

logger = get_logger(__name__)

DEFAULT_DATASET_PATH = PROJECT_ROOT / "eval" / "eval_set.json"
REPORT_DIR = PROJECT_ROOT / "eval" / "results"


def _mean(values: Sequence[float]) -> Optional[float]:
    """算术平均；样本为空时返回 None（而不是 0.0，避免被读成「全错」）。"""
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 4)


def find_placeholder_fields(item: EvalItem) -> List[str]:
    """找出仍然是 [待补充] 的字段，评测报告里会列出来提醒补全。"""
    fields: List[str] = []
    if item.ground_truth.strip().startswith("[待补充"):
        fields.append("ground_truth")
    if not item.expected_sources:
        fields.append("expected_sources")
    if not item.expected_keywords:
        fields.append("expected_keywords")
    return fields


def source_hit(expected_sources: Sequence[str], retrieved: Sequence[RetrievedChunk]) -> bool:
    """期望来源是否出现在召回结果中。

    保留此函数是为了兼容既有调用方；判定逻辑已下沉到 metrics，
    避免服务端与脚本端出现两套口径。
    """
    return M.source_hit_at_k(expected_sources, retrieved, len(retrieved) or 1)


def keyword_coverage(expected_keywords: Sequence[str], answer: str) -> float:
    """期望关键词在回答中的覆盖率（口径见 metrics.keyword_coverage）。"""
    return M.keyword_coverage(expected_keywords, answer)


def citation_indices(answer: str) -> Set[int]:
    return {int(match.group(1)) for match in CITATION_PATTERN.finditer(answer)}


def dump_model(value: Any) -> Any:
    """把 Pydantic 模型转成 dict，保证写进 JSON 报告时可序列化。"""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


class EvaluationService:
    def __init__(self, settings: Settings, qa: QAEngine) -> None:
        self._settings = settings
        self._qa = qa

    # ------------------------------------------------------------------ #
    def load_dataset(self, dataset_path: str | Path | None = None) -> Tuple[Path, EvalDataset]:
        path = Path(dataset_path) if dataset_path else DEFAULT_DATASET_PATH
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"评测集不存在：{path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        dataset = EvalDataset(**raw)
        return path, dataset

    def dataset_summary(self, dataset_path: str | Path | None = None) -> Dict[str, Any]:
        path, dataset = self.load_dataset(dataset_path)
        placeholder_items = [
            item.id for item in dataset.items if find_placeholder_fields(item)
        ]
        categories: Dict[str, int] = {}
        for item in dataset.items:
            categories[item.category or "(未分类)"] = categories.get(item.category or "(未分类)", 0) + 1
        return {
            "path": str(path),
            "name": dataset.name,
            "version": dataset.version,
            "total": len(dataset.items),
            "categories": categories,
            "incomplete_items": placeholder_items,
            "hint": (
                "这些条目的 ground_truth / expected_sources / expected_keywords 仍是 [待补充]，"
                "只会影响人工复核，不影响规则打分中的其他项。"
            )
            if placeholder_items
            else "评测集字段已填全。",
        }

    # ------------------------------------------------------------------ #
    async def run(
        self,
        dataset_path: str | Path | None = None,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
    ) -> Tuple[EvalRunResponse, Dict[str, Any]]:
        path, dataset = self.load_dataset(dataset_path)
        started = time.perf_counter()
        effective_k = int(top_k or self._settings.top_k)

        results: List[EvalItemResult] = []
        report_items: List[Dict[str, Any]] = []

        for item in dataset.items:
            item_started = time.perf_counter()
            row: Dict[str, Any] = {
                "id": item.id,
                "category": item.category,
                "question": item.question,
                "must_refuse": item.must_refuse,
                "placeholder_fields": find_placeholder_fields(item),
            }
            try:
                # debug=True 让 QAEngine 在一次调用里同时返回回答与召回明细，
                # 避免为了拿召回结果而重复检索一遍（省一次 embedding 调用）。
                outcome = await self._qa.answer(
                    item.question, top_k=top_k, mode=retrieval_mode, debug=True
                )
                results_chunks: List[RetrievedChunk] = list(outcome.get("retrieved_raw") or [])
                mode = str(outcome.get("retrieval_mode") or "")
            except Exception as exc:
                logger.exception("评测条目执行失败 id=%s", item.id)
                results.append(
                    EvalItemResult(
                        id=item.id,
                        question=item.question,
                        error=f"{type(exc).__name__}: {exc}",
                        total_ms=int((time.perf_counter() - item_started) * 1000),
                        failure_label=M.ERROR,
                    )
                )
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["failure_label"] = M.ERROR
                row["failure_label_cn"] = M.FAILURE_LABELS[M.ERROR]
                report_items.append(row)
                continue

            answer = str(outcome.get("answer") or "")
            citations: List[Citation] = list(outcome.get("citations") or [])
            refused_raw = bool(outcome.get("refused"))
            refusal_reason = str(outcome.get("refusal_reason") or "") or None
            # 有效拒答 = 硬拒答 or 纯拒答话术；「部分覆盖」不算拒答（见 metrics 文档）
            substantive = M.has_substantive_answer(item.expected_keywords, answer)
            # 拒答要按极性判：负样本「声明缺口」即为履行义务（含部分披露），
            # 正样本只有「纯拒答」才算没答。见 metrics.effective_refusal 的说明。
            refusal_declared = refused_raw or M.is_refusal_text(answer)
            refused = M.effective_refusal(
                must_refuse=item.must_refuse,
                refused_raw=refused_raw,
                answer=answer,
                substantive=substantive,
            )
            ref_path = M.refusal_path(refused_raw, answer, substantive)
            used_indices = citation_indices(answer)
            valid_indices = {citation.index for citation in citations}
            out_of_range = sorted(index for index in used_indices if index not in valid_indices)
            coverage = keyword_coverage(item.expected_keywords, answer)
            hit = M.source_hit_at_k(
                item.expected_sources, results_chunks, effective_k, item.expected_snippets
            )

            # ---- L1：检索侧排序质量（片段级优先，未标注则退回文件级） ----
            flags = M.relevance_flags(
                item.expected_sources, results_chunks, item.expected_snippets
            )
            recall_k = M.source_recall_at_k(
                item.expected_sources, results_chunks, effective_k, item.expected_snippets
            )
            level = M.relevance_level(item.expected_snippets)
            first_rank = M.first_relevant_rank(flags)
            mrr = M.reciprocal_rank(flags)
            ndcg = M.ndcg_at_k(flags, effective_k)
            precision = M.precision_at_k(flags, effective_k)

            # ---- L1：生成侧 ----
            f1 = M.token_f1(item.ground_truth, answer)
            attribution = outcome.get("attribution") or {}
            attr_score: Optional[float] = None
            attr_unsupported = 0
            attr_total = 0
            if isinstance(attribution, dict) and attribution.get("total"):
                raw_score = attribution.get("score")
                if raw_score is not None:
                    attr_score = round(float(raw_score), 4)
                attr_unsupported = int(attribution.get("unsupported") or 0)
                attr_total = int(attribution.get("total") or 0)

            failure = M.classify_failure(
                must_refuse=item.must_refuse,
                refused=refused,
                substantive=substantive,
                expected_sources=item.expected_sources,
                recall_at_k=recall_k,
                first_rank=first_rank,
                out_of_range=out_of_range,
                attribution_total=attr_total,
                attribution_unsupported=attr_unsupported,
                keyword_cov=coverage,
                f1=f1,
            )

            total_ms = int((time.perf_counter() - item_started) * 1000)
            results.append(
                EvalItemResult(
                    id=item.id,
                    question=item.question,
                    answer=answer,
                    refused=refused,
                    hit_source=hit,
                    keyword_coverage=coverage,
                    citation_count=len(citations),
                    top_score=round(max((c.score for c in results_chunks), default=0.0), 6),
                    total_ms=total_ms,
                    recall_at_k=recall_k,
                    mrr=mrr,
                    ndcg_at_k=ndcg,
                    precision_at_k=precision,
                    first_relevant_rank=first_rank,
                    answer_token_f1=f1,
                    attribution_score=attr_score,
                    failure_label=failure,
                    refused_raw=refused_raw,
                    refusal_declared=refusal_declared,
                    refusal_path=ref_path,
                    substantive=substantive,
                    relevance_level=level,
                    refusal_reason=refusal_reason,
                )
            )

            row.update(
                {
                    "retrieval_mode": mode,
                    "retrieved_sources": [
                        f"{c.source}#{c.position}" for c in results_chunks
                    ],
                    "expected_sources": item.expected_sources,
                    "source_hit": hit,
                    "expected_keywords": item.expected_keywords,
                    "keyword_coverage": coverage,
                    "citation_indices_in_answer": sorted(used_indices),
                    "citation_indices_out_of_range": out_of_range,
                    "must_refuse_satisfied": (refused == item.must_refuse),
                    "retrieval_metrics": {
                        "recall_at_k": recall_k,
                        "mrr": mrr,
                        "ndcg_at_k": ndcg,
                        "precision_at_k": precision,
                        "first_relevant_rank": first_rank,
                        "relevance_flags": flags,
                        "relevance_level": level,
                        "k": effective_k,
                    },
                    "generation_metrics": {
                        "keyword_coverage": coverage,
                        "answer_token_f1": f1,
                        "attribution_score": attr_score,
                        "attribution_unsupported": attr_unsupported,
                    },
                    "failure_label": failure,
                    "failure_label_cn": M.FAILURE_LABELS.get(failure, failure),
                    "refused_raw": refused_raw,
                    "refused": refused,
                    "refusal_path": ref_path,
                    "refusal_declared": refusal_declared,
                    "refusal_reason": refusal_reason,
                    "substantive": substantive,
                    "answer": answer,
                    "ground_truth": item.ground_truth,
                    "latency": dump_model(outcome.get("latency")),
                    "usage": dump_model(outcome.get("usage")),
                    "warnings": outcome.get("warnings") or [],
                    "total_ms": total_ms,
                }
            )
            report_items.append(row)

        completed = [result for result in results if not result.error]
        item_by_id = {item.id: item for item in dataset.items}
        judged = [item for item in dataset.items if item.expected_sources or item.expected_keywords]
        judged_ids = {item.id for item in judged}
        judged_results = [result for result in completed if result.id in judged_ids]

        # ---- 三类指标的分母完全不同，必须分开统计 ----
        # 检索：有期望来源
        retrieval_results = [r for r in completed if item_by_id[r.id].expected_sources]
        # 拒答：负样本（期望拒答）
        refusal_results = [r for r in completed if item_by_id[r.id].must_refuse]
        # 过度拒答：可答样本（不允许拒答）
        answerable_results = [r for r in completed if not item_by_id[r.id].must_refuse]

        source_hit_rate = (
            round(sum(1 for result in judged_results if result.hit_source) / len(judged_results), 4)
            if judged_results
            else 0.0
        )
        # 关键词覆盖只统计**实际作答**的条目（12.27 口径修正）：
        # 有效拒答的行不计入——负样本硬拒答（server_guard）本来就不该提到关键词，
        # 按原口径记 0 分是在惩罚正确行为（实测把 0.9783 拉低）；过度拒答
        # 已由 over_refusal_rate 单独考核，这里不再重复惩罚。
        answered_results = [result for result in judged_results if not result.refused]
        avg_coverage = (
            round(sum(result.keyword_coverage for result in answered_results) / len(answered_results), 4)
            if answered_results
            else 0.0
        )
        avg_latency = (
            round(sum(result.total_ms for result in completed) / len(completed), 2)
            if completed
            else 0.0
        )

        mean_recall = _mean([r.recall_at_k for r in retrieval_results])
        mean_mrr = _mean([r.mrr for r in retrieval_results])
        mean_ndcg = _mean([r.ndcg_at_k for r in retrieval_results])
        mean_precision = _mean([r.precision_at_k for r in retrieval_results])
        # 归因只在「有可校验断言」的回答上才有分数：纯拒答 / 声明缺口的句子会被
        # attribution 当作过渡语跳过，total=0 → 没有分数。
        # 所以这个均值必须连同分母一起报，否则读者会以为它覆盖了全部样本。
        # 实测 20 条里有 10 条是 None，恰好是那些诚实拒答 / 部分披露的回答。
        attribution_scored = [r for r in completed if r.attribution_score is not None]
        mean_faithfulness = _mean([r.attribution_score for r in attribution_scored])
        refusal_recall = M.ratio(
            sum(1 for r in refusal_results if r.refused), len(refusal_results)
        )
        over_refusal_rate = M.ratio(
            sum(1 for r in answerable_results if r.refused), len(answerable_results)
        )
        failure_distribution = M.distribution(
            [r.failure_label for r in results if r.failure_label]
        )
        # 拒答来源分布：区分「服务端拦截」与「模型自觉」，
        # 换模型后若 model 占比骤降，说明拒答能力退化而总分可能看不出来。
        refusal_sources: Dict[str, int] = {}
        for result in refusal_results:
            if not result.refused:
                continue
            if result.refused_raw:
                key = "server_guard" if result.refusal_reason else "server_empty"
            else:
                key = "model_soft"
            refusal_sources[key] = refusal_sources.get(key, 0) + 1

        summary = EvalRunResponse(
            dataset_path=str(path),
            total=len(dataset.items),
            completed=len(completed),
            source_hit_rate=source_hit_rate,
            avg_keyword_coverage=avg_coverage,
            avg_latency_ms=avg_latency,
            results=results,
            mean_recall_at_k=mean_recall,
            mean_mrr=mean_mrr,
            mean_ndcg_at_k=mean_ndcg,
            mean_precision_at_k=mean_precision,
            mean_faithfulness=mean_faithfulness,
            refusal_recall=refusal_recall,
            over_refusal_rate=over_refusal_rate,
            failure_distribution=failure_distribution,
            refusal_sources=refusal_sources,
        )

        report = {
            "dataset": {
                "path": str(path),
                "name": dataset.name,
                "version": dataset.version,
                "total": len(dataset.items),
                "scored_items": len(judged_results),
                "groups": {
                    "retrieval": len(retrieval_results),
                    "refusal": len(refusal_results),
                    "answerable": len(answerable_results),
                    # 负样本若同时带 expected_sources（如 Q011/Q012：核心问题无依据，
                    # 但有该被检索出来披露的相关证据），会同时进检索组和拒答组。
                    # 所以 retrieval + refusal 可能大于总数，报告必须显式给出重叠数，
                    # 否则读者一相加就会以为数据有问题。
                    "retrieval_and_refusal_overlap": len(
                        {r.id for r in retrieval_results} & {r.id for r in refusal_results}
                    ),
                },
                "relevance_levels": {
                    "snippet": sum(1 for item in dataset.items if item.expected_snippets),
                    "file": sum(
                        1
                        for item in dataset.items
                        if item.expected_sources and not item.expected_snippets
                    ),
                },
                "note": (
                    "检索指标只在有 expected_sources 的条目上统计；"
                    "标注了 expected_snippets 的条目按片段级判定（来源命中且含关键短语），"
                    "未标注的按文件级判定（只看来源文件）。两档粒度的数字不可混比。"
                    "拒答指标只在 must_refuse=true 的负样本上统计；"
                    "过度拒答只在可答样本上统计。三组分母不同，数字不可互相比较；"
                    "其中「带相关证据的负样本」会同时进检索组与拒答组，"
                    "因此 retrieval + refusal 可能大于总数，重叠数见 "
                    "retrieval_and_refusal_overlap。"
                    "ground_truth 只参与 answer_token_f1 的词面重合计算，不是准确率。"
                ),
            },
            "run": {
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "top_k": effective_k,
                "retrieval_mode": retrieval_mode or self._settings.retrieval_mode,
                "embedding_provider": self._settings.embedding_provider,
                "llm_model": self._settings.llm_model,
            },
            "metrics": {
                "source_hit_rate": source_hit_rate,
                "avg_keyword_coverage": avg_coverage,
                "avg_latency_ms": avg_latency,
                "completed": len(completed),
                "failed": len(results) - len(completed),
                "recall_at_k": mean_recall,
                "mrr": mean_mrr,
                "ndcg_at_k": mean_ndcg,
                "precision_at_k": mean_precision,
                "faithfulness": mean_faithfulness,
                "faithfulness_scored_items": len(attribution_scored),
                "faithfulness_total_items": len(completed),
                # 分句级构成：unsupported_ratio 才是分型用的信号，
                # faithfulness 混入了 no_citation 扣分，两者不可互换。
                "attribution_clause_stats": M.attribution_clause_stats(
                    row.get("attribution") for row in report_items
                ),
                "refusal_recall": refusal_recall,
                "over_refusal_rate": over_refusal_rate,
                "failure_distribution": failure_distribution,
                "refusal_sources": refusal_sources,
                "disclaimer": (
                    "以上为规则打分结果（来源命中 / 排序质量 / 关键词覆盖 / 逐句归因 / 拒答 / 耗时），"
                    "是本次运行的真实值。"
                    "它不是准确率，也没有与任何其他系统做对比。"
                ),
            },
            "items": report_items,
        }
        return summary, report

    @staticmethod
    def save_report(report: Dict[str, Any], filename: str | None = None) -> Path:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        name = filename or f"report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path = REPORT_DIR / name
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("评测报告已保存：%s", path)
        return path

    @staticmethod
    def list_reports() -> List[Dict[str, Any]]:
        if not REPORT_DIR.exists():
            return []
        reports: List[Dict[str, Any]] = []
        for path in sorted(REPORT_DIR.glob("report_*.json"), reverse=True):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            reports.append(
                {
                    "file": path.name,
                    "finished_at": (payload.get("run") or {}).get("finished_at"),
                    "metrics": payload.get("metrics"),
                }
            )
        return reports
