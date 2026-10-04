"""离线评测服务：跑评测集 -> 规则打分 -> 产出可复核的报告。

必须诚实说明的三件事：
1. 这里用的是**规则打分**（来源命中、关键词覆盖率、引用越界、拒答正确性），
   不是 LLM-as-judge，更不是人工评测；
2. 报告保存的是"某次运行的真实结果"，字段里不含任何基准对比或编造的指标；
3. ground_truth 字段只做展示与人工复核，不参与自动打分（避免把规则分数说成准确率）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from app.config import PROJECT_ROOT, Settings
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
    """期望来源是否出现在召回结果中（支持只写文件名，例如 '保养手册.md'）。"""
    if not expected_sources:
        return False
    hit_sources = {item.source for item in retrieved}
    for expected in expected_sources:
        expected_norm = expected.replace("\\", "/").strip()
        if expected_norm in hit_sources:
            return True
        # 允许只写文件名
        if any(source.endswith("/" + expected_norm) or source == expected_norm for source in hit_sources):
            return True
    return False


def keyword_coverage(expected_keywords: Sequence[str], answer: str) -> float:
    """期望关键词在回答中的覆盖率（大小写不敏感的子串匹配）。"""
    if not expected_keywords:
        return 0.0
    lowered = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.strip().lower() in lowered)
    return round(hits / len(expected_keywords), 4)


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
                    )
                )
                row["error"] = f"{type(exc).__name__}: {exc}"
                report_items.append(row)
                continue

            answer = str(outcome.get("answer") or "")
            citations: List[Citation] = list(outcome.get("citations") or [])
            refused = bool(outcome.get("refused"))
            used_indices = citation_indices(answer)
            valid_indices = {citation.index for citation in citations}
            out_of_range = sorted(index for index in used_indices if index not in valid_indices)
            coverage = keyword_coverage(item.expected_keywords, answer)
            hit = source_hit(item.expected_sources, results_chunks)

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
        judged = [item for item in dataset.items if item.expected_sources or item.expected_keywords]
        judged_ids = {item.id for item in judged}
        judged_results = [result for result in completed if result.id in judged_ids]

        source_hit_rate = (
            round(sum(1 for result in judged_results if result.hit_source) / len(judged_results), 4)
            if judged_results
            else 0.0
        )
        avg_coverage = (
            round(sum(result.keyword_coverage for result in judged_results) / len(judged_results), 4)
            if judged_results
            else 0.0
        )
        avg_latency = (
            round(sum(result.total_ms for result in completed) / len(completed), 2)
            if completed
            else 0.0
        )

        summary = EvalRunResponse(
            dataset_path=str(path),
            total=len(dataset.items),
            completed=len(completed),
            source_hit_rate=source_hit_rate,
            avg_keyword_coverage=avg_coverage,
            avg_latency_ms=avg_latency,
            results=results,
        )

        report = {
            "dataset": {
                "path": str(path),
                "name": dataset.name,
                "version": dataset.version,
                "total": len(dataset.items),
                "scored_items": len(judged_results),
                "note": (
                    "只对同时提供了 expected_sources 或 expected_keywords 的条目计分；"
                    "ground_truth 不参与自动打分。"
                ),
            },
            "run": {
                "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_ms": int((time.perf_counter() - started) * 1000),
                "top_k": top_k or self._settings.top_k,
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
                "disclaimer": (
                    "以上为规则打分结果（来源命中 / 关键词覆盖 / 耗时），是本次运行的真实值。"
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
