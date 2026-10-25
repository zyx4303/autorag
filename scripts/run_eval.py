#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""本地评测脚本（只用标准库，不依赖任何第三方包）。

用法：
    # 1) 先确保服务已启动： uvicorn app.main:app --port 8000
    # 2) 跑评测（默认读 eval/eval_set.json，调 /api/v1/chat）
    python scripts/run_eval.py

    # 指定文件、只跑前 5 条、换个后端地址：
    python scripts/run_eval.py --dataset eval/eval_set.json --limit 5 --base-url http://127.0.0.1:8000

    # 只测召回（不调用大模型、不消耗 token）：
    python scripts/run_eval.py --mode retrieve

打分说明（必须是规则打分，不是准确率）：
    source_hit    : 期望来源文件是否出现在召回结果里
    kw_coverage   : 期望关键词在回答中的覆盖率
    refused       : 有效拒答（负样本「声明缺口」即算，含部分披露）
    citation_ok   : 回答中的 [n] 是否都落在实际 citations 范围内
输出：
    eval/results/local_report_<时间戳>.json   —— 机器可读的完整结果
    eval/results/local_report_<时间戳>.md     —— 便于贴到简历/面试材料里人工复核
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "eval" / "eval_set.json"
RESULT_DIR = PROJECT_ROOT / "eval" / "results"

# 复用服务层的指标口径，避免脚本和 API 报告出现两套判定规则。
# app.core.metrics 只依赖标准库，不引入第三方包，本脚本「零依赖」的约定不变。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from app.core import metrics as M  # noqa: E402


def http_json(method: str, url: str, payload: Optional[dict] = None, timeout: float = 180.0) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} {url}：{detail[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"无法连接 {url}：{exc.reason}。请先启动服务：uvicorn app.main:app --port 8000"
        ) from exc
    return json.loads(body)


def load_dataset(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"评测集不存在：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"评测集 JSON 解析失败（第 {exc.lineno} 行第 {exc.colno} 列）：{exc.msg}"
        ) from exc
    if not isinstance(payload.get("items"), list):
        raise SystemExit("评测集缺少 items 数组")
    return payload


def is_placeholder(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().startswith("[待补充")
    if isinstance(value, list):
        return not value or all(is_placeholder(item) for item in value)
    return False


def keyword_coverage(expected: Sequence[str], answer: str) -> float:
    """期望关键词覆盖率（口径见 metrics.keyword_coverage）。"""
    if not expected or is_placeholder(list(expected)):
        return 0.0
    return M.keyword_coverage(expected, answer)


def citation_out_of_range(answer: str, citations: Sequence[dict]) -> List[int]:
    """回答里出现、但 citations 中不存在的编号（如只有 5 条却写了 [7]）。"""
    import re

    indices = {int(match.group(1)) for match in re.finditer(r"\[(\d{1,2})\]", answer)}
    if not indices:
        return []
    valid = {int(item.get("index", -1)) for item in citations}
    return sorted(index for index in indices if index not in valid)


def run(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    """跑一轮评测，返回报告字典（健康检查失败时返回 None）。"""
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (PROJECT_ROOT / dataset_path).resolve()
    dataset = load_dataset(dataset_path)
    items: List[dict] = dataset["items"]
    if args.limit:
        items = items[: args.limit]

    base = args.base_url.rstrip("/")
    chat_url = f"{base}/api/v1/chat"
    retrieve_url = f"{base}/api/v1/retrieve"

    # 先做一次健康检查，避免把"服务没起"误判成"模型答得差"
    try:
        health = http_json("GET", f"{base}/api/v1/health", timeout=30.0)
    except RuntimeError as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return None
    print("服务状态：")
    print(f"  status           = {health.get('status')}")
    print(f"  embedding_ready  = {health.get('embedding_ready')}  provider={health.get('detail', {}).get('config', {}).get('embedding_provider')}")
    print(f"  vector_store     = {health.get('vector_store_ready')}  count={health.get('collection_count')}")
    print(f"  llm_ready        = {health.get('llm_ready')}")
    for note in health.get("detail", {}).get("notes", []) or []:
        print(f"  [提示] {note}")
    print()
    if args.mode == "chat" and not health.get("llm_ready"):
        print("[警告] LLM 未就绪，问答接口会返回配置错误提示。", file=sys.stderr)

    # 会话历史是服务端常驻状态：不清掉的话，--repeat 的第二轮会带着第一轮的历史，
    # 多轮评测集测出来的就不是同一条件（不可重复 = 不可比较）。
    session_ids = sorted({item.get("session_id") for item in items if item.get("session_id")})
    if session_ids:
        for session_id in session_ids:
            try:
                http_json("POST", f"{base}/api/v1/session/clear", {"session_id": session_id})
            except RuntimeError as exc:
                print(f"[警告] 清理会话 {session_id} 失败：{exc}", file=sys.stderr)
        print(f"已清除 {len(session_ids)} 个会话的历史（保证每轮从同一起点开始）")

    rows: List[Dict[str, Any]] = []
    started_all = time.perf_counter()
    # 片段级判定需要完整正文；接口没返回 text 时 recall 会静默归零，必须单独记下来
    missing_text_items: List[str] = []

    for index, item in enumerate(items, start=1):
        item_id = item.get("id", f"item{index}")
        question = item.get("question", "")
        ground_truth = str(item.get("ground_truth", "") or "")
        expected_sources = item.get("expected_sources") or []
        expected_snippets = item.get("expected_snippets") or []
        expected_keywords = item.get("expected_keywords") or []
        must_refuse = bool(item.get("must_refuse"))

        row: Dict[str, Any] = {
            "id": item_id,
            "category": item.get("category", ""),
            "question": question,
            "must_refuse": must_refuse,
            "session_id": item.get("session_id") or "",
            "turn": item.get("turn") or 0,
            "ground_truth": item.get("ground_truth", ""),
            "expected_sources": expected_sources,
            "expected_snippets": expected_snippets,
            "expected_keywords": expected_keywords,
            "placeholder_fields": [
                field
                for field in ("ground_truth", "expected_sources", "expected_keywords")
                if is_placeholder(item.get(field))
            ],
        }

        # 先初始化：http_json 在赋值前抛错时，后面引用 retrieved 会 NameError
        retrieved: List[dict] = []
        answer = ""
        citations: List[dict] = []
        refused: Optional[bool] = None
        refusal_reason: Optional[str] = None
        attribution: Optional[dict] = None
        out_of_range: List[int] = []

        started = time.perf_counter()
        try:
            if args.mode == "retrieve":
                payload = {"query": question, "top_k": args.top_k}
                data = http_json("POST", retrieve_url, payload)
                retrieved = data.get("results") or []
                row["answer"] = ""
                row["citations"] = []
                row["refused"] = None
                row["scored"] = False
            else:
                payload = {"question": question, "top_k": args.top_k, "debug": True}
                # 多轮评测集：条目带 session_id 且按顺序排列时透传给服务端，
                # 服务端据此维护会话历史并做查询改写。评测集里必须保证同会话按轮次先后出现。
                if item.get("session_id"):
                    payload["session_id"] = item["session_id"]
                data = http_json("POST", chat_url, payload)
                retrieved = data.get("retrieved") or []
                answer = str(data.get("answer") or "")
                citations = data.get("citations") or []
                attribution = data.get("attribution")
                out_of_range = citation_out_of_range(answer, citations)
                row["answer"] = answer
                row["citations"] = citations
                refused_raw = bool(data.get("refused"))
                substantive = M.has_substantive_answer(expected_keywords, answer)
                refusal_reason = data.get("refusal_reason") or None
                # 拒答按极性判定（口径同服务层，见 metrics.effective_refusal）：
                # 负样本声明缺口即为履行义务；正样本只有纯拒答才算没答。
                refusal_declared = refused_raw or M.is_refusal_text(answer)
                refused = M.effective_refusal(
                    must_refuse=must_refuse,
                    refused_raw=refused_raw,
                    answer=answer,
                    substantive=substantive,
                )
                row["refused"] = refused
                row["refusal_declared"] = refusal_declared
                row["refusal_reason"] = refusal_reason
                row["refused_raw"] = refused_raw
                row["substantive"] = substantive
                row["refusal_path"] = M.refusal_path(refused_raw, answer, substantive)
                row["latency"] = data.get("latency")
                row["usage"] = data.get("usage")
                row["warnings"] = data.get("warnings") or []
                row["citation_ok"] = not out_of_range
                row["has_citation_marker"] = bool(citations)
                row["rewritten_query"] = data.get("rewritten_query") or ""
                row["used_history"] = bool(data.get("used_history"))
                row["kw_coverage"] = keyword_coverage(expected_keywords, answer)
                row["scored"] = not is_placeholder(expected_keywords)
        except RuntimeError as exc:
            row["error"] = str(exc)

        row["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        row["retrieved_sources"] = [str(entry.get("source", "")) for entry in retrieved]
        row["source_hit"] = M.source_hit_at_k(
            expected_sources, retrieved, args.top_k, expected_snippets
        )

        # ---- 排序指标（片段级优先，未标注则退回文件级） ----
        flags = M.relevance_flags(expected_sources, retrieved, expected_snippets)
        recall_k = M.source_recall_at_k(
            expected_sources, retrieved, args.top_k, expected_snippets
        )
        level = M.relevance_level(expected_snippets)
        first_rank = M.first_relevant_rank(flags)
        row["retrieval_metrics"] = {
            "recall_at_k": recall_k,
            "mrr": M.reciprocal_rank(flags),
            "ndcg_at_k": M.ndcg_at_k(flags, args.top_k),
            "precision_at_k": M.precision_at_k(flags, args.top_k),
            "first_relevant_rank": first_rank,
            "relevance_flags": flags,
            "relevance_level": level,
            "k": args.top_k,
        }

        # 防护：片段级判定依赖接口返回完整正文。若一条都没返回，
        # recall 会全部变成 0 而不报错 —— 必须显式记下来，否则会得出
        # 「检索能力为零」这种完全错误的结论。
        if level == "snippet" and retrieved and not any(
            str(entry.get("text") or "").strip() for entry in retrieved
        ):
            missing_text_items.append(item_id)

        # ---- 生成侧指标 ----
        f1 = M.token_f1(str(ground_truth or ""), answer) if answer else None
        attr_score = None
        attr_unsupported = 0
        attr_total = 0
        if isinstance(attribution, dict) and attribution.get("total"):
            if attribution.get("score") is not None:
                attr_score = round(float(attribution["score"]), 4)
            attr_unsupported = int(attribution.get("unsupported") or 0)
            attr_total = int(attribution.get("total") or 0)
        row["generation_metrics"] = {
            "keyword_coverage": row.get("kw_coverage", 0.0) if answer else 0.0,
            "answer_token_f1": f1,
            "attribution_score": attr_score,
            "attribution_unsupported": attr_unsupported,
        }
        # 逐句归因明细（只含 weak / unsupported / no_citation 的句子）。
        # 没有它，「归因不足」这个标签就没法复核 —— 不知道是哪一句被判无依据。
        row["attribution"] = attribution if isinstance(attribution, dict) else None

        # retrieve 模式只测召回，没有答案与拒答结果，不做端到端分型
        if args.mode == "retrieve":
            row["failure_label"] = ""
            row["failure_label_cn"] = ""
        else:
            label = M.classify_failure(
                error=row.get("error"),
                must_refuse=must_refuse,
                refused=refused,
                substantive=row.get("substantive", False),
                expected_sources=expected_sources,
                recall_at_k=recall_k,
                first_rank=first_rank,
                out_of_range=out_of_range,
                attribution_total=attr_total,
                attribution_unsupported=attr_unsupported,
                keyword_cov=row.get("kw_coverage", 0.0),
                f1=f1,
            )
            row["failure_label"] = label
            row["failure_label_cn"] = M.FAILURE_LABELS.get(label, label)

        rows.append(row)

        flag = "OK " if not row.get("error") else "ERR"
        coverage = row.get("kw_coverage")
        print(
            f"[{index:>2}/{len(items)}] {flag} {item_id} "
            f"命中来源={row['source_hit']} "
            f"关键词覆盖={coverage if coverage is not None else '-'} "
            f"耗时={row['elapsed_ms']}ms"
        )

    duration_ms = int((time.perf_counter() - started_all) * 1000)

    # 口径同 eval_service（12.27）：只统计实际作答的行，
    # 有效拒答的行不计入关键词覆盖（惩罚正确拒答是口径 bug）
    scored_rows = [
        row for row in rows if row.get("scored") and not row.get("refused")
    ]
    refused_rows = [row for row in rows if row.get("must_refuse")]
    # 旧版 bug：source_hit_rate 用「全部条目」做分母，
    # 而负样本本来就没有期望来源，恒为「未命中」，会把命中率系统性压低。
    # 现在按指标各自的适用样本分组统计（与服务层 eval_service 口径一致）。
    retrieval_rows = [row for row in rows if row.get("expected_sources") and not row.get("error")]
    answerable_rows = [row for row in rows if not row.get("must_refuse") and not row.get("error")]
    snippet_rows = [
        row
        for row in retrieval_rows
        if (row.get("retrieval_metrics") or {}).get("relevance_level") == "snippet"
    ]

    def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
        clean = [float(v) for v in values if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else None

    failure_labels = [row["failure_label"] for row in rows if row.get("failure_label")]

    refusal_sources: Dict[str, int] = {}
    for row in refused_rows:
        if not row.get("refused"):
            continue
        if row.get("refused_raw"):
            key = "server_guard" if row.get("refusal_reason") else "server_empty"
        else:
            key = "model_soft"
        refusal_sources[key] = refusal_sources.get(key, 0) + 1

    metrics = {
        "total_items": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "groups": {
            "retrieval": len(retrieval_rows),
            "refusal": len(refused_rows),
            "answerable": len(answerable_rows),
            # 带相关证据的负样本会同时进检索组与拒答组，
            # 所以 retrieval + refusal 可能大于总数（口径同服务层）
            "retrieval_and_refusal_overlap": len(
                {r["id"] for r in retrieval_rows} & {r["id"] for r in refused_rows}
            ),
        },
        "relevance_levels": {
            "snippet": len(snippet_rows),
            "file": len(retrieval_rows) - len(snippet_rows),
        },
        "snippet_text_missing_items": missing_text_items,
        "source_hit_rate": (
            round(sum(1 for row in retrieval_rows if row.get("source_hit")) / len(retrieval_rows), 4)
            if retrieval_rows
            else None
        ),
        "recall_at_k_snippet": _mean(
            [(row.get("retrieval_metrics") or {}).get("recall_at_k") for row in snippet_rows]
        ),
        "recall_at_k": _mean([(row.get("retrieval_metrics") or {}).get("recall_at_k") for row in retrieval_rows]),
        "mrr": _mean([(row.get("retrieval_metrics") or {}).get("mrr") for row in retrieval_rows]),
        "ndcg_at_k": _mean([(row.get("retrieval_metrics") or {}).get("ndcg_at_k") for row in retrieval_rows]),
        "precision_at_k": _mean([(row.get("retrieval_metrics") or {}).get("precision_at_k") for row in retrieval_rows]),
        "avg_keyword_coverage_on_scored": (
            round(sum(row["kw_coverage"] for row in scored_rows) / len(scored_rows), 4)
            if scored_rows
            else None
        ),
        # 归因只在「有可校验断言」的回答上有分数：纯拒答 / 声明缺口的句子会被
        # attribution 当作过渡语跳过，total=0 → 没有分数。必须连同分母一起报。
        "faithfulness": _mean(
            [(row.get("generation_metrics") or {}).get("attribution_score") for row in rows]
        ),
        "faithfulness_scored_items": sum(
            1
            for row in rows
            if (row.get("generation_metrics") or {}).get("attribution_score") is not None
        ),
        "faithfulness_total_items": len(rows),
        # 归因的分句级汇总。分型用的信号是「无依据分句占比」，
        # 与 faithfulness（混入 no_citation 扣分）不是同一个东西，必须分开报。
        "attribution_clause_stats": M.attribution_clause_stats(
            row.get("attribution") for row in rows
        ),
        "refusal_success": (
            round(sum(1 for row in refused_rows if row.get("refused")) / len(refused_rows), 4)
            # retrieve 模式下拿不到 refused，不能把「没测」算成「全错」
            if refused_rows and any(row.get("refused") is not None for row in refused_rows)
            else None
        ),
        "over_refusal_rate": (
            round(sum(1 for row in answerable_rows if row.get("refused")) / len(answerable_rows), 4)
            if answerable_rows and any(row.get("refused") is not None for row in answerable_rows)
            else None
        ),
        "failure_distribution": M.distribution(failure_labels),
        "refusal_sources": refusal_sources,
        "citation_out_of_range_items": [
            row["id"] for row in rows if row.get("citation_ok") is False
        ],
        "wall_clock_ms": duration_ms,
        "disclaimer": (
            "以上指标由脚本按规则统计（来源命中 / 排序质量 / 关键词覆盖 / 逐句归因 / 拒答 / 引用越界），"
            "未做人工评测，也未与任何其他系统对比，不构成准确率结论。"
            "三组指标的分母不同（检索=有期望来源的条目、拒答=负样本、过度拒答=可答样本），不可互相比较。"
        ),
    }

    report = {
        "dataset": {
            "path": str(dataset_path),
            "name": dataset.get("name"),
            "version": dataset.get("version"),
            "items_used": len(rows),
            "mode": args.mode,
            "top_k": args.top_k,
            "base_url": base,
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "health": health,
        "metrics": metrics,
        "items": rows,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = RESULT_DIR / f"local_report_{stamp}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = RESULT_DIR / f"local_report_{stamp}.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")

    print()
    print("=" * 72)
    print(f"跑完 {len(rows)} 条，总耗时 {duration_ms} ms")
    print(f"分组                     : {metrics['groups']}")
    if metrics["groups"].get("retrieval_and_refusal_overlap"):
        print("                           （检索组与拒答组有重叠，勿直接相加）")
    print(f"相关性粒度               : {metrics['relevance_levels']}（snippet=含关键片段 / file=仅来源文件）")
    print(f"来源命中率(检索组)       : {metrics['source_hit_rate']}")
    print(f"Recall@{args.top_k}(片段级)       : {metrics['recall_at_k_snippet']}")
    print(f"Recall@{args.top_k}                : {metrics['recall_at_k']}")
    print(f"MRR                      : {metrics['mrr']}")
    print(f"nDCG@{args.top_k}                : {metrics['ndcg_at_k']}")
    print(f"Precision@{args.top_k}            : {metrics['precision_at_k']}")
    print(f"平均关键词覆盖率(计分条) : {metrics['avg_keyword_coverage_on_scored']}")
    print(
        f"归因得分(逐句)           : {metrics['faithfulness']}"
        f"（{metrics.get('faithfulness_scored_items')}/{metrics.get('faithfulness_total_items')} 条有可校验断言）"
    )
    clause = metrics.get("attribution_clause_stats") or {}
    if clause.get("checked_clauses"):
        print(
            f"归因分句构成             : 校验 {clause['checked_clauses']} 句 | "
            f"有依据 {clause['supported']} / 弱 {clause['weak']} / "
            f"无依据 {clause['unsupported']}({clause['unsupported_ratio']}) / "
            f"未标引用 {clause['no_citation']} / 缺口陈述跳过 {clause['meta_skipped']}"
        )
    print(f"拒答成功率(负样本)       : {metrics['refusal_success']}")
    print(f"过度拒答率(可答样本)     : {metrics['over_refusal_rate']}")
    print(f"引用越界条目             : {metrics['citation_out_of_range_items'] or '无'}")
    print(f"失败条目                 : {metrics['errors']}")
    if metrics.get("snippet_text_missing_items"):
        print()
        print("!! 警告：以下条目的检索结果没有返回片段正文，片段级 recall 会被判为 0。")
        print(f"   受影响条目：{', '.join(metrics['snippet_text_missing_items'])}")
        print("   检查接口是否返回 `text` 字段（/api/v1/retrieve 应带 include_text=True）。")
    if metrics.get("refusal_sources"):
        print(f"拒答来源               : {metrics['refusal_sources']}")
    if metrics["failure_distribution"]:
        print("失败分型：")
        for name, count in metrics["failure_distribution"].items():
            print(f"  {name:<8}: {count}")
    suspect_ids = [
        row.get("id")
        for row in rows
        if not row.get("must_refuse") and row.get("refusal_declared")
    ]
    if suspect_ids:
        print(f"标注复核队列             : {', '.join(filter(None, suspect_ids))}"
              "（可答样本但系统自述无依据，建议核对标注）")
    print(f"JSON 报告: {json_path}")
    print(f"Markdown : {md_path}")
    print("=" * 72)
    print("提醒：以上为规则统计值，不是准确率，也没有基线对比，请勿直接当作效果结论。")
    return report


def render_markdown(report: dict) -> str:
    metrics = report["metrics"]
    lines: List[str] = []
    lines.append("# AutoRAG 本地评测报告（规则统计）\n")
    lines.append(f"- 评测集：`{report['dataset']['path']}`")
    lines.append(f"- 条目数：{report['dataset']['items_used']}")
    lines.append(f"- 模式：`{report['dataset']['mode']}`，top_k={report['dataset']['top_k']}")
    lines.append(f"- 后端：{report['dataset']['base_url']}")
    lines.append(f"- 完成时间：{report['dataset']['finished_at']}\n")
    lines.append("## 汇总指标\n")
    lines.append(
        f"样本分组：{metrics.get('groups')}"
        "（三组分母不同，数字不可互相比较；带相关证据的负样本会同时进检索组与拒答组，"
        "所以 retrieval + refusal 可能大于总数，重叠数见 retrieval_and_refusal_overlap）"
    )
    lines.append(
        f"相关性粒度：{metrics.get('relevance_levels')} —— "
        "snippet 表示该条已标注关键原文短语，判定「片段里是否真有这句话」；"
        "file 表示只判来源文件。**两档粒度的数字不可混比。**\n"
    )
    lines.append("| 指标 | 值 | 说明 |")
    lines.append("| --- | --- | --- |")
    lines.append(f"| 来源命中率 | {metrics['source_hit_rate']} | 检索组，期望来源是否进 top_k |")
    lines.append(f"| Recall@k（片段级） | {metrics['recall_at_k_snippet']} | 仅统计已标注关键片段的条目 |")
    lines.append(f"| Recall@k（全部） | {metrics['recall_at_k']} | 含未标注的条目 |")
    lines.append(f"| MRR | {metrics['mrr']} | 首个相关结果排名的倒数 |")
    lines.append(f"| nDCG@k | {metrics['ndcg_at_k']} | 排序质量，理想排序仅在召回集内 |")
    lines.append(f"| Precision@k | {metrics['precision_at_k']} | 前 k 条里相关结果占比 |")
    lines.append(f"| 平均关键词覆盖率 | {metrics['avg_keyword_coverage_on_scored']} | 仅计有期望词的条目 |")
    lines.append(
        f"| 归因得分 | {metrics['faithfulness']} | 逐句归因；"
        f"分母为「有可校验断言」的 {metrics.get('faithfulness_scored_items')}"
        f"/{metrics.get('faithfulness_total_items')} 条（纯拒答无断言可校验，不计入） |"
    )
    _clause = metrics.get("attribution_clause_stats") or {}
    if _clause.get("checked_clauses"):
        lines.append(
            f"| 无依据分句占比 | {_clause['unsupported_ratio']} | "
            f"{_clause['unsupported']}/{_clause['checked_clauses']} 句；**这是分型实际使用的信号**"
            f"（阈值 {M.ATTRIBUTION_UNSUPPORTED_RATIO}，且至少 {M.ATTRIBUTION_MIN_CLAUSES} 句） |"
        )
        lines.append(
            f"| 未标引用分句占比 | {_clause['no_citation_ratio']} | "
            f"{_clause['no_citation']}/{_clause['checked_clauses']} 句；"
            f"格式问题，不计入归因不足判定 |"
        )
    lines.append(f"| 负样本拒答成功率 | {metrics['refusal_success']} | 分母为负样本条数 |")
    lines.append(f"| 过度拒答率 | {metrics['over_refusal_rate']} | 分母为可答样本条数 |")
    lines.append(f"| 引用越界条目 | {metrics['citation_out_of_range_items'] or '无'} | |")
    lines.append(f"| 失败条目数 | {metrics['errors']} | |")
    lines.append(f"| 脚本总耗时(ms) | {metrics['wall_clock_ms']} | |\n")
    lines.append(f"> {metrics['disclaimer']}\n")

    missing = metrics.get("snippet_text_missing_items") or []
    if missing:
        lines.append(
            f"> ⚠️ **警告**：{len(missing)} 条检索结果没有返回片段正文"
            f"（{', '.join(missing)}），片段级 recall 会被判为 0，本报告检索指标不可信。"
            "请确认接口返回了 `text` 字段。\n"
        )

    dist = metrics.get("failure_distribution") or {}
    if dist:
        lines.append("## 失败分型\n")
        lines.append("一条样本只给一个主标签（判定优先级见 `app/core/metrics.py`）。\n")
        lines.append("| 分型 | 条数 |")
        lines.append("| --- | --- |")
        for name, count in dist.items():
            lines.append(f"| {name} | {count} |")
        lines.append("")

    # 标注复核队列：可答样本里，系统自己声明「没有依据」的那些。
    # 这不是失败——它可能是诚实的部分覆盖；但它也是「标注可能标错了」的最强信号。
    # 实测 Q011「长期停放蓄电池会亏电吗」就是从这里冒出来的：系统说未找到依据，
    # 而 ground_truth 只给了不等价的检查周期，说明该条应当标为负样本。
    suspect = [
        row
        for row in report["items"]
        if not row.get("must_refuse") and row.get("refusal_declared")
    ]
    if suspect:
        lines.append("## 标注复核队列\n")
        lines.append(
            "以下**可答样本**里，系统自己声明了「未找到依据」。这可能是诚实的部分覆盖，"
            "也可能说明该条的核心提问在语料中根本没有依据、应当标为负样本。请人工确认。\n"
        )
        lines.append("| ID | 问题 | 关键词覆盖 | 拒答路径 | 判断提示 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in suspect:
            gm = row.get("generation_metrics") or {}
            kw = gm.get("keyword_coverage")
            hint = (
                "覆盖高：可能只是答得保守"
                if (kw or 0) >= 0.5
                else "覆盖低：核心提问可能无依据，建议核对 ground_truth"
            )
            lines.append(
                "| {id} | {q} | {kw} | {path} | {hint} |".format(
                    id=row.get("id", ""),
                    q=str(row.get("question", "")).replace("|", "\\|")[:40],
                    kw=kw,
                    path=row.get("refusal_path", ""),
                    hint=hint,
                )
            )
        lines.append("")

    lines.append("## 逐条结果\n")
    lines.append(
        "| ID | 分类 | 问题 | 命中来源 | Recall@k | 首命中排名 | nDCG | 关键词覆盖 | 归因 | 分型 | 耗时(ms) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in report["items"]:
        rm = row.get("retrieval_metrics") or {}
        gm = row.get("generation_metrics") or {}
        lines.append(
            "| {id} | {cat} | {q} | {hit} | {recall} | {rank} | {ndcg} | {kw} | {attr} | {label} | {ms} |".format(
                id=row.get("id", ""),
                cat=row.get("category", ""),
                q=str(row.get("question", "")).replace("|", "\\|"),
                hit="是" if row.get("source_hit") else "否",
                recall=rm.get("recall_at_k", "-"),
                rank=rm.get("first_relevant_rank", "-"),
                ndcg=rm.get("ndcg_at_k", "-"),
                kw=gm.get("keyword_coverage", row.get("kw_coverage", "-")),
                attr=gm.get("attribution_score", "-"),
                label=row.get("failure_label_cn") or "-",
                ms=row.get("elapsed_ms", ""),
            )
        )
    lines.append("\n## 服务状态快照\n")
    lines.append("```json")
    lines.append(json.dumps(report.get("health", {}), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("\n## 逐条回答原文（人工复核用）\n")
    for row in report["items"]:
        lines.append(f"### {row.get('id')} · {row.get('question')}\n")
        if row.get("error"):
            lines.append(f"- 错误：`{row['error']}`\n")
            continue
        lines.append(f"- 召回来源：{', '.join(row.get('retrieved_sources') or []) or '无'}")
        lines.append(f"- 期望来源：{', '.join(row.get('expected_sources') or []) or '（未填写）'}")
        lines.append(f"- ground_truth：{row.get('ground_truth') or '（未填写）'}\n")
        lines.append("**系统回答：**\n")
        lines.append("```text")
        lines.append((row.get("answer") or "（retrieve 模式无回答）").strip())
        lines.append("```\n")
    return "\n".join(lines)


# 重复运行时逐项统计的指标（路径指向 report["metrics"] 下的键）。
# 只列**会随大模型输出浮动**的指标；检索侧是确定性的，本来就不需要区间。
_REPEAT_METRICS: List[tuple] = [
    ("Recall@k", ("recall_at_k",)),
    ("MRR", ("mrr",)),
    ("nDCG@k", ("ndcg_at_k",)),
    ("Precision@k", ("precision_at_k",)),
    ("关键词覆盖率", ("avg_keyword_coverage_on_scored",)),
    ("归因得分", ("faithfulness",)),
    ("归因覆盖条目", ("faithfulness_scored_items",)),
    ("被校验分句", ("attribution_clause_stats", "checked_clauses")),
    ("无依据分句", ("attribution_clause_stats", "unsupported")),
    ("无依据占比", ("attribution_clause_stats", "unsupported_ratio")),
    ("未标引用分句", ("attribution_clause_stats", "no_citation")),
    ("拒答成功率", ("refusal_success",)),
    ("过度拒答率", ("over_refusal_rate",)),
]


def _pick(report: Dict[str, Any], path: tuple) -> Any:
    value: Any = report.get("metrics", {})
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _stat(values: List[Any]) -> Optional[Dict[str, float]]:
    nums = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not nums:
        return None
    return {
        "mean": round(sum(nums) / len(nums), 4),
        "min": min(nums),
        "max": max(nums),
    }


def summarize_repeats(reports: Sequence[Dict[str, Any]], args: argparse.Namespace) -> Path:
    """把 N 次运行聚合成「均值 / 最小 / 最大」，并落盘。

    为什么需要这一步：生成侧指标（faithfulness、被校验分句数、失败分型）
    每次都不同，单次值看起来像精确结论，实际是抽样。分型尤其危险 ——
    「通过 19」和「通过 18」在 20 条样本上就是 5 个百分点。
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rows: List[Dict[str, Any]] = []
    for name, path in _REPEAT_METRICS:
        stat = _stat([_pick(report, path) for report in reports])
        if stat is None:
            continue
        rows.append({"name": name, "series": [_pick(r, path) for r in reports], **stat})

    # 分型与拒答来源是计数分布，跨轮取每个标签的出现次数
    def _dist(key: str) -> Dict[str, List[Optional[int]]]:
        labels: List[str] = []
        for report in reports:
            for label in (report.get("metrics", {}).get(key) or {}):
                if label not in labels:
                    labels.append(label)
        return {
            label: [
                (report.get("metrics", {}).get(key) or {}).get(label)
                for report in reports
            ]
            for label in labels
        }

    distribution = _dist("failure_distribution")
    refusal_sources = _dist("refusal_sources")

    payload = {
        "runs": len(reports),
        "mode": args.mode,
        "top_k": args.top_k,
        "metrics": rows,
        "failure_distribution": distribution,
        "refusal_sources": refusal_sources,
        "note": (
            "聚合的是 N 次独立运行的结果。检索指标是确定性的（min=max）；"
            "生成侧随大模型输出浮动，应读作区间而不是单点。"
        ),
    }
    out_dir = RESULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"repeat_{stamp}.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = out_dir / f"repeat_{stamp}.md"

    lines: List[str] = [
        f"# 重复运行汇总（N={len(reports)}）\n",
        f"- 模式：`{args.mode}`，top_k={args.top_k}",
        f"- 每轮各自产出独立报告，本文件只做聚合",
        f"- {payload['note']}\n",
        "| 指标 | 均值 | 最小 | 最大 | 判定 |",
        "| --- | --- | --- | --- | --- |",
    ]
    print()
    print("=" * 72)
    print(f"重复运行汇总（N={len(reports)}，mode={args.mode}，top_k={args.top_k}）")
    print(f"{'指标':<18}{'均值':>10}{'最小':>10}{'最大':>10}   判定")
    for row in rows:
        stable = "恒定" if row["min"] == row["max"] else "浮动"
        print(f"{row['name']:<18}{row['mean']:>10}{row['min']:>10}{row['max']:>10}   {stable}")
        lines.append(
            f"| {row['name']} | {row['mean']} | {row['min']} | {row['max']} | {stable} |"
        )

    for title, dist in (("失败分型", distribution), ("拒答来源", refusal_sources)):
        if not dist:
            continue
        print(f"\n{title}（每轮）:")
        lines.append(f"\n## {title}（每轮）\n")
        lines.append("| 标签 | " + " | ".join(f"第 {i+1} 轮" for i in range(len(reports))) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in reports) + " |")
        for label, series in dist.items():
            shown = " / ".join("—" if v is None else str(v) for v in series)
            print(f"  {label:<10}: {shown}")
            lines.append(
                f"| {label} | " + " | ".join("—" if v is None else str(v) for v in series) + " |"
            )

    # 之前这里只拼了 lines 却没落盘 —— 终端能看到、文件里没有，等于白拼。
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=" * 72)
    print(f"汇总报告: {md_path}")
    print(f"          {json_path}")
    return json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoRAG 本地评测脚本（规则统计）")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="评测集 JSON 路径")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="服务地址")
    parser.add_argument("--top-k", type=int, default=5, help="召回条数")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0 表示全部）")
    parser.add_argument(
        "--mode",
        choices=["chat", "retrieve"],
        default="chat",
        help="chat=完整问答（消耗 token）；retrieve=只测召回",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help=(
            "重复跑 N 次并给出均值/区间（N>1 时额外产出汇总报告）。"
            "生成侧指标随大模型输出浮动，单次值不能当基线用，取 3 次可看出波动范围。"
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repeat = max(1, int(getattr(args, "repeat", 1)))
    if repeat == 1:
        return 0 if run(args) is not None else 2

    reports: List[Dict[str, Any]] = []
    for index in range(repeat):
        print()
        print("#" * 24 + f" 第 {index + 1}/{repeat} 次运行 " + "#" * 24)
        report = run(args)
        if report is None:
            return 2
        reports.append(report)

    summarize_repeats(reports, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
