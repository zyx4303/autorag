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
    refusal_ok    : 期望拒答的条目是否真的拒答了
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


def source_hit(expected: Sequence[str], retrieved: Sequence[dict]) -> bool:
    if not expected or is_placeholder(list(expected)):
        return False
    hit_sources = {str(item.get("source", "")) for item in retrieved}
    for want in expected:
        want_norm = str(want).replace("\\", "/").strip()
        if want_norm in hit_sources:
            return True
        if any(source.endswith("/" + want_norm) for source in hit_sources):
            return True
    return False


def keyword_coverage(expected: Sequence[str], answer: str) -> float:
    if not expected or is_placeholder(list(expected)):
        return 0.0
    lowered = answer.lower()
    hits = sum(1 for word in expected if str(word).strip().lower() in lowered)
    return round(hits / len(expected), 4)


def citation_check(answer: str, citations: Sequence[dict]) -> bool:
    """回答里的 [n] 是否都能在实际 citations 中找到对应编号。"""
    import re

    indices = {int(match.group(1)) for match in re.finditer(r"\[(\d{1,2})\]", answer)}
    if not indices:
        return True  # 没有引用角标不算越界，但会在报告里由 has_citation 字段体现
    valid = {int(item.get("index", -1)) for item in citations}
    return indices.issubset(valid)


def run(args: argparse.Namespace) -> int:
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
        return 2
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

    rows: List[Dict[str, Any]] = []
    started_all = time.perf_counter()

    for index, item in enumerate(items, start=1):
        item_id = item.get("id", f"item{index}")
        question = item.get("question", "")
        expected_sources = item.get("expected_sources") or []
        expected_keywords = item.get("expected_keywords") or []
        must_refuse = bool(item.get("must_refuse"))

        row: Dict[str, Any] = {
            "id": item_id,
            "category": item.get("category", ""),
            "question": question,
            "must_refuse": must_refuse,
            "ground_truth": item.get("ground_truth", ""),
            "placeholder_fields": [
                field
                for field in ("ground_truth", "expected_sources", "expected_keywords")
                if is_placeholder(item.get(field))
            ],
        }

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
                data = http_json("POST", chat_url, payload)
                retrieved = data.get("retrieved") or []
                answer = str(data.get("answer") or "")
                citations = data.get("citations") or []
                row["answer"] = answer
                row["citations"] = citations
                row["refused"] = bool(data.get("refused"))
                row["latency"] = data.get("latency")
                row["usage"] = data.get("usage")
                row["warnings"] = data.get("warnings") or []
                row["citation_ok"] = citation_check(answer, citations)
                row["has_citation_marker"] = bool(citations)
                row["kw_coverage"] = keyword_coverage(expected_keywords, answer)
                row["scored"] = not is_placeholder(expected_keywords)
                if must_refuse:
                    row["refusal_ok"] = bool(data.get("refused")) or (
                        "未找到" in answer or "无法" in answer
                    )
        except RuntimeError as exc:
            row["error"] = str(exc)

        row["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        row["retrieved_sources"] = [str(entry.get("source", "")) for entry in retrieved]
        row["source_hit"] = source_hit(expected_sources, retrieved)
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

    scored_rows = [row for row in rows if row.get("scored")]
    refused_rows = [row for row in rows if row.get("must_refuse")]
    metrics = {
        "total_items": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "source_hit_rate": (
            round(sum(1 for row in rows if row.get("source_hit")) / len(rows), 4) if rows else 0.0
        ),
        "avg_keyword_coverage_on_scored": (
            round(sum(row["kw_coverage"] for row in scored_rows) / len(scored_rows), 4)
            if scored_rows
            else None
        ),
        "refusal_success": (
            round(sum(1 for row in refused_rows if row.get("refusal_ok")) / len(refused_rows), 4)
            if refused_rows
            else None
        ),
        "citation_out_of_range_items": [
            row["id"] for row in rows if row.get("citation_ok") is False
        ],
        "wall_clock_ms": duration_ms,
        "disclaimer": (
            "以上指标由脚本按规则统计（来源命中 / 关键词覆盖 / 拒答成功 / 引用越界），"
            "未做人工评测，也未与任何其他系统对比，不构成准确率结论。"
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
    print(f"来源命中率(规则统计)     : {metrics['source_hit_rate']}")
    print(f"平均关键词覆盖率(计分条) : {metrics['avg_keyword_coverage_on_scored']}")
    print(f"拒答成功率(负样本)       : {metrics['refusal_success']}")
    print(f"引用越界条目             : {metrics['citation_out_of_range_items'] or '无'}")
    print(f"失败条目                 : {metrics['errors']}")
    print(f"JSON 报告: {json_path}")
    print(f"Markdown : {md_path}")
    print("=" * 72)
    print("提醒：以上为规则统计值，不是准确率，也没有基线对比，请勿直接当作效果结论。")
    return 0


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
    lines.append("| 指标 | 值 |")
    lines.append("| --- | --- |")
    lines.append(f"| 来源命中率（规则统计） | {metrics['source_hit_rate']} |")
    lines.append(f"| 平均关键词覆盖率（仅计分条目） | {metrics['avg_keyword_coverage_on_scored']} |")
    lines.append(f"| 负样本拒答成功率 | {metrics['refusal_success']} |")
    lines.append(f"| 引用越界条目 | {metrics['citation_out_of_range_items'] or '无'} |")
    lines.append(f"| 失败条目数 | {metrics['errors']} |")
    lines.append(f"| 脚本总耗时(ms) | {metrics['wall_clock_ms']} |\n")
    lines.append(f"> {metrics['disclaimer']}\n")
    lines.append("## 逐条结果\n")
    lines.append("| ID | 分类 | 问题 | 命中来源 | 关键词覆盖 | 拒答 | 引用越界 | 耗时(ms) | 备注 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in report["items"]:
        note = row.get("error") or ("待补充字段：" + ",".join(row.get("placeholder_fields") or []) or "")
        lines.append(
            "| {id} | {cat} | {q} | {hit} | {kw} | {ref} | {cite} | {ms} | {note} |".format(
                id=row.get("id", ""),
                cat=row.get("category", ""),
                q=str(row.get("question", "")).replace("|", "\\|"),
                hit="是" if row.get("source_hit") else "否",
                kw=row.get("kw_coverage", "-"),
                ref=row.get("refused", "-") if row.get("refused") is not None else "-",
                cite="是" if row.get("citation_ok") is False else "否",
                ms=row.get("elapsed_ms", ""),
                note=str(note).replace("|", "\\|")[:120],
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
    return parser


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))
