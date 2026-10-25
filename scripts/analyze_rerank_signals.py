#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""重排信号分析工具：采集各信号取值与相关性，评估其排序能力。

用途：调 `RERANK_WEIGHTS` 之前先跑这个，看数据再动手。2026-10-25 用它得出的结论是
「当前权重已是局部最优，按信号分析去调反而全面变差」——详见 README 12.8。

必须先把服务跑起来（脚本走 HTTP，复用真实检索链路）：
    uvicorn app.main:app --port 8000
    python scripts/analyze_rerank_signals.py

输出：
  eval/results/signals_dataset.json  原始样本（14 条 query × 20 候选）
  终端：每个信号的相关/不相关均值、区分度、名义权重、实际贡献占比
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.core.metrics import normalize_for_snippet, normalize_source  # noqa: E402

BASE = "http://127.0.0.1:8000"
TOPK = 20


def retrieve(query: str):
    req = urllib.request.Request(
        f"{BASE}/api/v1/retrieve",
        data=json.dumps({"query": query, "top_k": TOPK}, ensure_ascii=False).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read())


def is_relevant(entry, item) -> bool:
    """与评测口径一致：来源命中 且 片段正文含关键短语；无短语则只判来源。"""
    sources = {normalize_source(s) for s in (item.get("expected_sources") or [])}
    if not sources:
        return False
    src = normalize_source(entry.get("source", ""))
    if not (src in sources or any(src.endswith("/" + s) for s in sources)):
        return False
    snippets = [s for s in (item.get("expected_snippets") or []) if str(s).strip()]
    if not snippets:
        return True
    text = normalize_for_snippet(entry.get("text") or "")
    return any(normalize_for_snippet(s) in text for s in snippets)


def main() -> int:
    dataset = json.loads((ROOT / "eval" / "eval_set.json").read_text(encoding="utf-8"))["items"]
    out = []
    for item in dataset:
        if item.get("must_refuse"):
            continue
        data = retrieve(item["question"])
        candidates = []
        for entry in data.get("results") or []:
            candidates.append(
                {
                    "rank": entry["rank"],
                    "chunk_id": entry["chunk_id"],
                    "relevant": is_relevant(entry, item),
                    "signals": entry.get("signals") or {},
                    "final_score": entry.get("score"),
                    "fusion_score": entry.get("fusion_score"),
                    "rerank_score": entry.get("rerank_score"),
                }
            )
        out.append({"id": item["id"], "question": item["question"], "candidates": candidates})
        rel = sum(1 for c in candidates if c["relevant"])
        print(f"{item['id']}: {len(candidates)} 候选，相关 {rel}")

    path = ROOT / "eval" / "results" / "signals_dataset.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {path}")

    # 快速体检：每个信号的区分度 + 实际贡献
    keys = ["fusion", "coverage", "precision", "cjk", "section", "phrase"]
    rel_vals = {k: [] for k in keys}
    irr_vals = {k: [] for k in keys}
    for entry in out:
        for c in entry["candidates"]:
            target = rel_vals if c["relevant"] else irr_vals
            for k in keys:
                target[k].append(float((c["signals"] or {}).get(k, 0.0)))

    weights = {"fusion": 0.54, "coverage": 0.12, "precision": 0.12,
               "cjk": 0.08, "section": 0.08, "phrase": 0.06}

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    print("\n信号          相关均值   不相关均值   差值     名义权重  实际贡献占比")
    contrib = {}
    for k in keys:
        r, i = mean(rel_vals[k]), mean(irr_vals[k])
        contrib[k] = weights[k] * r
    total = sum(contrib.values()) or 1.0
    for k in keys:
        r, i = mean(rel_vals[k]), mean(irr_vals[k])
        print(f"  {k:<10} {r:>9.4f} {i:>11.4f} {r - i:>8.4f}   {weights[k]:>6.2f}   "
              f"{contrib[k] / total * 100:>8.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
