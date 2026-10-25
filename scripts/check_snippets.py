#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""校验评测集里的 expected_snippets 标注是否有效。

为什么需要这个脚本
------------------
片段级相关性判定依赖「关键短语能在片段正文里被找到」。如果标注时写错一个字，
或者短语正好落在两个片段的切分边界上，判定就会永远不命中 —— 而且
**不会报错，只会让 recall 悄悄变成 0**，看起来像「检索能力下降」。

本脚本把这类问题在标注阶段就拦住。两层层检查缺一不可：
1. 文档全文里存在 —— 抓标注错字；
2. 切分后的某个片段里存在 —— 抓「被切断」的情况。

用法：
    python scripts/check_snippets.py

退出码：0 = 全部通过；1 = 有问题（会逐条列出）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.metrics import normalize_for_snippet  # noqa: E402

# 两个评测集都要校验：多轮评测集（multi_turn_set.json）的标注一旦写错，
# 失败方式与单轮完全一样 —— 不会报错，只会让 recall 悄悄变成 0。
DATASETS = (
    PROJECT_ROOT / "eval" / "eval_set.json",
    PROJECT_ROOT / "eval" / "multi_turn_set.json",
)
# 向后兼容的旧名字（有些脚本/文档引用它）
DATASET = DATASETS[0]
DOCS_DIR = PROJECT_ROOT / "data" / "documents"
BM25_INDEX = PROJECT_ROOT / "data" / "bm25_index.json"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    all_problems: List[Tuple[str, str, str]] = []
    total_items = 0
    total_annotated = 0
    total_checked = 0

    doc_text = {
        path.name: normalize_for_snippet(path.read_text(encoding="utf-8"))
        for path in DOCS_DIR.glob("*.md")
    }
    chunks: List[Dict[str, Any]] = []
    if BM25_INDEX.exists():
        chunks = load_json(BM25_INDEX).get("chunks") or []
    else:
        print("[警告] 找不到 data/bm25_index.json，只能做第一层校验（文档全文）。")

    for dataset_path in DATASETS:
        if not dataset_path.exists():
            print(f"[跳过] {dataset_path.name} 不存在")
            continue
        dataset = load_json(dataset_path)
        items = dataset.get("items") or []
        problems, annotated, checked = check_dataset(
            dataset_path, items, doc_text, chunks
        )
        total_items += len(items)
        total_annotated += annotated
        total_checked += checked
        all_problems.extend(problems)

    print(f"评测条目共 {total_items} 条，其中 {total_annotated} 条带关键片段标注，共校验 {total_checked} 个短语。")
    if all_problems:
        print(f"\n发现 {len(all_problems)} 处问题：")
        for qid, snippet, why in all_problems:
            print(f"  [{qid}] {snippet}")
            print(f"          {why}")
        print(
            "\n提示：修好标注后重新跑一次；若短语确实跨了切分边界，"
            "把它换成同一段里更短、更靠前的连续文本。"
        )
        return 1

    print("全部通过：每个关键短语都能在原文与切分片段中被找到。")
    _report_section_consistency()
    return 0


def check_dataset(
    dataset_path: Path,
    items: Sequence[Dict[str, Any]],
    doc_text: Dict[str, str],
    chunks: List[Dict[str, Any]],
) -> Tuple[List[Tuple[str, str, str]], int, int]:
    """校验单个评测集，返回 (问题列表, 带标注条数, 校验短语数)。"""
    problems: List[Tuple[str, str, str]] = []
    annotated = 0
    checked = 0

    doc_text = {
        path.name: normalize_for_snippet(path.read_text(encoding="utf-8"))
        for path in DOCS_DIR.glob("*.md")
    }

    chunks: List[Dict[str, Any]] = []
    if BM25_INDEX.exists():
        chunks = load_json(BM25_INDEX).get("chunks") or []
    else:
        print("[警告] 找不到 data/bm25_index.json，只能做第一层校验（文档全文）。")

    problems: List[Tuple[str, str, str]] = []
    checked = 0
    annotated = 0

    for item in items:
        qid = item.get("id", "?")
        snippets = [str(s) for s in (item.get("expected_snippets") or []) if str(s).strip()]
        if not snippets:
            continue
        annotated += 1
        sources = [str(s) for s in (item.get("expected_sources") or [])]

        if not sources:
            problems.append((qid, "-", "标了关键片段却没有 expected_sources，无法绑定来源"))
            continue

        for snippet in snippets:
            checked += 1
            norm = normalize_for_snippet(snippet)

            in_doc = [name for name in sources if norm in doc_text.get(name, "")]
            if not in_doc:
                problems.append((qid, snippet, "期望来源的文档全文中找不到"))
                continue

            if chunks:
                hit = [
                    c
                    for c in chunks
                    if c.get("source") in sources
                    and norm in normalize_for_snippet(str(c.get("text") or ""))
                ]
                if not hit:
                    problems.append(
                        (qid, snippet, "文档里有，但切分后的任何片段里都没有（疑似被切断）")
                    )

    return problems, annotated, checked


def _report_section_consistency() -> None:
    """第三层：片段元数据一致性（不阻断，仅告警）。

    切分时 Markdown 的水平分割线 `---` 会被当成**上一章节**的正文保留，
    打包阶段再把下一章节的段落并进来，而打包取的是「缓冲区第一个片段的 section」，
    于是整块的 section 被标成上一章节。实测 79 个片段里有 4 个中招。
    这不影响片段正文，但上下文增强会把 section 拼进索引文本 —— 错标签等于灌错语义。
    """
    section_bad = check_section_consistency()
    if section_bad:
        print(f"\n[警告] {len(section_bad)} 个片段的 section 疑似标错（不影响召回，但会影响上下文增强）：")
        for item in section_bad:
            print(f"  {item['source']} pos={item['position']} section={item['section']}")
        print(
            "  判据：片段正文以水平分割线开头 —— 说明它把上一章节的分隔符和下一章节的正文"
            "打包在了一起，section 取了上一章节。成因与修法见 README 12.12。"
        )
    else:
        print("片段元数据一致性：全部通过。")


def check_section_consistency() -> List[Dict[str, Any]]:
    """找出 section 与正文不一致的片段（不阻断，仅告警）。

    判据刻意保持保守：只报「正文以 `---` 开头」这一种确定性的错配，
    不猜语义。宁可漏报也不误报 —— 一个会误报的检查会很快被忽略。
    """
    if not BM25_INDEX.exists():
        return []
    index = load_json(BM25_INDEX)
    findings: List[Dict[str, Any]] = []
    for chunk in index.get("chunks") or []:
        text = str(chunk.get("text") or "").lstrip()
        if text.startswith("---") or text.startswith("***"):
            findings.append(
                {
                    "source": chunk.get("source", ""),
                    "position": chunk.get("position"),
                    "section": chunk.get("section", ""),
                }
            )
    return findings


if __name__ == "__main__":
    sys.exit(main())
