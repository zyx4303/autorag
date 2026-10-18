#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""根据代码改动，算出哪些文档页需要复查。

锚点（每个 md 开头的 sources）是一张"文档页 → 代码文件"的对照表。
这个脚本反着用它：拿到一次改动，反查命中了哪些页。

用法：
    python scripts/doc_impact.py                  # 工作区未提交的改动
    python scripts/doc_impact.py --staged         # 已 staged 的改动
    python scripts/doc_impact.py --since HEAD~3   # 最近若干个提交
    python scripts/doc_impact.py --json           # 机器可读

输出分三块：
    1. 受影响页面   —— 改到的文件被哪些页引用了，这些页建议复查
    2. 未覆盖的改动 —— 改到的文件没有任何页面引用，说明这块没人写文档
    3. 直接改了文档 —— 尤其冻结页被改，需要确认是人工更新而不是被程序覆盖

退出码：0 无影响或已列出；2 存在锚点失效（文件被删/改名）。
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import subprocess
import sys

DOCS_DIR = "wiki/docs"  # 统一用正斜杠：git 输出的路径也是正斜杠，避免 Windows 上 startswith 失配
SOURCE_RE = re.compile(r"^\s*-\s+(\S.*?)\s*$", re.M)


def load_anchors() -> dict:
    """返回 {page: {state, sources, note}}，page 是相对 wiki/docs 的路径。"""
    pages = {}
    for path in sorted(glob.glob(f"{DOCS_DIR}/**/*.md", recursive=True)):
        rel = os.path.relpath(path, DOCS_DIR).replace("\\", "/")
        text = io.open(path, encoding="utf-8").read()
        if not text.startswith("---"):
            pages[rel] = {"state": "unbound", "sources": [], "note": ""}
            continue
        parts = text.split("---", 2)
        if len(parts) < 3:
            pages[rel] = {"state": "unbound", "sources": [], "note": ""}
            continue
        head = parts[1]
        if re.search(r"^auto_update:\s*false\s*$", head, re.M):
            note = re.search(r"^note:\s*(.+)$", head, re.M)
            pages[rel] = {"state": "frozen", "sources": [], "note": note.group(1).strip() if note else ""}
            continue
        srcs = [s.strip() for s in SOURCE_RE.findall(head) if not s.startswith("note:")]
        pages[rel] = {"state": "bound", "sources": srcs, "note": ""}
    return pages


def git_changed(mode: str, since: str) -> list:
    """返回 [(status, path, adds, dels)]。status 取 M/A/D/R。"""
    if mode == "staged":
        args = ["git", "diff", "--cached", "--name-status"]
    elif mode == "since":
        args = ["git", "diff", since, "--name-status"]
    else:
        args = ["git", "diff", "HEAD", "--name-status"]
    out = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        print(f"[×] git 调用失败：{out.stderr.strip()}", file=sys.stderr)
        return None

    status_by_path = {}
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        st = parts[0][0]
        path = parts[-1] if st in ("R", "C") else parts[1]
        status_by_path[path.replace("\\", "/")] = st

    stat = {}
    out2 = subprocess.run(
        ["git", "diff", "--numstat"] + (["--cached"] if mode == "staged" else ([since] if mode == "since" else ["HEAD"])),
        capture_output=True, text=True, encoding="utf-8",
    )
    for line in out2.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        try:
            stat[parts[2].replace("\\", "/")] = (int(parts[0]) if parts[0] != "-" else 0,
                                                int(parts[1]) if parts[1] != "-" else 0)
        except ValueError:
            pass

    return [(st, p, *stat.get(p, (0, 0))) for p, st in status_by_path.items()]


def main() -> int:
    ap = argparse.ArgumentParser(description="根据代码改动反查受影响的文档页")
    ap.add_argument("--staged", action="store_true", help="只看已 staged 的改动")
    ap.add_argument("--since", default=None, help="看某个提交以来的改动，如 HEAD~3")
    ap.add_argument("--json", action="store_true", help="机器可读输出")
    ap.add_argument("--root", default=".", help="仓库根目录")
    args = ap.parse_args()

    os.chdir(args.root)
    if not os.path.isdir(DOCS_DIR):
        print(f"[×] 找不到 {DOCS_DIR}/，请在仓库根目录执行", file=sys.stderr)
        return 1

    mode = "staged" if args.staged else ("since" if args.since else "worktree")
    changes = git_changed(mode, args.since or "HEAD")
    if changes is None:
        # 不能当成"没有改动"返回 0：基准分支写错时，CI 会变成永远通过。
        print("[×] 无法计算改动范围，请确认 --since 的引用是否存在。", file=sys.stderr)
        return 3

    if not changes:
        print("没有检测到改动。")
        return 0

    pages = load_anchors()
    index = {}  # 代码文件 -> [页面]
    for page, meta in pages.items():
        for src in meta["sources"]:
            index.setdefault(src, []).append(page)

    hit = {}       # page -> [(file, status)]
    uncovered = [] # 改了但没有页面引用的文件
    doc_edits = [] # 直接改了文档页
    broken = []    # 改动删掉了被引用的文件

    for status, path, adds, dels in changes:
        if path.startswith(DOCS_DIR + "/"):
            rel = os.path.relpath(path, DOCS_DIR).replace("\\", "/")
            if rel.endswith(".md") and rel in pages:
                doc_edits.append({"page": rel, "state": pages[rel]["state"],
                                  "note": pages[rel]["note"], "adds": adds, "dels": dels})
            continue
        if status == "D":
            if path in index:
                broken.append({"file": path, "pages": index[path]})
            continue
        if path in index:
            for page in index[path]:
                hit.setdefault(page, []).append({"file": path, "status": status, "adds": adds, "dels": dels})
        else:
            uncovered.append({"file": path, "status": status, "adds": adds, "dels": dels})

    result = {
        "mode": mode,
        "changed_files": len(changes),
        "affected_pages": hit,
        "uncovered": uncovered,
        "doc_edits": doc_edits,
        "broken_anchors": broken,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return 2 if broken else 0

    print(f"改动 {len(changes)} 个文件 → 命中 {len(hit)} 个文档页\n")

    if hit:
        print("需要复查的文档页：")
        for page in sorted(hit, key=lambda p: -len(hit[p])):
            print(f"  · {page}")
            for h in hit[page]:
                print(f"      ← {h['file']}  ({h['status']}, +{h['adds']}/-{h['dels']})")
        print()

    if uncovered:
        print("改了但没有任何文档页引用（这块没人写文档）：")
        for u in uncovered:
            print(f"  · {u['file']}  ({u['status']}, +{u['adds']}/-{u['dels']})")
        print()

    if doc_edits:
        print("直接改动的文档页：")
        for d in doc_edits:
            tag = "冻结页，请确认是人工更新" if d["state"] == "frozen" else "普通页"
            print(f"  · {d['page']}  ({tag})")
        print()

    if broken:
        print("锚点失效（被引用的文件被删除或改名）：")
        for b in broken:
            print(f"  · {b['file']}  仍被这些页引用：{', '.join(b['pages'])}")
        print()
        return 2

    if not hit and not uncovered and not doc_edits:
        print("改动未落到任何文档范围。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
