#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查文档页的 front-matter 锚点是否指向真实存在的代码文件。

为什么需要这个脚本：
    文档页和代码之间的绑定是纯手工维护的。手工维护一定会漂移——
    改个文件名、挪个目录，文档不会报错，只是悄悄指向一个不存在的路径。
    这个脚本把"悄悄"变成"报错"。

约定（写在每个 md 文件开头的 front-matter 里）：

    ---
    sources:
      - app/core/chunker.py
    ---
        该页依赖这些代码。改到它们，这一页就该重新看一遍。

    ---
    auto_update: false
    note: 实测快照，不随代码自动改写
    ---
        冻结页：观点、快照、人工维护的清单。工具不允许自动改写它们。

    没有 front-matter 的页面 = 未绑定，会被列为待确认，不是错误。

用法：
    python scripts/check_doc_anchors.py              # 检查
    python scripts/check_doc_anchors.py --list       # 附带列出每页绑定情况
    python scripts/check_doc_anchors.py --json       # 机器可读输出

退出码：0 全部有效；1 存在失效锚点。
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys

DOCS_DIR = os.path.join("wiki", "docs")
SOURCE_RE = re.compile(r"^\s*-\s+(\S.*?)\s*$", re.M)


def parse_page(path: str) -> dict:
    """读一页的锚点。返回 {page, state, sources, note}。"""
    text = io.open(path, encoding="utf-8").read()
    page = os.path.relpath(path, DOCS_DIR).replace("\\", "/")

    if not text.startswith("---"):
        return {"page": page, "state": "unbound", "sources": [], "note": ""}

    parts = text.split("---", 2)
    if len(parts) < 3:
        return {"page": page, "state": "unbound", "sources": [], "note": ""}

    head = parts[1]
    if re.search(r"^auto_update:\s*false\s*$", head, re.M):
        note = ""
        m = re.search(r"^note:\s*(.+)$", head, re.M)
        if m:
            note = m.group(1).strip()
        return {"page": page, "state": "frozen", "sources": [], "note": note}

    sources = [s for s in SOURCE_RE.findall(head) if not s.startswith("note:")]
    return {"page": page, "state": "bound", "sources": sources, "note": ""}


def main() -> int:
    ap = argparse.ArgumentParser(description="检查文档页锚点")
    ap.add_argument("--list", action="store_true", help="列出每页绑定情况")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--root", default=".", help="仓库根目录，默认当前目录")
    args = ap.parse_args()

    os.chdir(args.root)
    if not os.path.isdir(DOCS_DIR):
        print(f"[×] 找不到 {DOCS_DIR}/，请在仓库根目录执行", file=sys.stderr)
        return 1

    pages = [parse_page(p) for p in sorted(glob.glob(f"{DOCS_DIR}/**/*.md", recursive=True))]

    broken = []
    for p in pages:
        for src in p["sources"]:
            if not os.path.exists(src):
                broken.append({"page": p["page"], "source": src})

    counts = {"bound": 0, "frozen": 0, "unbound": 0}
    total_anchors = 0
    for p in pages:
        counts[p["state"]] += 1
        total_anchors += len(p["sources"])

    result = {
        "pages": len(pages),
        "anchors": total_anchors,
        "broken": broken,
        "counts": counts,
        "detail": pages,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"文档页 {len(pages)} 个：绑定 {counts['bound']} · 冻结 {counts['frozen']} · 未绑定 {counts['unbound']}")
        print(f"锚点 {total_anchors} 个，失效 {len(broken)} 个")
        if broken:
            print("\n失效锚点（文件已不存在，文档已经指向空气）：")
            for b in broken:
                print(f"  × {b['page']}  ->  {b['source']}")
        if args.list:
            print()
            for p in pages:
                if p["state"] == "bound":
                    print(f"  [{len(p['sources'])}] {p['page']}")
                elif p["state"] == "frozen":
                    print(f"  [冻] {p['page']}  （{p['note']}）")
                else:
                    print(f"  [--] {p['page']}  未绑定，需人工确认")

    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
