#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""入库脚本（只用标准库）。

用途：把 data/documents 目录下的文档切分、向量化并写入向量库。
    1) 先启动服务： uvicorn app.main:app --port 8000
    2) 增量入库：   python scripts/ingest.py
    3) 全量重建：   python scripts/ingest.py --rebuild
    4) 换 embedding 模型后必须重建，否则新旧向量不在同一空间。

注意：服务必须先启动。写这个脚本而不是"离线直接操作 Chroma"，
是为了保证 BM25 索引、registry 增量记录、向量库三者状态一致。
本项目刻意不引入 chromadb 以外的重型依赖，所以这里用 urllib 而不是 requests/httpx。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 1800.0) -> dict:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"[错误] HTTP {exc.code}：{detail[:500]}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"[错误] 无法连接 {url}：{exc.reason}\n"
            "请先启动服务： uvicorn app.main:app --port 8000"
        ) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="AutoRAG 知识库入库脚本")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--rebuild", action="store_true", help="清空向量库后全量重建")
    parser.add_argument("--reset-registry", action="store_true", help="忽略增量记录，全部重切")
    parser.add_argument("--path", action="append", default=None, help="只处理指定相对路径，可重复")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    health = http_json("GET", f"{base}/api/v1/health", timeout=30.0)
    print(f"服务状态：{health.get('status')}，向量库条数={health.get('collection_count')}")
    for note in health.get("detail", {}).get("notes", []) or []:
        print(f"[提示] {note}")

    documents_dir = health.get("detail", {}).get("config", {}).get("documents_dir")
    print(f"知识库目录：{documents_dir}")
    directory = Path(documents_dir) if documents_dir else PROJECT_ROOT / "data" / "documents"
    if directory.exists():
        files = [p for p in directory.rglob("*") if p.is_file() and not p.name.startswith(".")]
        print(f"目录内文件数：{len(files)}")
        for path in files[:20]:
            print(f"  - {path.relative_to(directory).as_posix()}")
        if len(files) > 20:
            print(f"  …… 其余 {len(files) - 20} 个文件省略")
    else:
        print(f"[警告] 目录不存在：{directory}（服务会自动创建）")

    payload = {"rebuild": bool(args.rebuild), "reset_registry": bool(args.reset_registry)}
    if args.path:
        payload["paths"] = args.path

    print(f"\n开始入库：{json.dumps(payload, ensure_ascii=False)}")
    started = time.perf_counter()
    result = http_json("POST", f"{base}/api/v1/ingest", payload)
    elapsed = time.perf_counter() - started

    print("\n入库结果：")
    for key in ("scanned", "indexed", "skipped", "failed", "deleted", "total_chunks_in_store"):
        print(f"  {key:<24}= {result.get(key)}")
    print(f"  {'duration_ms(服务端)':<24}= {result.get('duration_ms')}")
    print(f"  {'脚本耗时(ms)':<24}= {int(elapsed * 1000)}")

    files = result.get("files") or []
    if files:
        print("\n逐文件明细：")
        for item in files:
            line = f"  [{item.get('status')}] {item.get('source')} chunks={item.get('chunks', 0)}"
            if item.get("error"):
                line += f" error={item['error']}"
            print(line)

    if result.get("failed"):
        print("\n[注意] 存在失败文件，请按上面 error 信息排查（多为缺可选依赖或文件编码问题）。")
    return 0 if not result.get("failed") else 1


if __name__ == "__main__":
    sys.exit(main())
