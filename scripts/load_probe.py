"""并发压测：测量服务在并发请求下的延迟分布（单请求 P95 之外的另一半真相）。

用法::

    python scripts/load_probe.py --concurrency 5 --total 15

测什么
------
单请求 P95（/metrics）只反映「没有竞争时多快」。真实部署里用户是并发来的，
竞争点有三个：

1. **LLM API 并发**——外部限流（429），客户端已有重试+退避兜底，但重试会拉长尾延迟；
2. **to_thread 线程池**——精排/切分在线程池里跑，池满则排队；
3. **Chroma / BM25 并发读**——预期无锁竞争，但要数据证明。

判定方法：同一组问题，先串行跑一遍作基线，再按目标并发跑；
比较 P50/P95/最大值与错误数。**错误 = 0 是硬门槛**——延迟涨可以解释，
报错不能。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from typing import Any, Dict, List, Tuple

import httpx

BASE = "http://127.0.0.1:8000"
API_KEY = ""

# 混合负载：可答题（两轮不同问法）、重复题（测缓存路径）、负样本（测拒答路径）
QUESTIONS = [
    "示例车型 A 的首保是多少公里？",
    "火花塞多久更换一次？",
    "P0420 故障码是什么意思？",
    "三包凭证遗失了怎么办？",
    "量子纠缠对悬挂系统的影响 zzz83",  # 负样本
]


def _percentile(sorted_values: List[float], pct: float) -> float:
    index = min(len(sorted_values) - 1, max(0, round(pct / 100 * (len(sorted_values) - 1))))
    return sorted_values[index]


def _stats(samples: List[float]) -> Dict[str, float]:
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "p50": _percentile(ordered, 50),
        "p95": _percentile(ordered, 95),
        "max": ordered[-1],
    }


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["X-API-Key"] = API_KEY
    return headers


async def _one(client: httpx.AsyncClient, question: str, session: str) -> Tuple[float, bool, str]:
    """发一次 chat 请求，返回 (耗时 ms, 是否错误, 简述)。"""
    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{BASE}/api/v1/chat",
            json={"question": question, "top_k": 8, "session_id": session},
            headers=_headers(),
            timeout=120,
        )
        elapsed = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        data = resp.json()
        return elapsed, False, "ok" if not data.get("refused") else "refused"
    except Exception as exc:  # noqa: BLE001 - 压测要统计错误，不能被单个失败打断
        return (time.perf_counter() - t0) * 1000, True, f"{type(exc).__name__}: {str(exc)[:60]}"


async def run_batch(client: httpx.AsyncClient, questions: List[str], concurrency: int) -> List[Tuple[float, bool, str]]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(q: str, i: int) -> Tuple[float, bool, str]:
        async with sem:
            # 每个请求独立会话，避免多轮历史互相干扰测量
            return await _one(client, q, f"load-{uuid.uuid4().hex[:8]}")

    return await asyncio.gather(*(guarded(q, i) for i, q in enumerate(questions)))


def summarize(label: str, results: List[Tuple[float, bool, str]]) -> None:
    ok = [r[0] for r in results if not r[1]]
    errors = [r for r in results if r[1]]
    s = _stats(results and [r[0] for r in results] or [0.0])
    ok_s = _stats(ok) if ok else {"p50": 0, "p95": 0, "max": 0}
    kinds: Dict[str, int] = {}
    for _, _, note in results:
        kinds[note.split(":")[0]] = kinds.get(note.split(":")[0], 0) + 1
    print(
        f"{label:<16} n={s['n']} | P50={s['p50']:7.0f}ms P95={s['p95']:7.0f}ms "
        f"max={s['max']:7.0f}ms | 错误={len(errors)} | {json.dumps(kinds, ensure_ascii=False)}"
    )
    if errors:
        for _, _, note in errors[:3]:
            print(f"    错误示例: {note}")


async def main_async(args: argparse.Namespace) -> int:
    base = args.url.rstrip("/")
    global BASE, API_KEY
    BASE = base
    if args.api_key:
        API_KEY = args.api_key

    # 混合并去重：重复题占一定比例，顺带验证缓存路径在并发下不串数据
    questions = []
    for i in range(args.total):
        if i % 3 == 2:
            questions.append(QUESTIONS[0])  # 每 3 个里 1 个重复题（缓存命中路径）
        else:
            questions.append(QUESTIONS[i % len(QUESTIONS)])

    async with httpx.AsyncClient() as client:
        health = await client.get(f"{BASE}/api/v1/health", timeout=10)
        health.raise_for_status()
        print(f"服务就绪：{base} | 压测 {args.total} 个请求（含重复题与负样本）\n")

        print("== 串行基线 ==")
        t0 = time.perf_counter()
        serial = await run_batch(client, questions, concurrency=1)
        summarize(f"串行(concurrency=1)", serial)
        print(f"  串行总耗时: {time.perf_counter() - t0:.1f}s\n")

        for conc in args.concurrency:
            print(f"== 并发 {conc} ==")
            t0 = time.perf_counter()
            results = await run_batch(client, questions, concurrency=conc)
            summarize(f"并发(concurrency={conc})", results)
            print(f"  并发总耗时: {time.perf_counter() - t0:.1f}s\n")

    print("判定：错误必须为 0；并发 P95 显著高于串行属预期（LLM 外部限流+线程池排队），")
    print("      但 P50 不应劣化到串行的数倍以上——那说明存在串行化瓶颈而非竞争。")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=BASE)
    parser.add_argument("--total", type=int, default=12, help="每轮总请求数")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[5, 10], help="并发梯度")
    parser.add_argument("--api-key", default="", help="服务端启用了 API_KEY 时传入")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
