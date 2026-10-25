"""事件循环阻塞探针：测量入库期间服务对其他请求的响应能力。

用法::

    python scripts/loop_block_probe.py [--url http://127.0.0.1:8000] [--rebuild]

原理：以 50ms 间隔高频请求 /health，同时提交一个异步入库任务。
事件循环被同步 CPU 段卡住时，健康检查的响应时间会出现尖峰——
「P95 / 最大耗时」就是阻塞程度的直接度量。

为什么需要它：入库代码里任何一段同步阻塞（文件解析、切分、Chroma 写、
BM25 建索引）都会让事件循环停转，**所有**并发请求跟着卡住。
2026-10-25 的线程级卸载（README 12.20）就是靠它验证的：
重建期间最大尖峰 103ms → 60ms。以后再动入库链路，跑一遍这个脚本
就能知道有没有把阻塞带回来。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
SAMPLE_INTERVAL_S = 0.05
MAX_WAIT_JOB_S = 120


def _get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _post_json(url: str, payload: dict, timeout: float = 10) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=BASE, help="服务基地址")
    parser.add_argument("--rebuild", action="store_true", help="任务带 rebuild=true（全量重建，阻塞更明显）")
    parser.add_argument("--duration", type=float, default=60.0, help="采样窗口上限（秒）")
    args = parser.parse_args()
    base = args.url.rstrip("/")

    samples: list = []
    submitted = False
    job_id: str = ""
    deadline = time.time() + args.duration

    print(f"探针启动：{base}（rebuild={args.rebuild}）")
    while time.time() < deadline:
        t0 = time.perf_counter()
        try:
            _get_json(f"{base}/api/v1/health")
            dt_ms = (time.perf_counter() - t0) * 1000
        except Exception:
            dt_ms = 9999.0  # 连不上按最差算（循环完全卡死/服务挂了）
        samples.append(dt_ms)

        if not submitted:
            job = _post_json(
                f"{base}/api/v1/ingest",
                {"async_mode": True, "rebuild": args.rebuild},
            )
            job_id = job["job_id"]
            submitted = True
            print(f"已提交入库任务 {job_id}，开始采样")

        if submitted:
            status = _get_json(f"{base}/api/v1/ingest/status/{job_id}")["status"]
            if status in ("done", "failed"):
                # 任务结束后再采 0.5s 作为「空闲基线」，然后收尾
                time.sleep(0.5)
                t0 = time.perf_counter()
                _get_json(f"{base}/api/v1/health")
                idle_ms = (time.perf_counter() - t0) * 1000
                print(f"任务结束状态: {status} | 空闲时健康检查: {idle_ms:.0f}ms")
                break
        time.sleep(SAMPLE_INTERVAL_S)
    else:
        print("[警告] 采样窗口内任务未结束")

    samples.sort()
    n = len(samples)
    if n == 0:
        print("没有采样数据")
        return 1
    print(
        f"采样 {n} 次 | 中位 {samples[n // 2]:.0f}ms | "
        f"P95 {samples[int(n * 0.95)]:.0f}ms | 最大 {samples[-1]:.0f}ms"
    )
    print("判定：中位应为个位数毫秒；P95/最大尖峰 = 事件循环被同步段卡住的时长")
    return 0


if __name__ == "__main__":
    sys.exit(main())
