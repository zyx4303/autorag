"""请求指标采集：延迟分位数 / token 消耗 / 缓存命中率 / 拒答率。

为什么需要它
------------
容量规划和成本核算都需要「真实流量长什么样」的数据：
- 延迟只看均值会被长尾骗了，运营真正关心的是 P95/P99；
- token 消耗是 RAG 服务最主要的可变成本，得有数才知道单位成本；
- 缓存命中率直接决定「上了缓存到底省了多少钱」——没这个数，
  缓存就只是个技术特性。

设计取舍：

- **进程内存 + 有界环形缓冲**（默认 1000 条）。指标是观测数据，
  丢历史不要紧；进程重启归零也是可接受的（不持久化）。
  上 Prometheus 的前提是多实例/长期监控，到时候把 snapshot() 换成
  暴露 counter/gauge 即可，record() 的调用点不用动。
- **分位数现算**（排序取下标）。1000 条排序是微秒级，不值得维护堆。
- **成本估算默认关闭**：价格是会变的商业数据，硬编码进代码必然过期；
  配了单价（每百万 token）才算，没配就只报 token 数。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from app.logging_conf import get_logger

logger = get_logger(__name__)


def _percentile(sorted_values, pct: float) -> Optional[float]:
    """排序数组的百分位数（最近邻下标法）。空数组返回 None。"""
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, round(pct / 100 * (len(sorted_values) - 1))))
    return sorted_values[index]


class RequestMetricsCollector:
    """进程内请求指标采集（线程安全，有界缓冲）。"""

    def __init__(self, max_records: int = 1000) -> None:
        self._max_records = max(10, int(max_records))
        self._records: Deque[Dict[str, Any]] = deque(maxlen=self._max_records)
        self._lock = threading.Lock()
        self._dropped = 0  # 被环形缓冲挤掉的记录数（观测窗口大于缓冲时的提示）

    def record(
        self,
        endpoint: str,
        total_ms: Optional[int] = None,
        retrieve_ms: Optional[int] = None,
        generate_ms: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        cache_hit: bool = False,
        refused: bool = False,
        error: bool = False,
    ) -> None:
        """记录一次请求。字段缺省就缺省——采集本身绝不能影响主流程。"""
        entry = {
            "ts": time.time(),
            "endpoint": endpoint,
            "total_ms": total_ms,
            "retrieve_ms": retrieve_ms,
            "generate_ms": generate_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_hit": bool(cache_hit),
            "refused": bool(refused),
            "error": bool(error),
        }
        with self._lock:
            if len(self._records) == self._records.maxlen:
                self._dropped += 1
            self._records.append(entry)

    def snapshot(
        self,
        price_prompt_per_m: Optional[float] = None,
        price_completion_per_m: Optional[float] = None,
    ) -> Dict[str, Any]:
        """聚合当前观测窗口。

        `price_*_per_m`：每百万 token 单价（元）。两个都配置时才输出成本估算，
        并且结果带 estimated 前缀语义（调用方在文档里说明）——价格会变，
        这个数只能用于量级判断。
        """
        with self._lock:
            records = list(self._records)

        if not records:
            return {
                "count": 0,
                "note": "当前窗口没有请求记录（进程重启后归零）",
                "dropped_by_buffer": self._dropped,
            }

        def _by_endpoint(name: str) -> list:
            return [r for r in records if r["endpoint"] == name]

        def _latency_block(rows: list) -> Dict[str, Any]:
            totals = sorted(r["total_ms"] for r in rows if r["total_ms"] is not None)
            return {
                "count": len(rows),
                "p50_ms": _percentile(totals, 50),
                "p95_ms": _percentile(totals, 95),
                "p99_ms": _percentile(totals, 99),
                "max_ms": totals[-1] if totals else None,
            }

        endpoints = {
            name: _latency_block(_by_endpoint(name))
            for name in sorted({r["endpoint"] for r in records})
        }

        prompt = sum(r["prompt_tokens"] or 0 for r in records)
        completion = sum(r["completion_tokens"] or 0 for r in records)

        cache_hits = sum(1 for r in records if r["cache_hit"])
        refusals = sum(1 for r in records if r["refused"])
        errors = sum(1 for r in records if r["error"])

        snapshot = {
            "count": len(records),
            "window": {
                "since": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(records[0]["ts"])),
                "until": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(records[-1]["ts"])),
            },
            "latency_overall": _latency_block(records),
            "latency_by_endpoint": endpoints,
            "tokens": {
                "prompt": prompt,
                "completion": completion,
                "total": prompt + completion,
                # 有 token 记录的请求数：缓存命中/拒答/流式的请求没有 usage，
                # 分母不写清楚，均值就会被误读
                "requests_with_usage": sum(
                    1 for r in records if r["prompt_tokens"] or r["completion_tokens"]
                ),
            },
            "cache_hit_rate": round(cache_hits / len(records), 4) if records else None,
            "refusal_rate": round(refusals / len(records), 4) if records else None,
            "error_rate": round(errors / len(records), 4) if records else None,
            "dropped_by_buffer": self._dropped,
        }

        if price_prompt_per_m and price_completion_per_m:
            snapshot["estimated_cost"] = {
                "currency": "CNY",
                "price_prompt_per_m": price_prompt_per_m,
                "price_completion_per_m": price_completion_per_m,
                "total": round(
                    prompt / 1e6 * price_prompt_per_m
                    + completion / 1e6 * price_completion_per_m,
                    4,
                ),
                "note": "按配置单价估算，仅供量级判断；单价会随供应商调整",
            }
        return snapshot


__all__ = ["RequestMetricsCollector"]
