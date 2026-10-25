"""查询缓存：同样的问题直接返回上次的回答，省 token 省延迟。

L5 生产化的第一块。RAG 问答里最贵的是 LLM 调用（token + 秒级延迟），
而真实流量里「同一个问题被反复问」非常常见——文档不变时，
答案也不该变。缓存就是把这个不变性利用起来。

正确性边界（比命中率重要）：

1. **键里带索引版本**（IndexState）。索引一变，旧答案全是脏数据；
   版本号进了键，入库后旧键自然失效，不需要遍历清理。
2. **多轮请求不缓存**。答案依赖会话历史，同一问题在不同历史下答案不同。
3. **debug 请求不缓存**。它是评测/调试路径——评测要测的是系统本身，
   `--repeat` 测方差时如果第二轮全命中缓存，测出来的方差是假的。
4. **命中时延迟字段重算**。缓存里的旧延迟是上次请求的，
   直接透传会误导调用方；命中请求的延迟就是查缓存的时间。

实现刻意保持简单：进程内存 + TTL + LRU。
上 Redis 的前提是多实例部署，而这个项目目前是单进程——
先在正确的位置留好接口（get/put/clear/stats），换后端时调用方不用动。
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional


class QueryCache:
    """LRU + TTL 的内存查询缓存（线程安全）。"""

    def __init__(self, max_entries: int = 256, ttl_seconds: float = 3600.0) -> None:
        self._max_entries = max(1, int(max_entries))
        self._ttl = max(1.0, float(ttl_seconds))
        # OrderedDict 为了 LRU 淘汰：访问过的键移到末尾，满时淘汰头部
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                self._misses += 1
                return None
            stored_at, value = item
            if now - stored_at > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            # LRU：命中即视为最近使用
            self._store.move_to_end(key)
            self._hits += 1
            return value

    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.monotonic(), value)
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)

    def clear(self) -> int:
        """清空缓存，返回清掉的条数（运维/测试用）。"""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            return count

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "entries": len(self._store),
                "max_entries": self._max_entries,
                "ttl_seconds": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
            }


__all__ = ["QueryCache"]
