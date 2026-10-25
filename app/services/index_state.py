"""索引版本状态：让查询缓存知道「索引什么时候变了」。

为什么需要它
------------
查询缓存的正确性依赖一个前提：**同样的索引 + 同样的问题 = 同样的回答**。
一旦索引变了（入库新文档、重建），缓存里的旧回答就成了脏数据。

与其在缓存里放过期时间赌索引没变，不如让入库方**显式宣布**「索引变了」：
一个进程内的版本计数器，每次入库成功后自增。缓存键里带版本号，
版本一变，旧键自然全部失效——不需要遍历清理，逻辑上是零成本的。

刻意不做的事：把版本持久化到磁盘。进程重启后缓存本来就是空的
（内存缓存），持久化版本号只会引入「磁盘版本 vs 内存索引」一致性的新问题。
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class IndexState:
    """进程内索引版本计数（线程安全）。"""

    def __init__(self) -> None:
        self._version = 1
        self._lock = threading.Lock()

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def bump(self, reason: str = "") -> int:
        """索引发生变化后调用；返回新版本号。reason 只用于日志。"""
        with self._lock:
            self._version += 1
            version = self._version
        if reason:
            logger.info("索引版本 → %d（%s）", version, reason)
        return version


__all__ = ["IndexState"]
