"""会话历史存储：给主问答链路 /chat 补上多轮能力。

为什么要单独一个类
------------------
LangGraph Agent 那条链路早就有会话（session_id + SqliteSaver checkpoint），
但主问答链路 /chat 的 `session_id` 一直是个**预留字段**——声明了、没人读。
结果是同一个知识库，Agent 能接住「那这个车呢」，/chat 完全接不住。

设计取舍（刻意保持简单）：

- **内存存储，进程重启即失**。这是问答的「上下文」，不是业务数据；
  要持久化的是用户的选择，不是系统该擅自做的决定。真要持久化时，
  换成 SQLite 的成本只是把两个方法换成 SQL。
- **每会话最多保留 N 轮**（`SESSION_MAX_TURNS`，默认 6）。历史越长，
  改写质量越差（旧话题干扰新问题），token 也越贵。旧轮次直接丢，不摘要
  —— 摘要需要额外一次 LLM 调用，换来的收益无法验证。
- **线程安全**：FastAPI 的并发请求可能同时读写同一个 session。
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Deque, Dict, List, Optional


class ConversationStore:
    """按 session_id 保存最近 N 轮问答（先问后答，每轮一条 dict）。"""

    def __init__(self, max_turns: int = 6, max_sessions: int = 512) -> None:
        # max_turns 至少保留 1 轮，否则多轮能力等于没开
        self._max_turns = max(1, int(max_turns))
        # 会话数也要有上限：这是常驻进程里的内存结构，
        # 没有上限的话每个新 session_id 都会占一份历史，慢泄漏。
        self._max_sessions = max(1, int(max_sessions))
        self._sessions: Dict[str, Deque[Dict[str, str]]] = {}
        self._lock = threading.Lock()

    @property
    def max_turns(self) -> int:
        return self._max_turns

    def history(self, session_id: str) -> List[Dict[str, str]]:
        """返回该会话的历史轮次（从旧到新）。没有会话时返回空列表。"""
        if not session_id:
            return []
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def append(
        self,
        session_id: str,
        question: str,
        answer: str,
        refused: bool = False,
    ) -> List[Dict[str, str]]:
        """追加一轮问答，返回追加后的完整历史（从旧到新）。

        `refused` 标记该轮是否被拒答。它不是可有可无的元数据：
        查询改写要用它区分「上一轮答了 A」和「上一轮根本没答上」——
        拒答轮的话题往往是用户随手试的无关问题，追问「那正常保养呢？」
        是在换话题，不该把那个话题拼进改写结果（实测踩过）。
        """
        if not session_id:
            return []
        turn = {
            "question": (question or "").strip(),
            "answer": (answer or "").strip(),
            "refused": bool(refused),
        }
        with self._lock:
            if (
                session_id not in self._sessions
                and len(self._sessions) >= self._max_sessions
            ):
                # 淘汰最早创建的会话（dict 保插入序）。被淘汰的会话如果还在被用，
                # 下一轮会重新创建、只是丢了旧历史 —— 比进程 OOM 好得多。
                oldest = next(iter(self._sessions))
                del self._sessions[oldest]
            turns = self._sessions.setdefault(session_id, deque(maxlen=self._max_turns))
            turns.append(turn)
            return list(turns)

    def clear(self, session_id: str) -> int:
        """清掉一个会话，返回清掉的轮数。"""
        if not session_id:
            return 0
        with self._lock:
            turns = self._sessions.pop(session_id, None)
            return len(turns) if turns else 0

    def session_ids(self) -> List[str]:
        with self._lock:
            return list(self._sessions)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)


__all__ = ["ConversationStore"]
