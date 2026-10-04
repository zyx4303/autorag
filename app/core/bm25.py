"""BM25 关键词检索（纯 Python 实现，零第三方依赖）。

为什么需要关键词通道：
- 售后场景里"故障码 P0420""5W-30""2 万公里"这类精确串，向量检索容易漂；
- 与向量检索做 RRF 融合后，能显著减少"专业名词被同义改写"的漏召回。

索引内容：从向量库导出的全量片段（chunk_id + text + 来源元数据），
以 JSON 落盘到 BM25_INDEX_PATH，重启后可直接加载，无需重新分词。
"""
from __future__ import annotations

import json
import math
import re
import threading
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.core.text_utils import tokenize
from app.logging_conf import get_logger

logger = get_logger(__name__)

BM25_K1 = 1.5
BM25_B = 0.75
# 关键词通道的相对分数下限：低于最高分这个比例的候选会被丢弃。
# 取值理由：中文单字+bigram 切分下，无关查询也能拿到很小的分数，
# 0.15 能滤掉这类噪声，又不会误伤真正的次优候选（它们的分数通常 ≥ 最高分的 1/3）。
MIN_RELATIVE_SCORE = 0.15
# "查询词组整体命中"的加成系数（1.3 = 该片段最终 BM25 分乘以 1.3）。
# 为什么需要：中文按单字+bigram 切分时，疑问句里的"是/什/么/意/思"会各贡献一份分数，
# 实测 "P0300 是什么意思" 里真正有信息量的 P0300 片段只排第 7，
# 前面几个片段全是靠虚词单字蹭分。词组连读命中是强证据（"P0300"、"更换机油"
# 这种连续串同时出现），用它把有实义的片段提上来，比给虚词加停用词表更通用。
PHRASE_BOOST = 1.3
# 单个中文字符的权重。中文按单字+bigram 切分时，"是/什/么/意/思"这类虚词单字
# 每个都会贡献一份 IDF 分数，实测能把毫无关系的保修条款片段顶到第 1 名，
# 而真正写着 "P0300 | 引擎曾经有失火现象" 的片段排到第 5。
# 单字区分度天然低于双字词与 ASCII 词，因此按 0.3 降权：
# 需要多个单字同时命中才能抵得上一个有实义的双字词或故障码。
SINGLE_CJK_WEIGHT = 0.3
# 这类纯疑问/客套短语几乎出现在所有语料里，不具备区分度，不参与"词组命中"加成
_STOP_PHRASES = {
    "是什么意思",
    "什么意思",
    "是什么",
    "怎么办",
    "怎么处理",
    "为什么",
    "如何",
    "请教",
    "请问",
    "谢谢",
    "麻烦",
}
# 用于识别"单字"的 CJK 正则
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")


class BM25Index:
    """内存索引 + JSON 持久化；写操作加锁，读操作无锁（单进程 asyncio 场景足够）。"""

    def __init__(self, persist_path: Optional[Path] = None) -> None:
        self._persist_path = persist_path
        self._lock = threading.Lock()
        self._chunks: List[Dict[str, Any]] = []
        self._token_counts: List[Counter] = []
        self._lengths: List[int] = []
        self._doc_freq: Counter = Counter()
        self._avg_len: float = 0.0
        self._idf: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # 状态
    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        return len(self._chunks)

    # ------------------------------------------------------------------ #
    # 构建与持久化
    # ------------------------------------------------------------------ #
    def build(self, chunks: Sequence[Dict[str, Any]]) -> int:
        """用全量片段重建索引。chunks 元素需含 chunk_id/text/source/title/section/position。"""
        with self._lock:
            self._chunks = []
            self._token_counts = []
            self._lengths = []
            self._doc_freq = Counter()

            for item in chunks:
                text = str(item.get("text") or "")
                tokens = tokenize(text)
                if not tokens:
                    continue
                counts = Counter(tokens)
                self._chunks.append(
                    {
                        "chunk_id": str(item.get("chunk_id", "")),
                        "text": text,
                        "source": str(item.get("source", "")),
                        "title": str(item.get("title", "")),
                        "section": str(item.get("section", "")),
                        "position": int(item.get("position", 0) or 0),
                        "doc_id": str(item.get("doc_id", "")),
                    }
                )
                self._token_counts.append(counts)
                self._lengths.append(len(tokens))
                for token in counts:
                    self._doc_freq[token] += 1

            self._recompute_idf()
            logger.info("BM25 索引构建完成：%d 个片段，词典 %d 个词", self.size, len(self._idf))
        self.save()
        return self.size

    def _recompute_idf(self) -> None:
        total = len(self._chunks)
        if total == 0:
            self._avg_len = 0.0
            self._idf = {}
            return
        self._avg_len = sum(self._lengths) / total
        # BM25 常用平滑 IDF，保证 idf > 0，避免高频词产生负分
        self._idf = {
            token: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for token, freq in self._doc_freq.items()
        }

    def clear(self) -> None:
        with self._lock:
            self._chunks = []
            self._token_counts = []
            self._lengths = []
            self._doc_freq = Counter()
            self._avg_len = 0.0
            self._idf = {}
        self.save()

    def save(self) -> None:
        if not self._persist_path:
            return
        payload = {
            "version": 1,
            "k1": BM25_K1,
            "b": BM25_B,
            "size": self.size,
            "chunks": self._chunks,
            "lengths": self._lengths,
        }
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps(payload, ensure_ascii=False), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover - 磁盘问题
            logger.warning("BM25 索引落盘失败：%s", exc)

    def load(self) -> bool:
        """从磁盘加载；文件不存在或损坏时返回 False（调用方会重建）。"""
        if not self._persist_path or not self._persist_path.exists():
            return False
        try:
            payload = json.loads(self._persist_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("BM25 索引读取失败，将重建：%s", exc)
            return False

        chunks = payload.get("chunks") or []
        if not isinstance(chunks, list) or not chunks:
            return False

        with self._lock:
            self._chunks = []
            self._token_counts = []
            self._lengths = []
            self._doc_freq = Counter()
            for item in chunks:
                text = str(item.get("text") or "")
                tokens = tokenize(text)
                if not tokens:
                    continue
                self._chunks.append(item)
                self._token_counts.append(Counter(tokens))
                self._lengths.append(len(tokens))
                for token in set(tokens):
                    self._doc_freq[token] += 1
            self._recompute_idf()
        logger.info("BM25 索引已从磁盘加载：%d 个片段", self.size)
        return self.size > 0

    # ------------------------------------------------------------------ #
    # 检索
    # ------------------------------------------------------------------ #
    def search(self, query: str, top_k: int) -> List[Tuple[Dict[str, Any], float]]:
        """返回 [(chunk, bm25_score)]，按分数降序。

        两步降噪（都是为了对抗中文单字+bigram 切分带来的虚词噪声）：
        1. 查询被空格/标点分开后，每一段如果整体出现在片段里，给该片段乘 PHRASE_BOOST；
           例："P0300 是什么意思" → 词组 ["P0300", "是什么意思"]，
           含 "P0300" 连读的故障码片段会被提上来。
        2. 丢弃分数远低于最高分的候选（< top_score * MIN_RELATIVE_SCORE）。
        """
        if top_k <= 0 or not self._chunks:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        query_phrases = self._query_phrases(query)

        scored: List[Tuple[int, float]] = []
        for index, counts in enumerate(self._token_counts):
            length = self._lengths[index] or 1
            score = 0.0
            for token in query_tokens:
                freq = counts.get(token)
                if not freq:
                    continue
                idf = self._idf.get(token, 0.0)
                if idf <= 0:
                    continue
                denominator = freq + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * (length / (self._avg_len or 1.0))
                )
                weight = SINGLE_CJK_WEIGHT if len(token) == 1 and _CJK_CHAR_RE.match(token) else 1.0
                score += weight * idf * (freq * (BM25_K1 + 1.0)) / denominator
            if score <= 0:
                continue
            phrase_hits = sum(
                1 for phrase in query_phrases if phrase in self._chunks[index]["text"]
            )
            if phrase_hits:
                score *= PHRASE_BOOST ** min(phrase_hits, 3)
            scored.append((index, score))

        if not scored:
            return []

        scored.sort(key=lambda pair: pair[1], reverse=True)
        cutoff = scored[0][1] * MIN_RELATIVE_SCORE
        kept = [(index, score) for index, score in scored[:top_k] if score >= cutoff]
        return [(self._chunks[index], score) for index, score in kept]

    @staticmethod
    def _query_phrases(query: str) -> List[str]:
        """把查询切成"词组"，用于判断是否整体命中。

        只保留长度 >= 2 的片段（单个字没有区分度），
        并跳过纯虚词片段，避免 "是什么意思" 这种整串在多个片段里都出现而失去意义。
        """
        parts = re.split(r"[\s,，。？?、；;：:！!]+", query.strip())
        phrases: List[str] = []
        for part in parts:
            part = part.strip()
            if len(part) < 2:
                continue
            if part.lower() in _STOP_PHRASES:
                continue
            phrases.append(part)
        return phrases

    def describe(self) -> Dict[str, Any]:
        return {
            "size": self.size,
            "vocabulary": len(self._idf),
            "avg_doc_length": round(self._avg_len, 2),
            "persist_path": str(self._persist_path) if self._persist_path else None,
        }
