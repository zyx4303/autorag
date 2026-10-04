"""检索层：向量检索 + BM25 关键词检索 + 融合/重排。

支持三种模式（RETRIEVAL_MODE 配置）：
- vector : 只看语义相似度
- keyword: 只看 BM25
- hybrid : 两路召回后用 RRF 融合（默认，推荐）

融合后再做一层轻量启发式重排：按查询词覆盖率与整句命中加权，
把"真正包含问题关键词"的片段顶上来，缓解小模型 embedding 的漂移。
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.config import Settings
from app.core.bm25 import BM25Index
from app.core.text_utils import make_snippet, tokenize
from app.core.vector_store import VectorStore
from app.logging_conf import get_logger
from app.models import RetrievedChunk

logger = get_logger(__name__)

VALID_MODES = {"vector", "keyword", "hybrid"}

# 单个中文字符在"查询词覆盖率"里的权重。
# 与 app/core/bm25.py 的 SINGLE_CJK_WEIGHT 保持同一原则：
# 中文按单字+bigram 切分时，单字命中几乎不构成"相关"的证据，必须降权。
SINGLE_CJK_WEIGHT = 0.25
_CJK_CHAR_RE = re.compile(r"[\u4e00-\u9fff]")

__all__ = ["VALID_MODES", "HybridRetriever"]


def _normalize_mode(mode: Optional[str], default: str) -> str:
    if mode and mode.lower() in VALID_MODES:
        return mode.lower()
    return default


def _base_record(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "chunk_id": str(item.get("chunk_id", "")),
        "doc_id": str(item.get("doc_id", "")),
        "source": str(item.get("source", "")),
        "title": str(item.get("title", "")),
        "section": str(item.get("section", "")),
        "position": int(item.get("position", 0) or 0),
        "text": str(item.get("text", "")),
    }


class HybridRetriever:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        bm25_index: BM25Index,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._bm25 = bm25_index

    # ------------------------------------------------------------------ #
    # 单路召回
    # ------------------------------------------------------------------ #
    async def _vector_recall(self, query: str, candidate_k: int) -> List[Tuple[Dict[str, Any], float]]:
        try:
            items = await self._vector_store.query(query, top_k=candidate_k)
        except Exception as exc:  # 向量检索失败不应让整个请求 500
            logger.error("向量检索失败：%s", exc)
            return []
        return [(item, float(item.get("vector_score", 0.0))) for item in items]

    def _keyword_recall(self, query: str, candidate_k: int) -> List[Tuple[Dict[str, Any], float]]:
        return self._bm25.search(query, top_k=candidate_k)

    # ------------------------------------------------------------------ #
    # 融合与重排
    # ------------------------------------------------------------------ #
    @staticmethod
    def _weighted_token_coverage(query_tokens: Sequence[str], text: str) -> float:
        """查询词覆盖率：按覆盖率算，但单个中文字符只计 0.25 权重。

        设计取舍：这里刻意不做 IDF 加权（可解释、无额外索引依赖），
        但**必须**给单字降权。实测教训：问 "车辆失火是什么原因" 时，
        保修条款片段因为同时包含"车/辆/失/火/原/因"这些单字而拿到满分覆盖率，
        把真正写着 "P0300 | 检测到随机或多缸失火" 的故障码片段挤掉；
        给单字降到 0.25 后，需要凑齐多个单字才能抵得上一句有实义的长词。
        """
        weights = [
            SINGLE_CJK_WEIGHT if len(token) == 1 and _CJK_CHAR_RE.match(token) else 1.0
            for token in query_tokens
            if token
        ]
        total = sum(weights)
        if total <= 0:
            return 0.0
        text_tokens = set(tokenize(text))
        hit = sum(
            weight
            for token, weight in zip(
                (token for token in query_tokens if token), weights
            )
            if token in text_tokens
        )
        return hit / total

    @staticmethod
    def _is_precision_token(token: str) -> bool:
        """是否属于"高精度 token"：含数字或字母的短词，如 P0420 / 5w / 3000 / obd2。

        为什么单独拎出来：这类串几乎不会同义改写，出现即强相关。
        实测多部分提问 "P0300 是什么意思？P0420 呢？" 里，
        两个故障码各占查询 token 的 1/5，普通词覆盖率会被大量虚词拉平，
        导致含 P0420 的片段反而排不进去；给这类 token 单独加权就能纠正。
        """
        core = token.replace("_", "")
        if not core or len(core) > 12:
            return False
        return any(char.isdigit() or ("a" <= char <= "z") for char in core)

    @staticmethod
    def _cjk_bigrams(text: str) -> Set[str]:
        """取文本中的"连续两个中文字符"集合。

        为什么需要：分词是按整串生成 bigram 的，问句 "车辆失火是什么原因" 只会产出
        "辆失""失火""火是" 这类**跨词**组合，而文档里是 "气缸失火"（bigram 为 "缸失""失火"）。
        两边只有在词边界对齐时才会共享 token，于是 "失火" 这种真正的关键词反而匹配不上。
        实测现象：问"车辆失火是什么原因"时，P0300 定义所在的故障码表完全排不进前 3。
        这里补一层"字符相邻即算命中"的兜底，专门救这类中文词边界不一致的情况。
        """
        flattened = "".join(char for char in text if "\u4e00" <= char <= "\u9fff")
        if len(flattened) < 2:
            return set()
        return {flattened[index : index + 2] for index in range(len(flattened) - 1)}

    def _rerank(self, candidates: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        query_tokens = tokenize(query)
        flat_query = " ".join(query.split())
        token_set = {token for token in query_tokens if token}
        precision_tokens = {token for token in token_set if self._is_precision_token(token)}

        # 查询中的中文词（长度 >= 2）转成字符 bigram，用于跨词边界的兜底匹配
        query_cjk_terms = {token for token in token_set if len(token) >= 2 and token.isalpha() and not token.isascii()}
        query_bigrams: Set[str] = set()
        for term in query_cjk_terms:
            query_bigrams |= self._cjk_bigrams(term)

        for item in candidates:
            text = str(item.get("text", ""))
            coverage = self._weighted_token_coverage(query_tokens, text)
            phrase_bonus = 0.0
            if flat_query and flat_query in " ".join(text.split()):
                phrase_bonus = 1.0
            elif len(query) >= 4 and query.strip() in text:
                phrase_bonus = 1.0

            # 高精度 token 命中比例：P0420 这类串命中一条就该被明显提权
            text_tokens = set(tokenize(text))
            if precision_tokens:
                precision_hits = len(precision_tokens & text_tokens) / len(precision_tokens)
            else:
                precision_hits = 0.0

            # 中文词组兜底：查询里的中文词若以连续字符形式出现在片段中，也算命中
            if query_bigrams and query_cjk_terms:
                text_bigrams = self._cjk_bigrams(text)
                matched_terms = 0
                for term in query_cjk_terms:
                    term_pairs = {
                        term[index : index + 2] for index in range(len(term) - 1)
                    }
                    if term_pairs and term_pairs.issubset(text_bigrams):
                        matched_terms += 1
                cjk_hits = matched_terms / len(query_cjk_terms)
            else:
                cjk_hits = 0.0

            # 章节标题命中：片段所属的章节路径（如 "P0 开头通用故障码 > 点火或气缸失火"）
            # 是切分时写进 metadata 的，它比正文更凝练，命中即为强相关信号。
            # 实测价值：问"车辆失火是什么原因"时，正确片段的章节含"失火"，
            # 而干扰它的保修条款章节不含，这个信号能把两者分开。
            section_text = str(item.get("section", ""))
            if section_text:
                section_coverage = self._weighted_token_coverage(query_tokens, section_text)
            else:
                section_coverage = 0.0

            item["key_coverage"] = round(coverage, 4)
            item["precision_hits"] = round(precision_hits, 4)
            item["cjk_hits"] = round(cjk_hits, 4)
            item["section_hits"] = round(section_coverage, 4)
            # 权重含义：0.54 融合分，0.12 词覆盖，0.12 高精度串，0.08 中文词组，
            #          0.08 章节命中，0.06 整句命中
            item["score"] = round(
                0.54 * float(item.get("fusion_score", 0.0))
                + 0.12 * coverage
                + 0.12 * precision_hits
                + 0.08 * cjk_hits
                + 0.08 * section_coverage
                + 0.06 * phrase_bonus,
                6,
            )
        return sorted(candidates, key=lambda item: item["score"], reverse=True)

    def _fuse(
        self,
        vector_hits: List[Tuple[Dict[str, Any], float]],
        keyword_hits: List[Tuple[Dict[str, Any], float]],
        mode: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        settings = self._settings
        merged: Dict[str, Dict[str, Any]] = {}

        def ensure(item: Dict[str, Any]) -> Dict[str, Any]:
            chunk_id = str(item.get("chunk_id", ""))
            if chunk_id not in merged:
                record = _base_record(item)
                record.update(
                    {
                        "vector_score": None,
                        "keyword_score": None,
                        "from_vector": False,
                        "from_keyword": False,
                        "fusion_score": 0.0,
                    }
                )
                merged[chunk_id] = record
            return merged[chunk_id]

        if mode in {"vector", "hybrid"}:
            # vector 模式下 fusion_score 直接用余弦相似度（可解释、可用于阈值判断）。
            # hybrid 模式下**必须归一化**：余弦相似度是 0~1 量级，而"名次分"是 1/(k+rank)
            # 的 0.01 量级，两者直接相加会让向量通道的绝对值压过名次信息。
            # 实测教训：命名写的是"归一化"，但 vector 分支漏了归一化，
            # 导致 BM25 排第 1 的正确答案（名次分仅 0.016）被名次靠后的片段靠余弦值盖过。
            max_vector = max((score for _, score in vector_hits), default=0.0)
            for rank, (item, score) in enumerate(vector_hits, start=1):
                record = ensure(item)
                record["vector_score"] = float(score)
                record["from_vector"] = True
                if mode == "hybrid":
                    normalized = (float(score) / max_vector) if max_vector > 0 else 0.0
                    record["fusion_score"] += settings.vector_weight * (
                        normalized / (settings.rrf_k + rank)
                    )
                else:
                    record["fusion_score"] = float(score)

        if mode in {"keyword", "hybrid"}:
            # hybrid 模式同样是"归一化名次分"，天然屏蔽两路分数量纲差异；
            # keyword 模式则按本轮最高 BM25 分数归一化，便于统一展示与阈值判断。
            max_keyword = max((score for _, score in keyword_hits), default=0.0)
            for rank, (item, score) in enumerate(keyword_hits, start=1):
                record = ensure(item)
                record["keyword_score"] = float(score)
                record["from_keyword"] = True
                if mode == "hybrid":
                    normalized = (float(score) / max_keyword) if max_keyword > 0 else 0.0
                    record["fusion_score"] += settings.keyword_weight * (
                        normalized / (settings.rrf_k + rank)
                    )
                else:
                    normalized = (float(score) / max_keyword) if max_keyword > 0 else 0.0
                    record["fusion_score"] = normalized

        return self._rerank(list(merged.values()), query)

    # ------------------------------------------------------------------ #
    # 对外入口
    # ------------------------------------------------------------------ #
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> Tuple[List[RetrievedChunk], str]:
        """执行检索，返回 (片段列表, 实际使用的模式)。"""
        query = (query or "").strip()
        if not query:
            return [], _normalize_mode(mode, self._settings.retrieval_mode)

        effective_mode = _normalize_mode(mode, self._settings.retrieval_mode)
        final_k = max(1, int(top_k or self._settings.top_k))
        candidate_k = max(final_k, int(self._settings.candidate_k))

        vector_hits: List[Tuple[Dict[str, Any], float]] = []
        keyword_hits: List[Tuple[Dict[str, Any], float]] = []

        if effective_mode in {"vector", "hybrid"}:
            vector_hits = await self._vector_recall(query, candidate_k)
        if effective_mode in {"keyword", "hybrid"}:
            # BM25 是纯 CPU 计算，放到线程池避免在大库上阻塞事件循环
            keyword_hits = await asyncio.to_thread(self._keyword_recall, query, candidate_k)

        fused = self._fuse(vector_hits, keyword_hits, effective_mode, query)

        results: List[RetrievedChunk] = []
        for item in fused[:final_k]:
            if float(item.get("score", 0.0)) <= 0:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    source=item["source"],
                    title=item["title"],
                    section=item["section"],
                    position=item["position"],
                    text=item["text"],
                    score=float(item["score"]),
                    vector_score=item.get("vector_score"),
                    keyword_score=item.get("keyword_score"),
                    from_vector=bool(item.get("from_vector")),
                    from_keyword=bool(item.get("from_keyword")),
                )
            )
        return results, effective_mode

    # ------------------------------------------------------------------ #
    # 调试视图
    # ------------------------------------------------------------------ #
    @staticmethod
    def to_debug(results: Sequence[RetrievedChunk], snippet_chars: int = 160) -> List[Dict[str, Any]]:
        return [
            {
                "rank": index + 1,
                "chunk_id": item.chunk_id,
                "source": item.source,
                "title": item.title,
                "section": item.section,
                "position": item.position,
                "score": round(item.score, 6),
                "vector_score": (
                    round(item.vector_score, 6) if item.vector_score is not None else None
                ),
                "keyword_score": (
                    round(item.keyword_score, 4) if item.keyword_score is not None else None
                ),
                "from_vector": item.from_vector,
                "from_keyword": item.from_keyword,
                "snippet": make_snippet(item.text, snippet_chars),
            }
            for index, item in enumerate(results)
        ]
