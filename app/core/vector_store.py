"""Chroma 向量库封装（持久化客户端 + 显式 embedding）。

要点：
- 使用 PersistentClient 落盘到 CHROMA_DIR，重启不丢数据；
- 显式传入自己算好的向量（embedding_function 不交给 Chroma），
  这样 embedding 走哪家服务完全由 .env 决定，避免隐式下载默认模型；
- 距离度量使用 cosine，检索时换算成相似度 = 1 - distance；
- 写入使用 upsert，相同 chunk_id 覆盖，天然支持增量更新。
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.embedding import BaseEmbedder
from app.logging_conf import get_logger
from app.models import Chunk

logger = get_logger(__name__)


class VectorStoreError(RuntimeError):
    """向量库操作失败。"""


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str, embedder: BaseEmbedder) -> None:
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self._embedder = embedder

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "description": "AutoRAG 汽车售后知识库"},
        )
        logger.info(
            "向量库就绪 dir=%s collection=%s dim=%d count=%d",
            persist_dir,
            collection_name,
            embedder.dimension,
            self.count(),
        )

    # ------------------------------------------------------------------ #
    # 基础信息
    # ------------------------------------------------------------------ #
    @property
    def collection_name(self) -> str:
        return self._collection_name

    def count(self) -> int:
        try:
            return int(self._collection.count())
        except Exception as exc:  # pragma: no cover
            logger.warning("读取向量库数量失败：%s", exc)
            return 0

    def health(self) -> bool:
        try:
            self._collection.count()
            return True
        except Exception:  # pragma: no cover
            return False

    # ------------------------------------------------------------------ #
    # 写入
    # ------------------------------------------------------------------ #
    async def upsert_chunks(self, chunks: Sequence[Chunk]) -> int:
        """向量化并写入片段，返回写入条数。"""
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        vectors = await self._embedder.embed_documents(texts)
        if len(vectors) != len(chunks):
            raise VectorStoreError(
                f"向量数量({len(vectors)})与片段数量({len(chunks)})不一致，拒绝写入"
            )

        expected_dim = self._embedder.dimension
        if expected_dim and vectors and len(vectors[0]) != expected_dim:
            raise VectorStoreError(
                f"向量维度({len(vectors[0])})与 embedder 声明维度({expected_dim})不一致。"
                "如果刚换过 embedding 模型，请调用 "
                "POST /api/v1/ingest 并带 {\"rebuild\": true, \"reset_registry\": true} 重建向量库。"
            )

        metadatas: List[Dict[str, Any]] = []
        for chunk in chunks:
            metadatas.append(
                {
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "title": chunk.title,
                    "section": chunk.section,
                    "position": int(chunk.position),
                    "char_start": int(chunk.char_start),
                    "char_end": int(chunk.char_end),
                }
            )

        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        return len(chunks)

    async def delete_by_source_async(self, source: str) -> int:
        """delete_by_source 的异步版本（在线程池里执行，避免阻塞事件循环）。"""
        return await asyncio.to_thread(self.delete_by_source, source)

    def delete_by_source(self, source: str) -> int:
        """删除某个来源文件的所有片段，返回删除前该来源的片段数。

        注：这里不传 include 参数（走 Chroma 默认行为，只回 ids），
        因为部分 Chroma 客户端会对空数组 `include=[]` 做非空校验而报错。
        """
        existing = self._collection.get(where={"source": source})
        ids = existing.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def delete_doc(self, doc_id: str) -> int:
        existing = self._collection.get(where={"doc_id": doc_id})
        ids = existing.get("ids") or []
        if not ids:
            return 0
        self._collection.delete(ids=ids)
        return len(ids)

    def reset(self) -> int:
        """清空整个集合（重建时使用），返回清空前的条数。"""
        before = self.count()
        try:
            self._client.delete_collection(self._collection_name)
        except Exception as exc:  # 集合不存在时忽略
            logger.warning("删除集合失败（可能不存在）：%s", exc)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine", "description": "AutoRAG 汽车售后知识库"},
        )
        logger.info("已清空集合 %s，删除前条数=%d", self._collection_name, before)
        return before

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    async def query(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """向量检索，返回按相似度降序的结果列表。

        每项包含：chunk_id / text / source / title / section / position /
        vector_score(余弦相似度，越大越相似) / metadata。
        """
        # 说明：chromadb 的接口是同步阻塞的，这里用 to_thread 挪到线程池，
        # 避免在 asyncio 事件循环里做磁盘/索引 IO 阻塞其它请求。
        total = await asyncio.to_thread(self._collection.count)
        if top_k <= 0 or total == 0:
            return []

        vector = await self._embedder.embed_query(query_text)
        expected_dim = self._embedder.dimension
        if expected_dim and len(vector) != expected_dim:
            raise VectorStoreError(
                f"查询向量维度({len(vector)})与当前 embedder 维度({expected_dim})不一致。"
                "如果刚换过 embedding 模型，请重建向量库。"
            )

        raw = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[vector],
            n_results=min(top_k, max(1, total)),
            include=["documents", "metadatas", "distances"],
        )

        ids = (raw.get("ids") or [[]])[0]
        documents = (raw.get("documents") or [[]])[0]
        metadatas = (raw.get("metadatas") or [[]])[0]
        distances = (raw.get("distances") or [[]])[0]

        results: List[Dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            metadata = dict(metadata or {})
            distance = float(distances[index]) if index < len(distances) else 1.0
            # cosine distance ∈ [0, 2] -> 相似度 ∈ [-1, 1]
            similarity = 1.0 - distance
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": documents[index] if index < len(documents) else "",
                    "source": str(metadata.get("source", "")),
                    "title": str(metadata.get("title", "")),
                    "section": str(metadata.get("section", "")),
                    "position": int(metadata.get("position", 0) or 0),
                    "doc_id": str(metadata.get("doc_id", "")),
                    "vector_score": similarity,
                    "metadata": metadata,
                }
            )
        return results

    async def all_chunks_async(self) -> List[Dict[str, Any]]:
        """all_chunks 的异步版本（用于入库流程中重建 BM25 索引）。"""
        return await asyncio.to_thread(self.all_chunks)

    def all_chunks(self) -> List[Dict[str, Any]]:
        """取出全量片段（用于构建 BM25 索引或导出快照）。"""
        if self.count() == 0:
            return []
        raw = self._collection.get(include=["documents", "metadatas"])
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        metadatas = raw.get("metadatas") or []
        results: List[Dict[str, Any]] = []
        for index, chunk_id in enumerate(ids):
            metadata = dict((metadatas[index] if index < len(metadatas) else {}) or {})
            results.append(
                {
                    "chunk_id": chunk_id,
                    "text": documents[index] if index < len(documents) else "",
                    "source": str(metadata.get("source", "")),
                    "title": str(metadata.get("title", "")),
                    "section": str(metadata.get("section", "")),
                    "position": int(metadata.get("position", 0) or 0),
                    "doc_id": str(metadata.get("doc_id", "")),
                    "metadata": metadata,
                }
            )
        return results

    def sources(self) -> List[str]:
        """当前库中出现过的来源文件列表。

        这里显式传 include=["metadatas"] 而不是依赖 Chroma 的默认 include 行为，
        因为"默认是否返回 metadatas"在不同版本里不太一致；显式声明更稳。
        """
        try:
            raw = self._collection.get(include=["metadatas"])
        except Exception:  # pragma: no cover
            return []
        found = set()
        for metadata in raw.get("metadatas") or []:
            source = (metadata or {}).get("source")
            if source:
                found.add(str(source))
        return sorted(found)

    async def peek_async(self, limit: int = 5) -> List[Dict[str, Any]]:
        """peek 的异步版本（预览接口用，避免阻塞事件循环）。"""
        return await asyncio.to_thread(self.peek, limit)

    def peek(self, limit: int = 5) -> List[Dict[str, Any]]:
        """预览少量片段，用于 /stats 与人工抽查。"""
        try:
            raw = self._collection.get(limit=limit, include=["documents", "metadatas"])
        except Exception:  # pragma: no cover
            return []
        ids = raw.get("ids") or []
        documents = raw.get("documents") or []
        out = []
        for index, chunk_id in enumerate(ids):
            out.append(
                {
                    "chunk_id": chunk_id,
                    "text": (documents[index] or "")[:200],
                }
            )
        return out
