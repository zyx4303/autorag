"""入库服务：扫描 -> 增量判断 -> 切分 -> 向量化 -> 写入 Chroma -> 重建 BM25。

增量策略（用 data/registry.json 记录每个来源文件的 checksum）：
- 新增文件      -> 切分入库
- checksum 变化 -> 先按 source 删除旧片段，再重新入库
- checksum 未变 -> 跳过（可用 rebuild=true 强制全量重建）
- 磁盘已删除    -> 从向量库与 BM25 索引中清除
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.config import Settings
from app.core.bm25 import BM25Index
from app.core.chunker import chunk_document
from app.core.loader import UnsupportedFormatError, iter_document_files, load_document
from app.core.vector_store import VectorStore
from app.logging_conf import get_logger
from app.models import Chunk, IngestFileResult, IngestResponse, SourceDocument

logger = get_logger(__name__)


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        index_state: Optional[Any] = None,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._bm25 = bm25_index
        # 索引版本计数（可选）：入库成功后自增，让查询缓存立即失效。
        # 没装配时缓存键退化为恒定版本（缓存仍正确，只是不会因入库而提前失效）。
        self._index_state = index_state
        self._last_result: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ #
    # registry（增量记录）
    # ------------------------------------------------------------------ #
    def _load_registry(self) -> Dict[str, Any]:
        path = self._settings.registry_path
        if not path.exists():
            return {"version": 1, "documents": {}, "last_run": None}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("registry 读取失败，将视为空：%s", exc)
            return {"version": 1, "documents": {}, "last_run": None}
        if not isinstance(data, dict) or "documents" not in data:
            return {"version": 1, "documents": {}, "last_run": None}
        return data

    def _save_registry(self, registry: Dict[str, Any]) -> None:
        path = self._settings.registry_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError as exc:  # pragma: no cover
            logger.warning("registry 落盘失败：%s", exc)

    # ------------------------------------------------------------------ #
    # 索引维护
    # ------------------------------------------------------------------ #
    def rebuild_bm25(self) -> int:
        """用向量库当前全量片段重建 BM25 索引（同步版本，用于启动阶段）。"""
        chunks = self._vector_store.all_chunks()
        return self._bm25.build(chunks)

    async def rebuild_bm25_async(self) -> int:
        """重建 BM25 索引（异步版本，用于请求处理流程中，避免阻塞事件循环）。"""
        chunks = await self._vector_store.all_chunks_async()
        # 分词 + 建索引是纯 CPU，语料越大越久，放线程池里跑
        return await asyncio.to_thread(self._bm25.build, chunks)

    async def index_document_async(
        self,
        document: SourceDocument,
        registry_documents: Dict[str, Any],
    ) -> IngestFileResult:
        # 切分是纯 CPU（分句/打包/哈希），语料大时明显，放线程池避免卡事件循环
        chunks: List[Chunk] = await asyncio.to_thread(chunk_document, document, self._settings)
        if not chunks:
            return IngestFileResult(
                source=document.source, status="failed", chunks=0, error="切分结果为空"
            )
        await self._vector_store.delete_by_source_async(document.source)
        await self._vector_store.upsert_chunks(chunks)
        registry_documents[document.source] = {
            "doc_id": document.doc_id,
            "checksum": document.checksum,
            "chunks": len(chunks),
            "title": document.title,
        }
        return IngestFileResult(source=document.source, status="indexed", chunks=len(chunks))

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    async def ingest(
        self,
        rebuild: bool = False,
        reset_registry: bool = False,
        only_sources: Optional[Sequence[str]] = None,
        progress_cb: Optional[Any] = None,
    ) -> IngestResponse:
        """增量入库。`progress_cb(dict)` 在关键节点被调用，供异步任务上报进度；

        回调里抛异常会被吞掉（进度上报是增益，不能拖垮入库本身）。
        节点：phase=start / file（每处理完一个文件）/ rebuild_bm25 / done。
        """
        def _report(**data: Any) -> None:
            if progress_cb is None:
                return
            try:
                progress_cb(data)
            except Exception:  # noqa: BLE001 - 进度上报失败不影响入库
                logger.warning("进度回调异常（已忽略）", exc_info=True)

        started = time.perf_counter()
        settings = self._settings
        settings.ensure_directories()

        registry = {"version": 1, "documents": {}, "last_run": None} if reset_registry else self._load_registry()
        registry_documents: Dict[str, Any] = dict(registry.get("documents") or {})

        if rebuild:
            cleared = await asyncio.to_thread(self._vector_store.reset)
            logger.info("rebuild=true，已清空向量库（原 %d 条）", cleared)
            registry_documents = {}
            self._bm25.clear()
        _report(phase="start", rebuild=rebuild, reset_registry=reset_registry)

        only_set = {source.replace("\\", "/") for source in (only_sources or [])}

        # 一次性扫描目录：既用于确定待处理文件，也用于判断哪些来源已从磁盘删除
        on_disk: set = set()
        files: List[Path] = []
        for path in iter_document_files(settings.documents_dir):
            relative = path.relative_to(settings.documents_dir).as_posix()
            on_disk.add(relative)
            if only_set and relative not in only_set:
                continue
            files.append(path)

        # 磁盘上已不存在的来源 -> 清理
        results: List[IngestFileResult] = []
        deleted = 0
        if not only_set:
            for source in list(registry_documents.keys()):
                if source in on_disk:
                    continue
                removed = await self._vector_store.delete_by_source_async(source)
                registry_documents.pop(source, None)
                deleted += 1
                results.append(
                    IngestFileResult(source=source, status="deleted", chunks=removed)
                )

        indexed = failed = skipped = 0
        for path in files:
            relative = path.relative_to(settings.documents_dir).as_posix()
            previous = registry_documents.get(relative) or {}
            try:
                # 文件解析（PDF/docx 尤其慢）是同步阻塞调用，放线程池
                document = await asyncio.to_thread(load_document, path, settings.documents_dir)
            except UnsupportedFormatError as exc:
                failed += 1
                results.append(IngestFileResult(source=relative, status="failed", error=str(exc)))
                logger.error("加载失败 %s：%s", relative, exc)
                _report(phase="file", source=relative, status="failed",
                        indexed=indexed, skipped=skipped, failed=failed)
                continue
            except Exception as exc:  # 兜底，保证一个坏文件不影响整批
                failed += 1
                results.append(
                    IngestFileResult(source=relative, status="failed", error=f"{type(exc).__name__}: {exc}")
                )
                logger.exception("加载异常 %s", relative)
                _report(phase="file", source=relative, status="failed",
                        indexed=indexed, skipped=skipped, failed=failed)
                continue

            if not rebuild and previous.get("checksum") == document.checksum:
                skipped += 1
                results.append(
                    IngestFileResult(
                        source=relative,
                        status="skipped",
                        chunks=int(previous.get("chunks") or 0),
                    )
                )
                _report(phase="file", source=relative, status="skipped",
                        indexed=indexed, skipped=skipped, failed=failed)
                continue

            try:
                file_result = await self.index_document_async(document, registry_documents)
            except Exception as exc:
                failed += 1
                results.append(
                    IngestFileResult(
                        source=relative, status="failed", error=f"{type(exc).__name__}: {exc}"
                    )
                )
                logger.exception("入库异常 %s", relative)
                _report(phase="file", source=relative, status="failed",
                        indexed=indexed, skipped=skipped, failed=failed)
                continue

            indexed += 1
            results.append(file_result)
            _report(phase="file", source=relative, status="indexed",
                    indexed=indexed, skipped=skipped, failed=failed)

        _report(phase="rebuild_bm25", total_files=len(files))
        bm25_size = await self.rebuild_bm25_async()

        registry.update(
            {
                "version": 1,
                "documents": registry_documents,
                "last_run": {
                    "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "indexed": indexed,
                    "skipped": skipped,
                    "failed": failed,
                    "deleted": deleted,
                    "rebuild": rebuild,
                    "bm25_size": bm25_size,
                    "collection_count": self._vector_store.count(),
                },
            }
        )
        self._save_registry(registry)

        # 索引内容变了才宣布版本递增 —— no-op 入库（全 skipped）不碰缓存，
        # 否则每次入库调用都会把还有效的答案白白失效掉
        if self._index_state is not None and (indexed or deleted or rebuild):
            self._index_state.bump(
                reason=f"入库 indexed={indexed} deleted={deleted} rebuild={rebuild}"
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        response = IngestResponse(
            scanned=len(files),
            indexed=indexed,
            skipped=skipped,
            failed=failed,
            deleted=deleted,
            total_chunks_in_store=self._vector_store.count(),
            embedding_provider=self._settings.embedding_provider,
            duration_ms=duration_ms,
            files=results,
        )
        self._last_result = response.model_dump()
        _report(phase="done", duration_ms=duration_ms,
                indexed=indexed, skipped=skipped, failed=failed, deleted=deleted)
        logger.info(
            "入库完成：扫描 %d，新增/更新 %d，跳过 %d，失败 %d，删除 %d，总片段 %d，耗时 %dms",
            len(files),
            indexed,
            skipped,
            failed,
            deleted,
            response.total_chunks_in_store,
            duration_ms,
        )
        return response

    # ------------------------------------------------------------------ #
    def registry_summary(self) -> Dict[str, Any]:
        registry = self._load_registry()
        documents = registry.get("documents") or {}
        return {
            "indexed_documents": len(documents),
            "total_chunks_expected": sum(int(item.get("chunks") or 0) for item in documents.values()),
            "sources": sorted(documents.keys()),
            "last_run": registry.get("last_run"),
        }

    @property
    def last_result(self) -> Optional[Dict[str, Any]]:
        return self._last_result
