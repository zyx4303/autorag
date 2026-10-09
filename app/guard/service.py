"""变更影响分析（FreshGuard 的核心）。

流程：
    1. 加载文档 → 切分片段（复用 app/core/chunker.py）
    2. 计算每个片段的内容哈希，与库中记录比对
       · 哈希一致 → 直接复用该片段上一次的声明（零 LLM 调用）
       · 哈希变化或新片段 → 调 LLM 重新抽取
       · 已删除的片段 → 其声明在新版本中消失
    3. 把本次结果写成一个新的 doc_version
    4. 与上一个版本做差分，得到 added / removed / modified 三类变更，
       并按严重度排序，输出"哪些结论需要重新确认"
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings
from app.core.chunker import chunk_document
from app.core.llm_client import LLMClient
from app.core.loader import load_document
from app.guard.claims import Claim, classify_change, severity_of
from app.guard.extractor import ClaimExtractor, plan_incremental
from app.guard.store import ClaimStore
from app.logging_conf import get_logger

logger = get_logger(__name__)

SEVERITY_ORDER = {"high": 0, "medium": 1, "recheck": 2, "low": 3, "none": 4}


@dataclass
class ExtractedChunk:
    chunk_id: str
    claims: List[Claim] = field(default_factory=list)


class DriftService:
    def __init__(
        self,
        settings: Settings,
        llm: LLMClient,
        store: ClaimStore,
    ) -> None:
        self._settings = settings
        self._store = store
        self._extractor = ClaimExtractor(llm)

    @property
    def extractor_stats(self) -> Dict[str, Any]:
        return dict(self._extractor.stats)

    # ------------------------------------------------------------------ #
    # 抽取（增量）
    # ------------------------------------------------------------------ #
    async def extract_document(
        self,
        rel_path: str,
        force: bool = False,
    ) -> Dict[str, Any]:
        """对单个文档做增量抽取，并生成新版本记录。

        force=True 时忽略缓存，对所有片段重新抽取（用于换模型或改提示词后重建）。
        """
        settings = self._settings
        path = settings.documents_dir / rel_path
        if not path.exists():
            raise FileNotFoundError(f"文档不存在：{path}")

        document = load_document(path, settings.documents_dir)
        chunks = chunk_document(document, settings)
        if not chunks:
            return {"doc_source": rel_path, "error": "切分结果为空"}

        previous_version = self._store.latest_version(rel_path)
        previous_checksum = previous_version["checksum"] if previous_version else None
        if not force and previous_checksum == document.checksum:
            return {
                "doc_source": rel_path,
                "status": "unchanged",
                "checksum": document.checksum[:12],
                "chunk_count": len(chunks),
                "extracted_chunks": 0,
                "reused_chunks": len(chunks),
                "claims": int(previous_version["claim_count"]) if previous_version else 0,
                "llm_calls": 0,
                "message": "文档内容未变化，跳过（零 LLM 调用）",
            }

        known_hashes = {} if force else self._store.load_chunk_hashes(rel_path)
        to_extract, reusable = plan_incremental(chunks, known_hashes)

        # 复用旧声明：从上一个版本里按 chunk_id 取回
        reused_claims: List[Claim] = []
        if previous_version and reusable:
            reusable_ids = {chunk.chunk_id for chunk in reusable}
            for claim in self._store.claims_of_version(int(previous_version["id"])):
                if claim.chunk_id in reusable_ids:
                    reused_claims.append(claim)

        # 重新抽取变化的片段
        fresh_claims: List[Claim] = []
        llm_calls_before = self._extractor.stats["llm_calls"]
        for chunk in to_extract:
            claims = await self._extractor.extract_from_chunk(chunk, document.checksum)
            fresh_claims.extend(claims)

        all_claims = fresh_claims + reused_claims
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        version_id = self._store.create_version(
            doc_source=rel_path,
            checksum=document.checksum,
            chunk_count=len(chunks),
            claim_count=len(all_claims),
            now=now,
            model=self._settings.llm_model,
            note=f"增量：重抽 {len(to_extract)} 片，复用 {len(reusable)} 片" + ("（force）" if force else ""),
        )
        self._store.save_claims(version_id, all_claims)

        # 更新片段哈希（并清理已消失片段的历史哈希）
        hashes = {chunk.chunk_id: ClaimExtractor.chunk_content_hash(chunk.text) for chunk in chunks}
        self._store.save_chunk_hashes(rel_path, hashes, now)
        self._store.drop_chunk_hashes(rel_path, list(hashes.keys()))

        self._extractor.stats["reused_chunks"] += len(reusable)
        llm_calls = self._extractor.stats["llm_calls"] - llm_calls_before

        return {
            "doc_source": rel_path,
            "status": "extracted",
            "checksum": document.checksum[:12],
            "version_id": version_id,
            "chunk_count": len(chunks),
            "extracted_chunks": len(to_extract),
            "reused_chunks": len(reusable),
            "claims": len(all_claims),
            "claims_fresh": len(fresh_claims),
            "claims_reused": len(reused_claims),
            "llm_calls": llm_calls,
            "previous_version_id": int(previous_version["id"]) if previous_version else None,
        }

    async def scan_all(self, force: bool = False) -> Dict[str, Any]:
        """扫描知识库目录下所有文档。"""
        settings = self._settings
        results: List[Dict[str, Any]] = []
        files = sorted(
            p for p in settings.documents_dir.rglob("*")
            if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in {".md", ".markdown", ".txt"}
        )
        started = time.perf_counter()
        for path in files:
            rel = path.relative_to(settings.documents_dir).as_posix()
            try:
                results.append(await self.extract_document(rel, force=force))
            except Exception as exc:  # 单个文档失败不影响整批
                logger.exception("文档抽取失败：%s", rel)
                results.append({"doc_source": rel, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        return {
            "scanned": len(files),
            "duration_ms": int((time.perf_counter() - started) * 1000),
            "results": results,
            "extractor_stats": self.extractor_stats,
        }

    # ------------------------------------------------------------------ #
    # 差分
    # ------------------------------------------------------------------ #
    def diff_document(self, doc_source: str) -> Dict[str, Any]:
        """对比某文档最近两个版本的声明集合，输出变更清单。"""
        versions = self._versions_of(doc_source, limit=2)
        if len(versions) < 2:
            return {
                "doc_source": doc_source,
                "status": "insufficient_versions",
                "message": "该文档只有一个版本，无法比较（改一次文档后再跑抽取即可）",
                "changes": [],
                "summary": {"added": 0, "removed": 0, "modified": 0, "qualifier_changed": 0, "unchanged": 0},
            }

        new_version, old_version = versions[0], versions[1]
        old_claims = {c.claim_key: c for c in self._store.claims_of_version(int(old_version["id"]))}
        new_claims = {c.claim_key: c for c in self._store.claims_of_version(int(new_version["id"]))}

        changes: List[Dict[str, Any]] = []
        counts = {"added": 0, "removed": 0, "modified": 0, "qualifier_changed": 0, "unchanged": 0}

        # 证据复核用的当前正文（把所有片段拼起来，忽略空白差异后做子串匹配）
        current_text = self._current_document_text(doc_source)

        for key in sorted(set(old_claims) | set(new_claims)):
            old, new = old_claims.get(key), new_claims.get(key)
            change = classify_change(old, new)
            if change == "unchanged":
                counts["unchanged"] += 1
                continue

            source = new or old
            assert source is not None

            severity = severity_of(change, source.claim_type)
            note = ""

            # ---- 抽取抖动抑制 ----
            # 实测问题：大模型每次抽取的颗粒度并不完全一致，
            # 同一条声明这次可能没被抽出来，于是被误判为"删除"。
            # 判据：如果标为删除（或值未变的 added）的声明，其 evidence 原文
            # 仍然出现在当前文档正文中，那它大概率只是这次没被抽到，
            # 而不是知识被删掉了 —— 降级为 recheck 并标注原因。
            if change == "removed" and self._evidence_still_present(old, current_text):
                severity = "recheck"
                note = "原文仍存在，疑似本次未被抽取（非知识删除），建议复核后确认"
            elif change == "added" and new is not None and not new.value_normalized:
                severity = "recheck"
                note = "新增声明且无法归一化，建议人工确认是否属于补充信息"
            elif change == "qualifier_changed":
                note = "值未变，仅限定条件的抽取结果不同（模型对限定条件的识别不稳定）"

            counts[change] += 1
            changes.append(
                {
                    "claim_key": key,
                    "change": change,
                    "severity": severity,
                    "claim_type": source.claim_type,
                    "subject": source.subject,
                    "predicate": source.predicate,
                    "old_value": old.value if old else None,
                    "new_value": new.value if new else None,
                    "subject_section": source.section,
                    "evidence": (new.evidence if new and change != "removed" else old.evidence if old else ""),
                    "note": note,
                }
            )

        changes.sort(key=lambda item: (SEVERITY_ORDER.get(item["severity"], 9), item["subject"]))
        return {
            "doc_source": doc_source,
            "status": "ok",
            "old_version": {
                "id": int(old_version["id"]),
                "checksum": (old_version["checksum"] or "")[:12],
                "claim_count": int(old_version["claim_count"]),
                "extracted_at": old_version["extracted_at"],
            },
            "new_version": {
                "id": int(new_version["id"]),
                "checksum": (new_version["checksum"] or "")[:12],
                "claim_count": int(new_version["claim_count"]),
                "extracted_at": new_version["extracted_at"],
            },
            "summary": counts,
            "changes": changes,
        }

    def _current_document_text(self, doc_source: str) -> str:
        """把文档当前正文（重新切分后的全部片段）拼成一个忽略空白的字符串，
        用于证据复核。文档不存在或读取失败时返回空串。"""
        import re as _re

        try:
            path = self._settings.documents_dir / doc_source
            if not path.exists():
                return ""
            document = load_document(path, self._settings.documents_dir)
            chunks = chunk_document(document, self._settings)
            joined = "".join(chunk.text for chunk in chunks)
            return _re.sub(r"\s+", "", joined)
        except Exception as exc:
            logger.warning("读取文档正文失败（证据复核将跳过）：%s -> %s", doc_source, exc)
            return ""

    @staticmethod
    def _evidence_still_present(claim: Optional[Claim], current_text: str) -> bool:
        """判断声明的 evidence 是否仍能在当前正文中找到。"""
        import re as _re

        if claim is None or not current_text:
            return False
        evidence = _re.sub(r"\s+", "", claim.evidence or "")
        if not evidence:
            return False
        # 长证据按前 40 字匹配（表格行等可能被切分处轻微影响）
        probe = evidence[:40]
        return probe in current_text

    def _versions_of(self, doc_source: str, limit: int = 2):        # 直接读库：按 doc_source 取最近 limit 个版本
        rows = []
        with self._store._lock:  # noqa: SLF001 - 同一模块内的受控访问
            rows = self._store._conn.execute(  # noqa: SLF001
                "SELECT * FROM doc_version WHERE doc_source = ? ORDER BY id DESC LIMIT ?",
                (doc_source, limit),
            ).fetchall()
        return rows

    def diff_all(self) -> Dict[str, Any]:
        """对所有有 ≥2 个版本的文档做差分，汇总"需要重新确认的结论"。"""
        with self._store._lock:  # noqa: SLF001
            docs = [
                row["doc_source"]
                for row in self._store._conn.execute(  # noqa: SLF001
                    "SELECT doc_source, COUNT(*) AS n FROM doc_version GROUP BY doc_source HAVING n >= 2"
                ).fetchall()
            ]
        per_doc = [self.diff_document(doc) for doc in docs]
        high = [c for item in per_doc for c in item.get("changes", []) if c["severity"] == "high"]
        medium = [c for item in per_doc for c in item.get("changes", []) if c["severity"] == "medium"]
        return {
            "documents_compared": len(per_doc),
            "high_severity_count": len(high),
            "medium_severity_count": len(medium),
            "high_severity": high,
            "per_document": per_doc,
        }

    # ------------------------------------------------------------------ #
    def affects(self, keyword: str) -> Dict[str, Any]:
        """"这句话要是变了，谁会受影响"——按关键词反查当前库中的相关声明。

        这是人工审核知识库时最想做但做不到的事：
        一条保养周期被改动，究竟还有哪些文档/章节在引用同一事实。
        """
        rows = self._store.find_claims_by_subject(keyword)
        docs: Dict[str, int] = {}
        for row in rows:
            docs[row["doc_source"]] = docs.get(row["doc_source"], 0) + 1
        return {
            "keyword": keyword,
            "matched_claims": len(rows),
            "affected_documents": docs,
            "claims": rows,
        }
