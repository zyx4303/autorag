"""声明库（SQLite）：以"文档版本"为单位保存声明快照，支撑跨版本差分。

为什么用 SQLite 而不是直接塞进 Chroma：
- 声明是**结构化记录**，需要按 doc / 类型 / 主体做精确查询与聚合，这是关系库的强项；
- 需要保留历史版本用于差分（向量库只保留"当前状态"，做不了版本对比）；
- 零额外依赖（标准库 sqlite3），便于部署。

表设计：
  doc_version  每个文档每次抽取产生一条版本记录（含内容哈希、片段数、声明数）
  chunk_hash   片段内容哈希，用于增量抽取判断（内容没变就不重新调 LLM）
  claim        声明快照，按 version_id 归档，永久保留历史
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.guard.claims import Claim
from app.logging_conf import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS doc_version (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_source    TEXT NOT NULL,
    checksum      TEXT NOT NULL,
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    claim_count   INTEGER NOT NULL DEFAULT 0,
    extracted_at  TEXT NOT NULL,
    model         TEXT DEFAULT '',
    note          TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_version_doc ON doc_version(doc_source, id DESC);

CREATE TABLE IF NOT EXISTS chunk_hash (
    chunk_id    TEXT PRIMARY KEY,
    doc_source  TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunk_doc ON chunk_hash(doc_source);

CREATE TABLE IF NOT EXISTS claim (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    version_id    INTEGER NOT NULL,
    claim_key     TEXT NOT NULL,
    doc_source    TEXT NOT NULL,
    section       TEXT DEFAULT '',
    claim_type    TEXT NOT NULL,
    subject       TEXT NOT NULL,
    predicate     TEXT DEFAULT '',
    value         TEXT DEFAULT '',
    value_normalized TEXT,
    qualifiers    TEXT DEFAULT '[]',
    evidence      TEXT DEFAULT '',
    chunk_id      TEXT DEFAULT '',
    char_start    INTEGER DEFAULT 0,
    char_end      INTEGER DEFAULT 0,
    doc_checksum  TEXT DEFAULT '',
    extracted_at  TEXT DEFAULT '',
    model         TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_claim_version ON claim(version_id);
CREATE INDEX IF NOT EXISTS idx_claim_key ON claim(claim_key);
CREATE INDEX IF NOT EXISTS idx_claim_subject ON claim(subject);
"""


class ClaimStore:
    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()
        logger.info("声明库就绪：%s", self._path)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    # 片段哈希（增量抽取用）
    # ------------------------------------------------------------------ #
    def load_chunk_hashes(self, doc_source: str) -> Dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id, content_hash FROM chunk_hash WHERE doc_source = ?",
                (doc_source,),
            ).fetchall()
        return {row["chunk_id"]: row["content_hash"] for row in rows}

    def save_chunk_hashes(self, doc_source: str, hashes: Dict[str, str], now: str) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT INTO chunk_hash(chunk_id, doc_source, content_hash, updated_at) "
                "VALUES(?, ?, ?, ?) ON CONFLICT(chunk_id) DO UPDATE SET "
                "content_hash = excluded.content_hash, updated_at = excluded.updated_at",
                [(cid, doc_source, h, now) for cid, h in hashes.items()],
            )
            self._conn.commit()

    def drop_chunk_hashes(self, doc_source: str, keep_ids: List[str]) -> None:
        """删除该文档下已经不存在的片段哈希（文档被改短时会发生）。"""
        with self._lock:
            rows = self._conn.execute(
                "SELECT chunk_id FROM chunk_hash WHERE doc_source = ?", (doc_source,)
            ).fetchall()
            stale = [row["chunk_id"] for row in rows if row["chunk_id"] not in set(keep_ids)]
            if stale:
                self._conn.executemany(
                    "DELETE FROM chunk_hash WHERE chunk_id = ?", [(cid,) for cid in stale]
                )
                self._conn.commit()

    # ------------------------------------------------------------------ #
    # 版本与声明
    # ------------------------------------------------------------------ #
    def latest_version(self, doc_source: str) -> Optional[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM doc_version WHERE doc_source = ? ORDER BY id DESC LIMIT 1",
                (doc_source,),
            ).fetchone()

    def create_version(
        self,
        doc_source: str,
        checksum: str,
        chunk_count: int,
        claim_count: int,
        now: str,
        model: str,
        note: str = "",
    ) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO doc_version(doc_source, checksum, chunk_count, claim_count, "
                "extracted_at, model, note) VALUES(?, ?, ?, ?, ?, ?, ?)",
                (doc_source, checksum, chunk_count, claim_count, now, model, note),
            )
            self._conn.commit()
            return int(cursor.lastrowid)

    def save_claims(self, version_id: int, claims: List[Claim]) -> int:
        if not claims:
            return 0
        rows = [
            (
                version_id, claim.claim_key, claim.doc_source, claim.section, claim.claim_type,
                claim.subject, claim.predicate, claim.value, claim.value_normalized,
                json.dumps(claim.qualifiers, ensure_ascii=False), claim.evidence, claim.chunk_id,
                claim.char_start, claim.char_end, claim.doc_checksum, claim.extracted_at, claim.model,
            )
            for claim in claims
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO claim(version_id, claim_key, doc_source, section, claim_type, "
                "subject, predicate, value, value_normalized, qualifiers, evidence, chunk_id, "
                "char_start, char_end, doc_checksum, extracted_at, model) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def claims_of_version(self, version_id: int) -> List[Claim]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM claim WHERE version_id = ? ORDER BY id", (version_id,)
            ).fetchall()
        return [self._row_to_claim(row) for row in rows]

    def claims_of_latest(self, doc_source: str) -> List[Claim]:
        version = self.latest_version(doc_source)
        if version is None:
            return []
        return self.claims_of_version(int(version["id"]))

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> Claim:
        try:
            qualifiers = json.loads(row["qualifiers"] or "[]")
        except json.JSONDecodeError:
            qualifiers = []
        return Claim(
            claim_key=row["claim_key"],
            doc_source=row["doc_source"],
            section=row["section"] or "",
            claim_type=row["claim_type"],
            subject=row["subject"],
            predicate=row["predicate"] or "",
            value=row["value"] or "",
            value_normalized=row["value_normalized"],
            qualifiers=list(qualifiers),
            evidence=row["evidence"] or "",
            chunk_id=row["chunk_id"] or "",
            char_start=int(row["char_start"] or 0),
            char_end=int(row["char_end"] or 0),
            doc_checksum=row["doc_checksum"] or "",
            extracted_at=row["extracted_at"] or "",
            model=row["model"] or "",
        )

    # ------------------------------------------------------------------ #
    # 统计与查询
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        with self._lock:
            versions = self._conn.execute("SELECT COUNT(*) AS n FROM doc_version").fetchone()["n"]
            claims = self._conn.execute("SELECT COUNT(*) AS n FROM claim").fetchone()["n"]
            docs = self._conn.execute(
                "SELECT COUNT(DISTINCT doc_source) AS n FROM doc_version"
            ).fetchone()["n"]
            by_type = self._conn.execute(
                "SELECT claim_type, COUNT(*) AS n FROM claim GROUP BY claim_type ORDER BY n DESC"
            ).fetchall()
            latest = self._conn.execute(
                "SELECT doc_source, checksum, claim_count, extracted_at FROM doc_version v "
                "WHERE id = (SELECT MAX(id) FROM doc_version WHERE doc_source = v.doc_source) "
                "ORDER BY doc_source"
            ).fetchall()
        return {
            "documents": docs,
            "versions": versions,
            "claims_total": claims,
            "claims_by_type": {row["claim_type"]: row["n"] for row in by_type},
            "latest_versions": [
                {
                    "doc_source": row["doc_source"],
                    "checksum": (row["checksum"] or "")[:12],
                    "claim_count": row["claim_count"],
                    "extracted_at": row["extracted_at"],
                }
                for row in latest
            ],
        }

    def find_claims_by_subject(self, keyword: str, limit: int = 50) -> List[Dict[str, Any]]:
        """按主体/值模糊查询当前最新版本里的声明（面板上的检索框）。"""
        pattern = f"%{keyword}%"
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.*, v.extracted_at AS version_time FROM claim c "
                "JOIN doc_version v ON v.id = c.version_id "
                "WHERE c.version_id IN (SELECT MAX(id) FROM doc_version GROUP BY doc_source) "
                "AND (c.subject LIKE ? OR c.value LIKE ? OR c.predicate LIKE ?) "
                "ORDER BY c.doc_source LIMIT ?",
                (pattern, pattern, pattern, limit),
            ).fetchall()
        return [
            {
                "doc_source": row["doc_source"],
                "section": row["section"],
                "claim_type": row["claim_type"],
                "subject": row["subject"],
                "predicate": row["predicate"],
                "value": row["value"],
                "qualifiers": json.loads(row["qualifiers"] or "[]"),
                "evidence": row["evidence"],
                "version_time": row["version_time"],
            }
            for row in rows
        ]
