"""Agent 业务工具的数据层（SQLite）。

设计意图与边界（重要，面试会问）：
- 本模块只服务两个**结构化查询类**工具：故障码查询与保养项目计算；
- 数据来源是仓库内自带的示例文档（自编码表 + 虚构保养周期表），
  由 `bootstrap_from_documents()` 从 Markdown 表格**机械解析**入库——
  不用大模型抽取，保证同一份文档每次建库结果完全一致、可测试；
- 真实项目替换数据源时，只需实现同样表结构的 SQLite（或换成真实车型数据库），
  上层工具函数与 Agent 图完全不用改——这是"结构留好接口"的具体体现。

三张表：
  dtc_code       故障码定义（码值、描述、可能原因、所属模块）
  maintenance    保养项目（车型、项目、周期原文、周期里程、周期月数）
  vehicle_model  车型元信息（用于判断车型是否在库、以及里程单位）
"""
from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.logging_conf import get_logger

logger = get_logger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS dtc_code (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    code         TEXT NOT NULL,
    system       TEXT DEFAULT '',
    description  TEXT NOT NULL,
    possible_causes TEXT DEFAULT '',
    model        TEXT DEFAULT '通用',
    source       TEXT DEFAULT '',
    UNIQUE(code, model, description)
);
CREATE INDEX IF NOT EXISTS idx_dtc_code ON dtc_code(code);

CREATE TABLE IF NOT EXISTS maintenance (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    car_model     TEXT NOT NULL,
    item          TEXT NOT NULL,
    period_text   TEXT NOT NULL,
    period_km     INTEGER,
    period_months INTEGER,
    note          TEXT DEFAULT '',
    source        TEXT DEFAULT '',
    UNIQUE(car_model, item, period_text)
);
CREATE INDEX IF NOT EXISTS idx_maint_model ON maintenance(car_model);

CREATE TABLE IF NOT EXISTS vehicle_model (
    car_model   TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    source      TEXT DEFAULT '',
    is_fictional INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass
class DtcRecord:
    code: str
    system: str
    description: str
    possible_causes: str
    model: str
    source: str


@dataclass
class MaintenanceRecord:
    car_model: str
    item: str
    period_text: str
    period_km: Optional[int]
    period_months: Optional[int]
    note: str
    source: str


# --------------------------------------------------------------------------- #
# Markdown 表格解析（机械解析，无大模型参与）
# --------------------------------------------------------------------------- #
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")


def _split_row(line: str) -> List[str]:
    match = _TABLE_ROW.match(line)
    if not match:
        return []
    cells = [cell.strip() for cell in match.group(1).split("|")]
    return cells


def _is_separator(cells: Iterable[str]) -> bool:
    joined = "".join(cells)
    return bool(joined) and set(joined) <= set("-: ")


def iter_tables(text: str) -> List[Tuple[List[str], List[List[str]], int]]:
    """从 Markdown 文本中切出所有表格，返回 [(表头, 数据行列表, 起始行号)]。"""
    lines = text.splitlines()
    tables: List[Tuple[List[str], List[List[str]], int]] = []
    index = 0
    while index < len(lines):
        cells = _split_row(lines[index])
        if cells and index + 1 < len(lines) and _is_separator(_split_row(lines[index + 1])):
            header = cells
            rows: List[List[str]] = []
            cursor = index + 2
            while cursor < len(lines):
                row_cells = _split_row(lines[cursor])
                if not row_cells or _is_separator(row_cells):
                    break
                rows.append(row_cells)
                cursor += 1
            tables.append((header, rows, index + 1))
            index = cursor
        else:
            index += 1
    return tables


def parse_period(period_text: str) -> Tuple[Optional[int], Optional[int]]:
    """把周期原文解析成 (里程 km, 月数)。解析不出就返回 None，不做猜测。

    例：
      "每 8,000 km 或每 8 个月（以先到者为准）" -> (8000, 8)
      "每 2 年"                                  -> (None, 24)
      "每 64,000 km"                             -> (64000, None)
    """
    text = (period_text or "").replace(",", "").replace("，", "")
    km: Optional[int] = None
    months: Optional[int] = None

    km_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:万)?\s*(?:km|公里|千米)", text)
    if km_match:
        value = float(km_match.group(1))
        if "万" in km_match.group(0):
            value *= 10000
        km = int(value)

    month_match = re.search(r"(\d+(?:\.\d+)?)\s*个月", text)
    if month_match:
        months = int(float(month_match.group(1)))
    else:
        year_match = re.search(r"(\d+(?:\.\d+)?)\s*年", text)
        if year_match:
            months = int(float(year_match.group(1)) * 12)

    return km, months


def iter_dtc_rows(source_text: str, source_name: str) -> List[DtcRecord]:
    """从码表文档中提取故障码记录。

    支持两种表头（仓库里两种都存在）：
      故障码 | 中文描述
      DTC | FTB | 中文描述 | 所属模块
    """
    records: List[DtcRecord] = []
    for header, rows, _ in iter_tables(source_text):
        normalized = [cell.replace(" ", "") for cell in header]
        if "故障码" in normalized and "中文描述" in normalized:
            code_idx = normalized.index("故障码")
            desc_idx = normalized.index("中文描述")
            for row in rows:
                if len(row) <= max(code_idx, desc_idx):
                    continue
                code = row[code_idx].strip()
                description = row[desc_idx].strip()
                # 区间行（P0300–P0399）不是具体码，跳过
                if not code or "–" in code or "-–" in code or re.search(r"[–—]", code):
                    continue
                if not re.fullmatch(r"[PBCU]\d{4}", code):
                    continue
                records.append(
                    DtcRecord(code=code, system="", description=description,
                              possible_causes="", model="通用", source=source_name)
                )
        elif "DTC" in normalized and "中文描述" in normalized:
            code_idx = normalized.index("DTC")
            desc_idx = normalized.index("中文描述")
            mod_idx = normalized.index("所属模块") if "所属模块" in normalized else None
            ftb_idx = normalized.index("FTB") if "FTB" in normalized else None
            for row in rows:
                if len(row) <= max(code_idx, desc_idx):
                    continue
                code = row[code_idx].strip()
                if not re.fullmatch(r"[PBCU]\d{4}", code):
                    continue
                description = row[desc_idx].strip()
                module = row[mod_idx].strip() if mod_idx is not None and len(row) > mod_idx else ""
                ftb = row[ftb_idx].strip() if ftb_idx is not None and len(row) > ftb_idx else ""
                causes = f"FTB={ftb}" if ftb else ""
                records.append(
                    DtcRecord(code=code, system=module, description=description,
                              possible_causes=causes, model="示例车企B", source=source_name)
                )
    return records


def iter_maintenance_rows(source_text: str, source_name: str, car_model: str) -> List[MaintenanceRecord]:
    """从保养周期表中提取保养项目。只识别「保养项目 | 周期」这种两列表。"""
    records: List[MaintenanceRecord] = []
    for header, rows, _ in iter_tables(source_text):
        normalized = [cell.replace(" ", "") for cell in header]
        if normalized[:2] != ["保养项目", "周期"]:
            continue
        for row in rows:
            if len(row) < 2:
                continue
            item = row[0].strip()
            period_text = row[1].strip()
            if not item or not period_text:
                continue
            km, months = parse_period(period_text)
            records.append(
                MaintenanceRecord(car_model=car_model, item=item, period_text=period_text,
                                  period_km=km, period_months=months, note="", source=source_name)
            )
    return records


# --------------------------------------------------------------------------- #
# 仓储
# --------------------------------------------------------------------------- #
class AgentKbStore:
    """业务工具的数据仓储。线程安全（FastAPI 多线程 + to_thread 调用）。"""

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ #
    def is_empty(self) -> bool:
        with self._lock:
            dtc = self._conn.execute("SELECT COUNT(*) AS n FROM dtc_code").fetchone()["n"]
            maint = self._conn.execute("SELECT COUNT(*) AS n FROM maintenance").fetchone()["n"]
        return dtc == 0 and maint == 0

    def upsert_dtc(self, records: Iterable[DtcRecord]) -> int:
        rows = [
            (r.code, r.system, r.description, r.possible_causes, r.model, r.source) for r in records
        ]
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO dtc_code(code, system, description, possible_causes, model, source) "
                "VALUES(?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def upsert_maintenance(self, records: Iterable[MaintenanceRecord]) -> int:
        rows = [
            (r.car_model, r.item, r.period_text, r.period_km, r.period_months, r.note, r.source)
            for r in records
        ]
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO maintenance(car_model, item, period_text, period_km, "
                "period_months, note, source) VALUES(?,?,?,?,?,?,?)",
                rows,
            )
            self._conn.commit()
        return len(rows)

    def register_model(self, car_model: str, display_name: str, source: str, is_fictional: bool = True) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO vehicle_model(car_model, display_name, source, is_fictional) "
                "VALUES(?,?,?,?)",
                (car_model, display_name, source, 1 if is_fictional else 0),
            )
            self._conn.commit()

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO meta(key, value) VALUES(?,?)", (key, value)
            )
            self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    # ------------------------------------------------------------------ #
    def find_dtc(self, code: str) -> List[DtcRecord]:
        """按码值精确查询（大小写不敏感）。返回可能来自多个数据源的多条记录。"""
        normalized = (code or "").strip().upper()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM dtc_code WHERE UPPER(code) = ? ORDER BY model, source", (normalized,)
            ).fetchall()
        return [
            DtcRecord(code=row["code"], system=row["system"], description=row["description"],
                      possible_causes=row["possible_causes"], model=row["model"], source=row["source"])
            for row in rows
        ]

    def search_dtc(self, keyword: str, limit: int = 20) -> List[DtcRecord]:
        """按描述模糊查询（用户可能只记得"失火""机油"这类症状词）。"""
        pattern = f"%{keyword}%"
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM dtc_code WHERE description LIKE ? LIMIT ?", (pattern, limit)
            ).fetchall()
        return [
            DtcRecord(code=row["code"], system=row["system"], description=row["description"],
                      possible_causes=row["possible_causes"], model=row["model"], source=row["source"])
            for row in rows
        ]

    def get_model(self, car_model: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM vehicle_model WHERE car_model = ?", (car_model,)
            ).fetchone()
        if row is None:
            return None
        return {
            "car_model": row["car_model"],
            "display_name": row["display_name"],
            "source": row["source"],
            "is_fictional": bool(row["is_fictional"]),
        }

    def list_models(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT car_model FROM vehicle_model ORDER BY car_model"
            ).fetchall()
        return [row["car_model"] for row in rows]

    def maintenance_for(self, car_model: str) -> List[MaintenanceRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM maintenance WHERE car_model = ? ORDER BY id", (car_model,)
            ).fetchall()
        return [
            MaintenanceRecord(car_model=row["car_model"], item=row["item"],
                              period_text=row["period_text"], period_km=row["period_km"],
                              period_months=row["period_months"], note=row["note"] or "",
                              source=row["source"])
            for row in rows
        ]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            dtc = self._conn.execute("SELECT COUNT(*) AS n FROM dtc_code").fetchone()["n"]
            maint = self._conn.execute("SELECT COUNT(*) AS n FROM maintenance").fetchone()["n"]
            models = self._conn.execute("SELECT COUNT(*) AS n FROM vehicle_model").fetchone()["n"]
            by_model = self._conn.execute(
                "SELECT model, COUNT(*) AS n FROM dtc_code GROUP BY model"
            ).fetchall()
        return {
            "dtc_total": dtc,
            "maintenance_total": maint,
            "vehicle_models": models,
            "dtc_by_model": {row["model"]: row["n"] for row in by_model},
            "built_at": self.get_meta("built_at"),
            "db_path": str(self._path),
        }


# --------------------------------------------------------------------------- #
# 从示例文档建库
# --------------------------------------------------------------------------- #
# 车型展示名映射（示例数据只有虚构车型 A）
_MODEL_ALIASES = {
    "虚构车型A": "虚构车型 A",
    "虚构车型 A": "虚构车型 A",
    "示例车型A": "虚构车型 A",
    "车型A": "虚构车型 A",
    "车型 A": "虚构车型 A",
}

# 码表文档 -> (文件名, 归类到哪个 model)
_DTC_DOCS = (
    ("故障码表：OBD-II标准码表中文对照（自编）.md", "通用"),
    ("故障码表：OBD-II标准码表英文对照（自编）.md", "通用"),
    ("故障码表：示例车企B-DTC总列表（虚构）.md", "示例车企B"),
)

_MAINT_DOC = "示例-虚构车型A-保养与故障码.md"
_MAINT_MODEL = "虚构车型 A"


def normalize_car_model(raw: str) -> str:
    """把用户写的车型归一化到库内标准名；无法识别时返回原样（由上层判定为不支持）。"""
    text = (raw or "").strip()
    if not text:
        return ""
    if text in _MODEL_ALIASES:
        return _MODEL_ALIASES[text]
    compact = text.replace(" ", "")
    for alias, standard in _MODEL_ALIASES.items():
        if alias.replace(" ", "") == compact:
            return standard
    return text


def bootstrap_from_documents(store: AgentKbStore, documents_dir: Path) -> Dict[str, Any]:
    """从 data/documents 下的示例文档建库（幂等，可重复执行）。

    返回建库统计。若文档缺失则跳过对应部分并如实报告，不伪造数据。
    """
    import time

    result: Dict[str, Any] = {"dtc_inserted": 0, "maintenance_inserted": 0, "missing": [], "sources": []}

    for filename, model in _DTC_DOCS:
        path = documents_dir / filename
        if not path.exists():
            result["missing"].append(filename)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        records = iter_dtc_rows(text, filename)
        for record in records:
            record.model = model
        result["dtc_inserted"] += store.upsert_dtc(records)
        result["sources"].append({"file": filename, "records": len(records), "kind": "dtc"})

    maint_path = documents_dir / _MAINT_DOC
    if maint_path.exists():
        text = maint_path.read_text(encoding="utf-8", errors="replace")
        records = iter_maintenance_rows(text, _MAINT_DOC, _MAINT_MODEL)
        result["maintenance_inserted"] = store.upsert_maintenance(records)
        store.register_model(_MAINT_MODEL, "虚构车型 A（示例数据）", _MAINT_DOC, is_fictional=True)
        result["sources"].append({"file": _MAINT_DOC, "records": len(records), "kind": "maintenance"})
    else:
        result["missing"].append(_MAINT_DOC)

    store.set_meta("built_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    store.set_meta("documents_dir", str(documents_dir))
    logger.info(
        "业务工具知识库建库完成：故障码 %d 条，保养项目 %d 条，缺失文档 %d 个",
        result["dtc_inserted"], result["maintenance_inserted"], len(result["missing"]),
    )
    return result
