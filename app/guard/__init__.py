"""FreshGuard：知识库变更影响分析（把文档改动变成可审计的结论变更清单）。"""

from app.guard.claims import Claim, classify_change, normalize_value, severity_of
from app.guard.extractor import ClaimExtractor, plan_incremental
from app.guard.service import DriftService
from app.guard.store import ClaimStore

__all__ = [
    "Claim",
    "ClaimExtractor",
    "ClaimStore",
    "DriftService",
    "classify_change",
    "normalize_value",
    "plan_incremental",
    "severity_of",
]
