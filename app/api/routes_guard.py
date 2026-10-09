"""FreshGuard 接口：声明抽取、版本差分、影响面查询、审计面板。

接口一览（前缀 /api/v1/guard）：
  POST /extract        对单个文档做增量抽取（force=true 则全量重抽）
  POST /scan           扫描整个知识库目录
  GET  /versions       列出各文档的最新版本
  GET  /diff           查看某文档最近两个版本的变更清单
  GET  /diff/all       汇总所有文档的高严重度变更
  GET  /affects        按关键词反查相关声明（"这句话变了谁受影响"）
  GET  /claims         浏览当前库中的声明
  GET  /stats          声明库统计
  GET  /panel          审计面板（HTML）
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse

from app.api.deps import ServiceContainer, get_container, verify_api_key
from app.config import PROJECT_ROOT
from app.guard.service import DriftService
from app.logging_conf import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/guard", tags=["guard"])

PANEL_HTML = PROJECT_ROOT / "app" / "web" / "guard.html"


def get_drift_service(container: ServiceContainer = Depends(get_container)) -> DriftService:
    service = container.drift
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="变更分析服务未就绪：" + "；".join(container.init_errors or ["初始化失败"]),
        )
    return service


@router.post("/extract", summary="对单个文档做增量抽取")
async def extract(
    source: str = Query(..., description="相对 data/documents 的文档路径，如 示例-虚构车型A-保养与故障码.md"),
    force: bool = Query(default=False, description="true 时忽略缓存，对全部片段重新抽取"),
    service: DriftService = Depends(get_drift_service),
    _: None = Depends(verify_api_key),
) -> dict:
    try:
        return await service.extract_document(source, force=force)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/scan", summary="扫描整个知识库并增量抽取")
async def scan(
    force: bool = Query(default=False),
    service: DriftService = Depends(get_drift_service),
    _: None = Depends(verify_api_key),
) -> dict:
    return await service.scan_all(force=force)


@router.get("/versions", summary="各文档的最新版本")
async def versions(
    container: ServiceContainer = Depends(get_container),
    _: None = Depends(verify_api_key),
) -> dict:
    if container.claim_store is None:
        raise HTTPException(status_code=503, detail="声明库未就绪")
    return container.claim_store.summary()


@router.get("/diff", summary="查看某文档最近两个版本的变更清单")
async def diff(
    source: str = Query(..., description="文档相对路径"),
    service: DriftService = Depends(get_drift_service),
    _: None = Depends(verify_api_key),
) -> dict:
    return service.diff_document(source)


@router.get("/diff/all", summary="汇总所有文档的高严重度变更")
async def diff_all(
    service: DriftService = Depends(get_drift_service),
    _: None = Depends(verify_api_key),
) -> dict:
    return service.diff_all()


@router.get("/affects", summary="按关键词反查相关声明")
async def affects(
    keyword: str = Query(..., min_length=1, description="如 机油、火花塞、包修期"),
    service: DriftService = Depends(get_drift_service),
    _: None = Depends(verify_api_key),
) -> dict:
    return service.affects(keyword)


@router.get("/claims", summary="浏览当前库中的声明")
async def claims(
    keyword: str = Query(default="", description="留空则返回全部"),
    limit: int = Query(default=50, ge=1, le=500),
    container: ServiceContainer = Depends(get_container),
    _: None = Depends(verify_api_key),
) -> dict:
    if container.claim_store is None:
        raise HTTPException(status_code=503, detail="声明库未就绪")
    if keyword.strip():
        rows = container.claim_store.find_claims_by_subject(keyword.strip(), limit=limit)
    else:
        rows = [
            claim.to_dict()
            for doc in container.claim_store.summary()["latest_versions"]
            for claim in container.claim_store.claims_of_latest(doc["doc_source"])
        ][:limit]
    return {"count": len(rows), "claims": rows}


@router.get("/stats", summary="声明库统计与抽取器运行统计")
async def stats(
    container: ServiceContainer = Depends(get_container),
    service: DriftService = Depends(get_drift_service),
    _: None = Depends(verify_api_key),
) -> dict:
    if container.claim_store is None:
        raise HTTPException(status_code=503, detail="声明库未就绪")
    return {
        "store": container.claim_store.summary(),
        "extractor": service.extractor_stats,
        "db_path": str(container.settings.guard_db_path),
    }


@router.get("/panel", include_in_schema=False)
async def panel() -> FileResponse:
    if not PANEL_HTML.exists():
        raise HTTPException(status_code=404, detail="guard.html 缺失")
    return FileResponse(str(PANEL_HTML), media_type="text/html")
