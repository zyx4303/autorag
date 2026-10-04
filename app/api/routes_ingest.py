"""入库接口：目录扫描入库 / 文件上传。

安全与健壮性处理：
- 上传文件名做 basename 清洗，禁止 ../ 路径穿越；
- 后缀白名单校验 + 大小上限（MAX_UPLOAD_MB）；
- 目录扫描入库是幂等的：未变更文件按 checksum 跳过。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile, status

from app.api.deps import ServiceContainer, get_container, get_ingestion, verify_api_key
from app.core.loader import SUPPORTED_SUFFIXES
from app.logging_conf import get_logger
from app.models import IngestRequest, IngestResponse, UploadResponse
from app.services.ingestion_service import IngestionService

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse, summary="扫描 data/documents 并增量入库")
async def ingest(
    payload: IngestRequest = Body(default_factory=IngestRequest),
    ingestion: IngestionService = Depends(get_ingestion),
    _: None = Depends(verify_api_key),
) -> IngestResponse:
    """把知识库目录里的文档切分、向量化并写入 Chroma，然后重建 BM25 索引。

    - 首次使用传 `{"rebuild": true}` 可以从零干净重建；
    - 日常增量直接传 `{}`，未变更文件会返回 status=skipped；
    - 只调试某几个文件时用 `{"paths": ["保养手册.md"]}`。
    """
    return await ingestion.ingest(
        rebuild=payload.rebuild,
        reset_registry=payload.reset_registry,
        only_sources=payload.paths,
    )


@router.post("/upload", response_model=UploadResponse, summary="上传单个文档到知识库目录")
async def upload(
    file: UploadFile = File(..., description="支持 .md/.markdown/.txt，可选 .pdf/.docx"),
    auto_ingest: bool = Query(default=False, description="上传后是否立即入库"),
    container: ServiceContainer = Depends(get_container),
    ingestion: IngestionService = Depends(get_ingestion),
    _: None = Depends(verify_api_key),
) -> UploadResponse:
    settings = container.settings

    raw_name = file.filename or ""
    safe_name = Path(raw_name.replace("\\", "/")).name.strip()
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="文件名不合法或为空"
        )

    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"不支持的文件类型 {suffix or '(无后缀)'}。"
                f"当前支持：{sorted(SUPPORTED_SUFFIXES)}。"
                "如需解析 PDF/DOCX，请分别安装 pypdf / python-docx（见 README 可选依赖）。"
            ),
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="上传文件为空")
    if len(payload) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"文件大小 {len(payload) / 1024 / 1024:.2f}MB 超过上限 "
                f"MAX_UPLOAD_MB={settings.max_upload_mb}"
            ),
        )

    target_dir = settings.upload_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / safe_name
    target.write_bytes(payload)
    logger.info("已保存上传文件 %s（%d 字节）", target, len(payload))

    message = (
        f"已保存到 {target}。"
        + ("已自动入库。" if auto_ingest else "调用 POST /api/v1/ingest 后才会进入向量库。")
    )

    if auto_ingest:
        # 上传文件放在 upload_dir 下，与 documents_dir 是两棵不同的目录树。
        # 这里必须把 root 传成 upload_dir 的父目录，让 source 变成 "uploads/xxx.md"：
        # 若传成 upload_dir 本身，source 会退化成裸文件名 "xxx.md"，
        # 一旦 data/documents 下有同名文件，index_document_async 里"先按 source 删除旧片段"
        # 就会把知识库里那份文档的向量误删。
        from app.core.loader import load_document

        try:
            document = load_document(target, target_dir.parent)
            registry_documents: dict = {}
            result = await ingestion.index_document_async(document, registry_documents)
            await ingestion.rebuild_bm25_async()
            message = (
                f"已保存到 {target}，并以来源 {document.source} 入库 "
                f"{result.chunks} 个片段（状态：{result.status}）。"
            )
        except Exception as exc:  # 入库失败时保留文件并如实报错
            logger.exception("上传后自动入库失败")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"文件已保存到 {target}，但自动入库失败：{type(exc).__name__}: {exc}",
            ) from exc

    return UploadResponse(
        filename=safe_name,
        saved_to=str(target),
        size_bytes=len(payload),
        message=message,
    )
