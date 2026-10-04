"""FastAPI 应用入口。

启动：
    uvicorn app.main:app --reload --port 8000
或：
    python -m app.main

交互式调试页：http://127.0.0.1:8000/  （内置轻量控制台）
接口文档：http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.api import routes_eval, routes_ingest, routes_qa, routes_system
from app.api.deps import ServiceContainer, get_container, verify_api_key
from app.config import PROJECT_ROOT, get_settings
from app.logging_conf import get_logger, setup_logging
from app.services.container import build_container

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)

CONSOLE_HTML = PROJECT_ROOT / "app" / "web" / "console.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在初始化服务……（配置文件：%s/.env）", PROJECT_ROOT)
    app.state.container = build_container(settings)

    container: ServiceContainer = app.state.container
    logger.info(
        "初始化完成：向量库=%s，BM25=%d 片段，LLM=%s，Embedding=%s",
        "就绪" if container.vector_ready else "不可用",
        container.bm25.size if container.bm25 else 0,
        container.settings.llm_model,
        container.embedder.describe() if container.embedder else "不可用",
    )
    for note in container.degraded_notes():
        logger.warning("提示：%s", note)

    try:
        yield
    finally:
        logger.info("正在关闭服务……")
        try:
            await container.llm.aclose()
        except Exception as exc:  # pragma: no cover
            logger.warning("关闭 LLM 客户端异常：%s", exc)
        embedder = container.embedder
        if embedder is not None and hasattr(embedder, "aclose"):
            try:
                await embedder.aclose()  # type: ignore[attr-defined]
            except Exception as exc:  # pragma: no cover
                logger.warning("关闭 embedding 客户端异常：%s", exc)


app = FastAPI(
    title="AutoRAG · 汽车售后知识库问答系统",
    description=(
        "一个可运行的 RAG 后端示例：文档切分 -> 向量化(Chroma) -> 向量+BM25 混合检索 -> "
        "LLM 生成带引用的回答。\n\n"
        "所有接口都在 /api/v1 前缀下；除 /api/v1/health 外，若 .env 中配置了 API_KEY，"
        "请求需要携带 X-API-Key 请求头。"
    ),
    version=routes_system.API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_prefix = "/api/v1"
app.include_router(routes_system.router, prefix=api_prefix)
app.include_router(routes_ingest.router, prefix=api_prefix)
app.include_router(routes_qa.router, prefix=api_prefix)
app.include_router(routes_eval.router, prefix=api_prefix)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底异常处理：返回可读错误，同时把堆栈写进日志，方便你贴给我排错。"""
    logger.exception("未处理异常 %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": f"服务内部错误：{type(exc).__name__}: {exc}",
            "path": request.url.path,
            "hint": "完整堆栈已打印在服务端控制台日志中。",
        },
    )


@app.get("/", include_in_schema=False)
async def console() -> FileResponse:
    """内置调试控制台（纯静态 HTML，方便手动验证接口）。"""
    if not CONSOLE_HTML.exists():  # pragma: no cover
        raise HTTPException(status_code=404, detail="console.html 缺失")
    return FileResponse(str(CONSOLE_HTML), media_type="text/html")


@app.get(f"{api_prefix}/ask", tags=["qa"], summary="GET 版问答（便于浏览器/curl 直接试）")
async def ask_get(
    q: str = Query(..., min_length=1, max_length=2000, description="问题"),
    top_k: int | None = Query(default=None, ge=1, le=20),
    debug: bool = Query(default=False, description="是否返回召回明细"),
    container: ServiceContainer = Depends(get_container),
    _: None = Depends(verify_api_key),
) -> dict:
    if not container.qa_ready or container.qa is None:
        raise HTTPException(
            status_code=503,
            detail="问答链路未就绪：" + "；".join(container.init_errors or ["初始化失败"]),
        )
    outcome = await container.qa.answer(q, top_k=top_k, debug=debug)
    outcome.pop("retrieved_raw", None)
    return outcome


def main() -> int:
    """python -m app.main 的直接入口。"""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
