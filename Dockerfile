# =============================================================================
# AutoRAG 运行镜像
# 构建： docker build -t autorag:0.1.0 .
# 运行： docker run --rm -p 8000:8000 --env-file .env -v "%cd%/data:/app/data" autorag:0.1.0
# =============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 说明：
# - tini 用于正确转发信号，保证 Ctrl+C / docker stop 能优雅退出；
# - 若使用 EMBEDDING_PROVIDER=local，sentence-transformers 会需要编译工具链，
#   届时把 build-essential 补进下面的 apt-get install 列表，或改用 EMBEDDING_PROVIDER=api。
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini curl \
    && rm -rf /var/lib/apt/lists/*

# 先装依赖，利用 Docker 层缓存
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 再拷代码
COPY app ./app
COPY eval ./eval
COPY scripts ./scripts
COPY tests ./tests
COPY README.md ./
COPY .env.example ./
COPY .env.api.example ./

# 运行时数据目录（会被 docker-compose 的 volume 覆盖）
RUN mkdir -p /app/data/documents /app/data/chroma /app/data/uploads \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 容器内自检：只要 /health 能返回就算进程健康（不探测 LLM，避免额外 token 消耗）
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/v1/health || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
