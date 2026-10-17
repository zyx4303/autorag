# Docker 部署

## 三步跑起来

```bash
# 1) 准备 .env（同完整安装与配置第二步）
copy .env.example .env

# 2) 构建并启动
docker compose up --build

# 3) 验证
curl http://127.0.0.1:8000/api/v1/health
```

---

## 镜像与编排要点

| 要点 | 说明 |
| --- | --- |
| **数据持久化** | `data/` 与 `eval/` 挂载到宿主机，向量库与报告不会随容器销毁丢失 |
| **路径覆盖** | `docker-compose.yml` 里把路径统一覆盖为容器内 `/app/data/*` |
| **非 root 运行** | 容器以非 root 用户（uid `10001`）运行，入口用 `tini` 转发信号 |
| **健康检查** | 镜像内置 `HEALTHCHECK`，探测 `/api/v1/health`（**不**探测 LLM，避免额外 token 消耗） |
| **本地向量模型** | 使用 `EMBEDDING_PROVIDER=local` 时体积会显著变大，需要打开 `Dockerfile` 里被注释的 `build-essential`，或干脆改用 `api` 模式 |

---

## 只跑单容器

```bash
docker build -t autorag:0.1.0 .
docker run --rm -p 8000:8000 --env-file .env -v "%cd%/data:/app/data" autorag:0.1.0
```

---

## Agent 相关

Agent 依赖 `langgraph` 与 `aiosqlite`，已在 `requirements.txt` 里。
业务工具知识库（故障码 / 保养周期）会在启动时自动建立，也可以手动重建：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/agent/kb/rebuild?force=true"
```

详见 [LangGraph Agent](../modules/agent.md)。

---

!!! note "部署到公网的注意事项"
    若把本项目暴露到公网，请务必设置 `API_KEY`、限制 `CORS_ORIGINS`，
    并在前面加一层网关做鉴权与限流。
