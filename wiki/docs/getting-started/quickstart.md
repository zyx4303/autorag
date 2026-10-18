---
sources:
  - scripts/preflight.py
  - scripts/ingest.py
  - app/main.py
  - app/api/routes_qa.py
  - app/services/ingestion_service.py
---
# 5 分钟跑通检索链路

不填任何 Key，先把「加载 → 切分 → 向量化 → 入库 → 混合检索」整条链路跑通。这一步**零 token 消耗**。

## 三条命令

```bash
# 1) 安装依赖（Python 3.10+）
python -m venv .venv
.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2) 准备配置并自检（默认 EMBEDDING_PROVIDER=hash，无需任何 Key）
copy .env.example .env               # Linux/macOS: cp .env.example .env
python scripts/preflight.py          # 期望：阻塞项 0 / 问题项 0
python tests/smoke_test.py           # 期望：通过 30 项 / 失败 0 项

# 3) 启动并入库示例知识库
uvicorn app.main:app --reload --port 8000
#   → 内置调试控制台： http://127.0.0.1:8000/
#   → 接口文档：       http://127.0.0.1:8000/docs
python scripts/ingest.py --rebuild
```

到这里**检索链路**（加载 → 切分 → 向量化 → 入库 → 混合检索）就验证完了。

要让它在检索结果基础上**生成答案**，只需在 `.env` 填 `LLM_API_KEY`（任何 OpenAI 兼容服务），
再按 [完整安装与配置](install.md) 里的第二步切换向量模型即可。

```bash
# 只测召回、不花 token
python scripts/run_eval.py --mode retrieve --limit 5

# 完整问答评测（消耗 token）
python scripts/run_eval.py --limit 5
```

---

## 验证清单

做完上面三步，逐项对一下：

| 验证项 | 命令 | 期望结果 |
| --- | --- | --- |
| 环境与配置 | `python scripts/preflight.py` | 阻塞项 0、问题项 0 |
| 纯逻辑正确性 | `python tests/smoke_test.py` | 通过 30 项、失败 0 项 |
| 入库 | `python scripts/ingest.py --rebuild` | `indexed=6`、`total_chunks_in_store=87` |
| 召回（零 token） | 见下方 curl | 返回结果里同时带 `vector_score` 与 `keyword_score` |
| 批量召回评测（零 token） | `python scripts/run_eval.py --mode retrieve` | 跑通并生成 JSON + Markdown 报告 |

```bash
# 只看向量召回与 BM25 融合结果，不调用大模型、不消耗 token
curl -X POST http://127.0.0.1:8000/api/v1/retrieve ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"虚构车型A多久换一次机油\"}"
```

!!! note "`llm_ready=false` 是预期行为"
    此时 `/api/v1/health` 会显示 `llm_ready=false`（因为 `LLM_API_KEY` 还是 `[待补充]`）。
    健康检查会**如实报告**未配置，而不是假装正常。

---

## Python 版本注意事项（实测踩过的坑）

`chromadb` 0.5.x 依赖 `chroma-hnswlib`，而它**只发布了到 cp311 的 Windows wheel**，
所以在 Python 3.12 / 3.13 上会退化成源码编译，并报：

```text
Microsoft Visual C++ 14.0 or greater is required.
```

本项目因此在 `requirements.txt` 里固定 `chromadb>=1.0,<2.0`（提供 cp39-abi3 预编译 wheel，
Python 3.9+ 直接可装），并已在 Python 3.13.9 上实测通过。

!!! warning
    如果你必须用 `chromadb` 0.5.x，请改用 **Python 3.11** 解释器。

---

## 下一步

- 想配置对话模型和真正的语义检索 → [完整安装与配置](install.md)
- 想知道每个环境变量的含义 → [环境变量清单](configuration.md)
- 想用容器跑 → [Docker 部署](docker.md)
- 想了解检索内部怎么做的 → [混合检索与重排](../architecture/retrieval.md)
