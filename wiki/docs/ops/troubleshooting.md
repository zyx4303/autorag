---
sources:
  - scripts/preflight.py
  - app/config.py
  - app/main.py
---
# 排查手册

报错时先看这里。

---

## 先跑一次自检（最省时间的一步）

```bash
python scripts/preflight.py            # 不调用 LLM，零 token 消耗
python scripts/preflight.py --ping-llm # 额外真实探测一次 LLM，约 1 token
```

它会一次性检查：

- Python 与依赖版本（含 FastAPI 是否 ≥ 0.115）；
- `.env` 是否存在以及哪些项还是 `[待补充]`；
- 21 个模块能否导入；
- 切分 / BM25 / hash 向量 / 引用校验是否正常；
- embedding 接口的**真实维度**与 `EMBEDDING_DIM` 是否一致；
- Chroma 能否打开，以及**库里存的向量维度与当前配置是否一致**
  （专门用来发现"换了 embedding 却没重建"）；
- LLM 是否配置与能否连通；
- 知识库目录有没有文档；
- 评测集 JSON 是否合法。

!!! tip "输出不含任何密钥，可直接贴出来求助"
    输出同时打印到屏幕并写入 `data/preflight_report.txt`。

---

## 常见报错对照表

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| 启动即报 `Embedding 初始化失败：EMBEDDING_PROVIDER=api，但以下配置仍为占位符…` | `.env` 里还是 `[待补充]` | 填写 Key/地址/模型/维度，或先改回 `EMBEDDING_PROVIDER=hash` |
| `/api/v1/chat` 返回 `[生成失败] LLM_API_KEY 未配置…` | 没填对话模型 Key | 填 `LLM_API_KEY` 后重启 |
| 返回 `LLM 接口 404` | `LLM_BASE_URL` 少了 `/v1`，或模型名写错 | 核对服务商文档里的 base_url 与模型名 |
| 返回 `LLM 鉴权失败(401)` | Key 错误或额度用尽 | 换 Key / 充值 |
| `向量维度(...)与 embedder 声明维度(...)不一致` | 换了 embedding 模型但没重建 | `python scripts/ingest.py --rebuild` |
| `向量数量与片段数量不一致` / 入库部分失败 | 向量接口限流或网络抖动 | 看日志与 `/api/v1/ingest` 返回里的 `error`，调小批量或重试 |
| 入库后 `/stats` 的 `collection_count` 为 0 | 文档没放进 `DOCUMENTS_DIR`，或后缀不在白名单 | 看 `/stats` 的 `documents_on_disk` 与 `config_warnings` |
| 回答老是兜底"未找到依据" | 知识库确实没有该内容；或 `MIN_SCORE` 过高；或 `TOP_K` 太小 | 先 `POST /api/v1/retrieve` 看召回，再调参 |
| 回答里的引用编号是 `[7]` 但只有 5 条来源 | 模型编造引用 | 响应的 `warnings` 会明确指出；可降低 `LLM_TEMPERATURE` 或换更强的模型 |
| PDF 入库报 `解析 PDF 需要可选依赖` | 没装 `pypdf` | `pip install pypdf` |
| 第一次请求特别慢 | 本地向量模型首次下载权重 / Chroma 首次建索引 | 属正常现象，第二次会快；具体耗时请以你自己实测为准 |
| 端口被占用 | 8000 已被占用 | `uvicorn app.main:app --port 8001`，同步改 `.env` 的 `API_PORT` |
| 注册路由时报 `TypeError: Body() got an unexpected keyword argument 'default_factory'` | 本机 FastAPI 版本低于 0.115 | `pip install -U "fastapi>=0.115"`（`requirements.txt` 已声明该下限） |
| `POST /api/v1/ingest/upload?auto_ingest=true` 后原知识库文档消失 | 旧版本会把上传文件的 source 记成裸文件名，与 `data/documents` 同名文件相撞 | 当前代码已用 `uploads/xxx.md` 作为 source 避免撞名；若你改过这段逻辑，请确认 `load_document(target, target_dir.parent)` 的 root 参数 |
| BM25 对"完全不相关的中文问题"也返回结果 | 中文按单字 + bigram 切分时，任意两句中文都可能共享"的/是/关"这类单字 | 已加相对分数下限 `MIN_RELATIVE_SCORE`（默认 0.15，代码在 `app/core/bm25.py`）。这是分词策略的固有特性，不是 bug；想进一步降低噪声可接 rerank 模型 |
| 我在 `.env` 里配了 `CHUNK_SIZE=80`，实际却按 120 切 | 切分器里曾有一层硬编码下限，会静默覆盖用户配置 | 已移除切分器内的下限，统一由 `Settings.normalize()` 处理（低于 120 会纠正并在 `/health` 的 `config_warnings` 里提示） |
| 调用 `/api/v1/chat/stream` 后服务端日志出现 `async generator ignored GeneratorExit` | 客户端提前断开且 SSE 生成器在 `finally` 里 `yield` | 当前代码已把 `done` 事件移出 `finally`；若你自行改造过该生成器，请勿在 `finally` 中 `yield` |

---

!!! tip "求助时提供什么"
    服务端日志会打印完整堆栈。
    遇到问题时把**日志最后 30 行 + 接口原始响应**一起发出来即可定位。

---

## 相关页面

- [测试与自检](testing.md) —— 有哪些自动化验证可以一键跑
- [混合检索与重排](../architecture/retrieval.md) —— 召回不准时的调参对象
- [实测记录](../evaluation/results.md) —— 曾遇到过的真实问题与修复过程
