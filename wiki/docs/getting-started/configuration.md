---
sources:
  - app/config.py
  - .env.example
---
# 环境变量清单

复制 `.env.example` → `.env` 后按需修改。`[待补充]` 表示**必须由你填写**的项。

## 对话模型（LLM）

| 变量 | 默认值 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `LLM_API_KEY` | `[待补充]` | **是**（问答功能） | 对话模型密钥 |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | 是 | OpenAI 兼容地址，需含 `/v1` |
| `LLM_MODEL` | `deepseek-chat` | 是 | 模型名 |
| `LLM_TIMEOUT_SECONDS` | `60` | 否 | 单次请求超时 |
| `LLM_MAX_RETRIES` | `3` | 否 | 失败重试次数（429/5xx/网络错误才重试） |
| `LLM_TEMPERATURE` | `0.2` | 否 | 建议保持低温度，减少自由发挥 |
| `LLM_MAX_TOKENS` | `1024` | 否 | 单次生成上限 |

## 向量模型（Embedding）

| 变量 | 默认值 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `EMBEDDING_PROVIDER` | `hash` | 是 | `api` / `local` / `hash`，正式使用请选 `api` |
| `EMBEDDING_API_KEY` | `[待补充]` | provider=api 时必填 | 向量模型密钥 |
| `EMBEDDING_BASE_URL` | `[待补充]` | provider=api 时必填 | 向量接口地址（含 `/v1`） |
| `EMBEDDING_MODEL` | `[待补充]` | provider=api 时必填 | 向量模型名 |
| `EMBEDDING_DIM` | `[待补充]` | provider=api 时必填 | 向量维度，须与模型一致 |
| `LOCAL_EMBEDDING_MODEL` | `[待补充]` | provider=local 时必填 | 本地模型名或本地目录路径 |
| `LOCAL_EMBEDDING_DIM` | `[待补充]` | 否 | 留空则由模型自动推断 |
| `HASH_EMBEDDING_DIM` | `1024` | 否 | hash 模式向量维度 |

## 路径配置

| 变量 | 默认值 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `DOCUMENTS_DIR` | `./data/documents` | 否 | 知识库目录（相对项目根目录解析） |
| `CHROMA_DIR` | `./data/chroma` | 否 | 向量库落盘目录 |
| `UPLOAD_DIR` | `./data/uploads` | 否 | 上传文件目录 |
| `REGISTRY_PATH` | `./data/registry.json` | 否 | 增量索引记录 |
| `BM25_INDEX_PATH` | `./data/bm25_index.json` | 否 | BM25 索引落盘位置 |
| `CHROMA_COLLECTION` | `auto_after_sales` | 否 | 集合名 |

## 切分配置

| 变量 | 默认值 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `CHUNK_SIZE` | `600` | 否 | 目标片段字符数 |
| `CHUNK_OVERLAP` | `80` | 否 | 相邻片段重叠字符数（代码内保证 < CHUNK_SIZE） |
| `CHUNK_MIN_CHARS` | `80` | 否 | 小于该长度的碎块并入前一块 |

## 检索配置

| 变量 | 默认值 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `RETRIEVAL_MODE` | `hybrid` | 否 | `vector` / `keyword` / `hybrid` |
| `RRF_K` | `60` | 否 | RRF 融合常数 |
| `VECTOR_WEIGHT` | `1.0` | 否 | 融合中向量通道权重 |
| `KEYWORD_WEIGHT` | `0.8` | 否 | 融合中 BM25 通道权重 |
| `TOP_K` | `5` | 否 | 最终喂给模型的片段数 |
| `CANDIDATE_K` | `20` | 否 | 每路召回候选数 |
| `MIN_SCORE` | `0.05` | 否 | 仅 `RETRIEVAL_MODE=vector` 时生效的拒答阈值 |
| `REFUSE_WHEN_EMPTY` | `true` | 否 | 检索为空时是否跳过 LLM 直接兜底 |

## 语义精排（cross-encoder）

| 变量 | 默认值 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `RERANK_ENABLED` | `false` | 否 | 是否启用精排（需先下载模型权重） |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | 否 | 模型名或本地目录。**中文建议 `bge-reranker-v2-m3`**，base 版区分度不足 |
| `RERANK_TOP_N` | `20` | 否 | 只精排前 N 个候选；小于 `TOP_K` 会被自动纠正 |
| `RERANK_DEVICE` | `cpu` | 否 | `cpu` / `cuda` |
| `RERANK_MAX_LENGTH` | `512` | 否 | 截断长度，直接决定 CPU 耗时（192/256/384/512 实测每个 20 条约 2.8/3.3/5.0/6.8s）|
| `RERANK_BATCH_SIZE` | `16` | 否 | 批大小 |

## 归因校验与 Agent 自检

| 变量 | 默认值 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `ATTRIBUTION_ENABLED` | `true` | 否 | 逐句归因校验（防幻觉） |
| `ATTRIBUTION_WEAK_THRESHOLD` | `0.35` | 否 | 覆盖率 ≥ 此值判为"有依据" |
| `ATTRIBUTION_UNSUPPORTED_THRESHOLD` | `0.12` | 否 | 覆盖率低于此值判为"疑似无依据" |
| `AGENT_REFLECTION_ENABLED` | `true` | 否 | Agent 答完后是否自检 |
| `AGENT_MAX_REFLECTIONS` | `1` | 否 | 最多重答轮数（调大会成倍增加延迟与 token）|

## 服务与日志

| 变量 | 默认值 | 是否必填 | 说明 |
| --- | --- | --- | --- |
| `API_HOST` / `API_PORT` | `0.0.0.0` / `8000` | 否 | 监听地址与端口 |
| `API_KEY` | 空 | 否 | 填了才启用 `X-API-Key` 校验 |
| `CORS_ORIGINS` | `*` | 否 | 允许的跨域来源，逗号分隔 |
| `MAX_UPLOAD_MB` | `20` | 否 | 上传大小上限 |
| `LOG_LEVEL` | `INFO` | 否 | 日志级别 |

---

!!! tip "改完配置先跑自检"
    ```bash
    python scripts/preflight.py
    ```

    它会一次性检查：Python 与依赖版本、`.env` 里哪些项还是 `[待补充]`、21 个模块能否导入、
    embedding 接口的**真实维度**与 `EMBEDDING_DIM` 是否一致、
    **库里存的向量维度与当前配置是否一致**（专门用来发现"换了 embedding 却没重建"）等。
    输出**不含任何密钥**，可以直接整段贴出来求助。

!!! warning "关于密钥安全"
    密钥只从 `.env` / 环境变量读取，日志与接口响应**从不回显密钥明文**
    （`/health` 与 `/stats` 只返回"是否已配置"）。
    不要提交 `.env`（`.gitignore` 已忽略）。
