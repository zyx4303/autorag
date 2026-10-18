---
sources:
  - app/api/routes_qa.py
  - app/api/routes_ingest.py
  - app/api/routes_system.py
  - app/api/routes_eval.py
  - app/api/routes_agent.py
  - app/api/routes_guard.py
  - app/models.py
---
# 接口一览

所有业务接口前缀 `/api/v1`。

!!! note "鉴权"
    若 `.env` 里配置了 `API_KEY`，除 `/health` 外都需要请求头 `X-API-Key: <你的值>`；
    不配置则不需要鉴权（本地开发默认如此）。

---

## 系统与知识库

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/health` | 健康检查。`?probe_llm=true` 会真实调用一次 LLM（`max_tokens=1`） |
| GET | `/api/v1/stats` | 知识库统计：片段数、已索引文档数、BM25 规模、配置告警 |
| GET | `/api/v1/sources` | 列出向量库中已有的来源文件 |
| GET | `/api/v1/version` | 版本与配置指纹 |
| DELETE | `/api/v1/collection` | 清空向量库与 BM25 索引（**不会**删除原始文档） |
| POST | `/api/v1/ingest` | 扫描 `data/documents` 增量入库；`{"rebuild":true}` 全量重建 |
| POST | `/api/v1/ingest/upload` | 上传单个文档；`?auto_ingest=true` 时立即入库 |

## 检索与问答

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/retrieve` | **只检索**，不调用大模型（用于单独评估召回质量） |
| POST | `/api/v1/chat` | 完整问答，返回 `answer` + `citations` + `latency` + `usage` |
| POST | `/api/v1/chat/stream` | SSE 流式问答：先 `meta` 事件（含引用），再 `token`，最后 `done` |
| GET | `/api/v1/ask` | GET 版问答，便于浏览器或 curl 直接试 |
| GET | `/api/v1/chunks/preview` | 预览向量库中的少量片段（人工抽查切分质量） |

## 评测

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/eval/dataset` | 查看评测集概况与待补充字段 |
| POST | `/api/v1/eval/run` | 运行评测集（**会消耗 token**），报告落盘到 `eval/results/` |
| GET | `/api/v1/eval/reports` | 列出历史评测报告 |

## Agent（LangGraph）

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/v1/agent/chat` | Agent 问答（自主调用工具），返回 `answer` + `citations` + `tool_trace` |
| POST | `/api/v1/agent/chat/stream` | Agent 问答 SSE 流式：推送节点决策与每个工具的入参出参 |
| GET | `/api/v1/agent/tools` | 列出四个工具及完整 JSON schema |
| GET | `/api/v1/agent/status` | Agent 状态：可用工具、迭代上限、业务库规模 |
| POST | `/api/v1/agent/kb/rebuild` | 从 `data/documents` 重建业务工具知识库（故障码 / 保养周期） |

详见 [LangGraph Agent](../modules/agent.md)。

## FreshGuard

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/v1/guard/panel` | FreshGuard 变更审计面板（HTML，也可用 `http://127.0.0.1:8000/guard`） |
| POST | `/api/v1/guard/extract?source=…` | 对单个文档做增量抽取（`force=true` 则全量重抽） |
| POST | `/api/v1/guard/scan` | 扫描知识库并对所有文档增量抽取（**首次会消耗较多 token**） |
| GET | `/api/v1/guard/versions` | 各文档的最新版本与声明数 |
| GET | `/api/v1/guard/diff?source=…` | 某文档最近两个版本的失效清单（按严重度排序） |
| GET | `/api/v1/guard/diff/all` | 汇总所有文档的高严重度变更 |
| GET | `/api/v1/guard/affects?keyword=…` | 影响面反查：某事实在库内被哪些文档引用 |
| GET | `/api/v1/guard/claims?keyword=…` | 浏览当前库中的声明 |
| GET | `/api/v1/guard/stats` | 声明库统计与抽取器运行统计 |

详见 [FreshGuard 变更影响分析](../modules/freshguard.md)。

---

## 内置页面

| 路径 | 说明 |
| --- | --- |
| `http://127.0.0.1:8000/` | 内置调试控制台（零依赖单页） |
| `http://127.0.0.1:8000/guard` | 变更审计面板 |
| `http://127.0.0.1:8000/docs` | 自动生成的接口文档（OpenAPI） |

---

请求 / 响应示例与流式调用代码见 [调用示例](examples.md)。
