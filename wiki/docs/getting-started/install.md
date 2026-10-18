---
sources:
  - requirements.txt
  - requirements-local.txt
  - .env.example
  - scripts/preflight.py
---
# 完整安装与配置

!!! tip "已经跑通检索链路了？"
    如果你已经按 [5 分钟跑通检索链路](quickstart.md) 验证通过，可以直接跳到
    [第二步：填 Key，切换成真正的语义检索](#第二步填-key切换成真正的语义检索) 配置对话模型。
    本节其余内容是给第一次接触本项目的人准备的完整说明。

## 环境要求

- Python **3.10+**（开发时按 3.11 编写；容器镜像用 `python:3.11-slim`）
- 可访问外网（下载依赖、调用 LLM API）
- **不需要本地 GPU**

---

## 第一步：跑通检索链路（不填任何 Key）

安装、自检与入库三步见 [快速开始](quickstart.md)。这里补充**四步操作流程**，便于按需查阅。

### 放入你的知识库文档

把文档放进 `data/documents/`（可用子目录，来源标识为相对路径，例如 `保养/周期表.md`），然后：

```bash
python scripts/ingest.py            # 增量入库
python scripts/ingest.py --rebuild  # 全量重建
```

---

## 第二步：填 Key，切换成真正的语义检索

编辑 `.env`：

```dotenv
# 对话模型（示例为 DeepSeek，其它 OpenAI 兼容服务同理）
LLM_API_KEY=[待补充]                 # 换成你的真实 Key
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# 向量模型：改掉 provider，并填写三要素 + 维度
EMBEDDING_PROVIDER=api
EMBEDDING_API_KEY=[待补充]
EMBEDDING_BASE_URL=[待补充]          # 形如 https://xxx/v1
EMBEDDING_MODEL=[待补充]             # 形如 text-embedding-3-small / bge-m3
EMBEDDING_DIM=[待补充]               # 与模型输出维度一致，例如 1536 / 1024
```

!!! danger "改完向量模型必须重建向量库"
    新旧向量不在同一空间，**必须**重建：

    ```bash
    python scripts/ingest.py --rebuild
    ```

    忘了重建的典型症状是 `向量维度(...)与 embedder 声明维度(...)不一致`，
    自检脚本 `scripts/preflight.py` 会专门帮你发现这种情况。

---

## 第三步：提问题

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"更换机油后需要复位保养提醒吗？\",\"debug\":true}"
```

!!! note "跨平台差异"
    Linux/macOS 把 `^` 换成 `\`，并把内层引号改为单引号。

接入真实文档后如果召回不准，可以对照 [实测记录](../evaluation/results.md) 里第二轮、第三轮的排查过程，
那里记录了从「症状 → 根因 → 修复」的完整链路。

---

## 环境变量清单

详见独立的 [环境变量清单](configuration.md) —— 覆盖 LLM、Embedding、切分、检索、服务共 6 组约 35 个变量。

## 可选依赖

```bash
pip install pypdf                        # 解析 PDF
pip install python-docx                  # 解析 DOCX
pip install -r requirements-local.txt     # 本地向量模型（会拉取 torch，体积较大）
```

未安装对应依赖时，上传/入库该类文件会返回**明确的错误信息 + 安装命令**，不会静默跳过。

本地向量模型（`EMBEDDING_PROVIDER=local`）有三个实战坑已经写进代码，
详见 [Embedding 三种模式](../architecture/embedding.md)。
