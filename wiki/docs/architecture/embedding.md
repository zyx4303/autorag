---
sources:
  - app/core/embedding.py
---
# Embedding 三种模式

对应代码：`app/core/embedding.py`

同一个接口下有三种实现，用 `EMBEDDING_PROVIDER` 切换。
设计意图是：**让不了解检索的人也能零成本先跑通链路**，再按需升级。

---

## 三种模式对比

| 模式 | 是否需要 Key | 是否需要下载模型 | 是否有真正语义检索能力 | 适用场景 |
| --- | --- | --- | --- | --- |
| `hash` | 否 | 否 | **否** | 零依赖验证链路是否通 |
| `api` | 是 | 否 | 是 | **正式使用（推荐）** |
| `local` | 否 | 是（约 100 秒首次下载） | 是 | 离线 / 不想付费 / 数据不出本机 |

!!! warning "`hash` 模式会明确暴露自己的局限"
    使用 `hash` 时，`/health` 与每次问答的 `warnings` 里都会提示
    **当前不具备真正的语义检索能力**。不隐藏降级，是这个项目的既定原则。

---

## `api` 模式（推荐）

任何 OpenAI 兼容的向量服务：

```dotenv
EMBEDDING_PROVIDER=api
EMBEDDING_API_KEY=[待补充]
EMBEDDING_BASE_URL=https://xxx/v1
EMBEDDING_MODEL=bge-m3
EMBEDDING_DIM=1024
```

!!! danger "`EMBEDDING_DIM` 必须与模型真实输出维度一致"
    填错的症状是入库或检索时报维度不一致。
    自检脚本会**真实调用一次** embedding 接口，拿到实际维度与配置比对：

    ```bash
    python scripts/preflight.py
    ```

---

## `local` 模式与三个实战坑

这条路走通了，但踩过三个坑，**都已写进代码**：

```dotenv
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
LOCAL_EMBEDDING_DIM=512           # 实测该模型输出 512 维
HF_ENDPOINT=https://hf-mirror.com # 国内镜像；留空时代码会自动探测并兜底
HF_HUB_OFFLINE=1                  # 权重下载完成后置 1，改为纯本地加载
```

| 坑 | 现象 | 代码里的处理 |
| --- | --- | --- |
| hosts 把 huggingface.co 指向 127.0.0.1 | 报 `CERTIFICATE_VERIFY_FAILED`，看起来像网络不通 | `_huggingface_host_is_blackholed()` 检测到就自动切镜像并打日志 |
| `HF_ENDPOINT` 写进 `.env` 却不生效 | 配了镜像仍走 huggingface.co | `.env` 的值不会自动进 `os.environ`，且 `huggingface_hub` 在 import 时把端点固化进常量；现在会同时写环境变量 + 覆盖 `huggingface_hub.constants.ENDPOINT` |
| 部分库（实测 `transformers` 探测 `adapter_config.json`）绕过镜像 | 每次加载重试十几秒并刷一屏 SSL 报错 | `HF_HUB_OFFLINE=1` 时用 `local_files_only=True` 纯本地加载 |

---

## BGE 查询指令前缀

`bge` 系列官方要求**查询侧加指令前缀**：

```text
为这个句子生成表示以用于检索相关文章：
```

代码会**按模型名自动加，文档侧不加**（这是 BGE 的规范用法，加错侧反而降效果）。

实测在 3 条文本的样本上前缀让"正确答案/干扰项"的相似度比值从 1.45 提升到 1.55。

!!! note "样本太小，不足以作为效果证据"
    排序没有变化。记录在此只是说明"按规范做了"，不是效果声明。
    如需关闭：`BGE_QUERY_INSTRUCTION=none`。

---

## 切换 embedding 后必做

**改完必须重建向量库**，因为新旧向量不在同一空间：

```bash
python scripts/ingest.py --rebuild
```

忘了重建的典型症状：

```text
向量维度(...)与 embedder 声明维度(...)不一致
```

`scripts/preflight.py` 会专门检查**库里存的向量维度与当前配置是否一致**，
就是用来发现这种情况的。

---

## 性能预期

首次下载大约需要 100 秒左右（取决于网速），之后本地 CPU 编码一般每批百毫秒级。

!!! warning "以你自己的实测为准"
    以上为开发机观察值，不是性能基准。你的机器、模型、文档规模不同，结果会不同。
