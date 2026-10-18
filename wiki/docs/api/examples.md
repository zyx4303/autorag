---
sources:
  - app/api/routes_qa.py
  - app/api/routes_ingest.py
  - app/api/routes_system.py
---
# 调用示例

## `/api/v1/chat` 请求 / 响应

**请求**：

```json
{
  "question": "首保应该在多少公里或多长时间内进行？",
  "top_k": 5,
  "retrieval_mode": "hybrid",
  "debug": true
}
```

```bash
curl -X POST http://127.0.0.1:8000/api/v1/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"更换机油后需要复位保养提醒吗？\",\"debug\":true}"
```

!!! note "跨平台"
    Linux/macOS 把 `^` 换成 `\`，并把内层引号改为单引号。

**响应**（字段结构如下；具体内容取决于你的知识库与模型）：

```json
{
  "question": "首保应该在多少公里或多长时间内进行？",
  "answer": "……（模型输出，每处结论带 [1] 这类角标）……\n\n依据：\n[1] 保养周期.md · 保养 > 首保",
  "citations": [
    {
      "index": 1,
      "chunk_id": "doc_xxxxxxxxxx::3",
      "source": "保养周期.md",
      "title": "保养周期",
      "section": "保养 > 首保",
      "position": 3,
      "score": 0.0,
      "snippet": "……"
    }
  ],
  "retrieved": [],
  "refused": false,
  "retrieval_mode": "hybrid",
  "model": "deepseek-chat",
  "latency": { "retrieve_ms": 0, "generate_ms": 0, "total_ms": 0 },
  "usage": { "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0 },
  "warnings": []
}
```

`warnings` 值得关注，它会在这些情况下出现（服务端如实报告，不掩盖）：

- 模型给出的 `[n]` 超出了实际资料编号范围 → `引用编号越界`；
- 模型一个角标都没打 → 提示可能未遵循引用格式，`citations` 会退化为全部召回来源；
- 检索为空但 `REFUSE_WHEN_EMPTY=false` → 提示回答可能无依据；
- `EMBEDDING_PROVIDER=hash` → 提示当前不具备真正的语义检索能力。

---

## 流式调用（Python）

```python
import json
import httpx

with httpx.stream(
    "POST",
    "http://127.0.0.1:8000/api/v1/chat/stream",
    json={"question": "刹车片磨损到多少毫米需要更换？"},
    timeout=120.0,
) as response:
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            payload = json.loads(line.split(":", 1)[1].strip())
            if event == "meta":
                print("引用来源：", [c["source"] for c in payload["citations"]])
            elif event == "token":
                print(payload["text"], end="", flush=True)
            elif event == "done":
                print("\n", payload)
```

!!! warning "流式模式不做越界校验"
    流式模式**不做**服务端引用越界校验（回答还在生成中）。
    前端拼接完成后如需校验请改用 `POST /api/v1/chat`。这一点在 `done` 事件里也会明确提示。

---

## 只检索，不花 token

调试召回质量时用这个，**不调用大模型**：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/retrieve ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"虚构车型A多久换一次机油\"}"
```

返回结果里同时带 `vector_score` 与 `keyword_score` 双通道标记。

---

## Agent 问答

```bash
# 一次性返回（含完整 tool_trace）
curl -X POST http://127.0.0.1:8000/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"P0195 故障码什么意思"}'

# SSE 流式：能看到 agent 决策与每个工具的入参出参
curl -N -X POST http://127.0.0.1:8000/api/v1/agent/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"demo","message":"我该保养了吗"}'
```

SSE 事件序列（真实抓取）：

```text
start -> node(agent 决定调工具) -> tool_start -> tool_result
      -> node(agent) -> node(tools) -> ... -> answer -> done
```

`tool_trace` 字段示例（这是"透明性"的落点，也是以后做 Agent 评测的数据基础）：

```json
{"step": 1, "tool": "lookup_dtc", "arguments": {"dtc_code": "P0195"},
 "ok": true, "duration_ms": 0, "result_count": null, "error": ""}
```

更多 Agent 细节见 [LangGraph Agent](../modules/agent.md)。
