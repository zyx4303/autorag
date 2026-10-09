"""Agent 模块：LangGraph 状态图 + 工具集 + 业务数据层。

模块划分：
  kb.py      业务工具的结构化数据层（故障码 / 保养周期，SQLite）
  tools.py   三个业务工具 + ask_user 的 JSON schema 与实现
  graph.py   LangGraph 状态图（agent / tools / finalize 三个节点 + 条件路由）
  service.py Agent 服务封装（对接 FastAPI，含 SSE 流式）
"""
