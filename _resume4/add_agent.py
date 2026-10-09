# -*- coding: utf-8 -*-
"""把 LangGraph Agent 升级写进简历（作为 AutoRAG 的第二阶段能力）。

策略：不新增项目条目，而是把 AutoRAG 原有 3 条要点重新分配为：
  ① 架构 + 检索排错（合并原第 1、2 条的核心事实）
  ② LangGraph Agent（新增，含 3 个实测缺陷与停止条件）
  ③ 工程与交付（原第 3 条改写，吸收原第 1、2 条里的部署/测试事实）
这样事实一条不丢，同时给出"从固定 pipeline 演进到 Agent"的技术纵深。
"""
from __future__ import annotations

import sys
from pathlib import Path

import docx

REPLACEMENTS = {
    # ① 架构 + 检索排错
    "·主导 RAG 问答系统全链路落地": (
        "·主导 RAG 问答系统全链路落地：负责架构设计、API 接口定义与代码评审，"
        "完成「结构感知切分 → 向量化入库 → 混合检索 → 带引用生成」闭环；"
        "检索采用「向量召回 + 自研轻量级 BM25（纯 Python，避免引入重型 ES）」双路并行，"
        "RRF 融合后多信号重排。编写 46 项纯逻辑测试用例，"
        "实测定位并修复表格切分语义稀释、中文虚词噪声、融合归一化缺失等 4 个检索缺陷。"
    ),
    # ② LangGraph Agent（新增）
    "·检索策略与排错": (
        "·LangGraph Agent 升级：在固定流水线之上用显式 StateGraph（agent / tools / finalize "
        "三节点 + 条件路由）实现自主工具调用，定义 search_kb、lookup_dtc、compute_maintenance、"
        "ask_user 四个工具并写明适用边界；设迭代上限 6 次与「信息不足先追问」两条停止条件，"
        "用 AsyncSqliteSaver 持久化对话状态以支持多轮追问续接，并提供 SSE 流式回传每步入参出参。"
        "过程中定位并修复三个实现缺陷：同步 Saver 与异步图不兼容、add_messages 转换丢失 "
        "tool_call_id 导致 422、节点 yield 的事件被当作状态增量丢弃（改用 StreamWriter）；"
        "新增 79 项确定性行为测试（覆盖工具链、追问、迭代上限、工具报错换策略），"
        "并如实记录「提示词对工具调用次数约束有限」这一未解决收敛问题。"
    ),
    # ③ 工程与交付（吸收原要点里的部署与开源事实）
    "·工程与交付": (
        "·工程与交付：Docker 一键启动与增量入库脚本；代码在 GitHub 开源（MIT），"
        "并如实标注 AI 辅助开发边界与未解决的失败样例。"
    ),
}


def rewrite(para, new_text: str) -> None:
    runs = para.runs
    if not runs:
        para.add_run(new_text)
        return
    runs[0].text = new_text
    for extra in runs[1:]:
        extra._element.getparent().remove(extra._element)


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    doc = docx.Document(str(src))

    # 只处理 AutoRAG 标题之后、FreshGuard 标题之前的要点段落
    inside = False
    handled = 0
    for para in doc.paragraphs:
        text = para.text.strip()
        if text.startswith("AutoRAG ·"):
            inside = True
            continue
        if text.startswith("FreshGuard ·"):
            inside = False
        if not inside or not text.startswith("·"):
            continue
        for prefix, replacement in REPLACEMENTS.items():
            if text.startswith(prefix):
                rewrite(para, replacement)
                handled += 1
                print(f"  已改写: {prefix}...")
                break

    doc.save(str(dst))
    print(f"共改写 {handled} 条要点 -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
