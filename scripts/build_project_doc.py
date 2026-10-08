#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成《AutoRAG 项目说明文档》HTML（供浏览器打印成 PDF）。

设计要点：
- 只使用真实数据：模块行数、接口数量、知识库规模均从代码与本地文件实时统计；
- 只引用真实代码：文档里的代码片段由本脚本从源码文件中按行号提取，不手写；
- 只使用真实截图：运行效果图取自 docs/images/（由 Edge headless 实测截图）；
- A4 打印友好：page-break 控制分页、页边距、字体回退、避免代码块跨页断裂。

用法：
    python scripts/build_project_doc.py            # 输出到 resume/项目说明文档.html
然后由 scripts/build_project_doc.py 的同级调用方用 Edge --print-to-pdf 转换。
"""
from __future__ import annotations

import base64
import html
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------------------------------- #
# 真实代码片段：从源码按行号提取，避免手写失真
# --------------------------------------------------------------------------- #

# (标题, 文件, 起始行, 结束行, 说明)
CODE_EXCERPTS = [
    (
        "表格识别：把 Markdown 表格当原子块",
        "app/core/chunker.py", 72, 86,
        "表格行占比 ≥60% 即判定为表格块。这个判断让表格不再被当成普通段落被打包，"
        "是修复「故障码定义被稀释」缺陷的第一步。",
    ),
    (
        "表格与段落分离打包 + 重叠保护",
        "app/core/chunker.py", 135, 168,
        "表格独立成块、不与段落混合；当前块以表格开头时不做重叠，避免产出行内混排的非法表格。",
    ),
    (
        "长表格按行切分并逐片重复表头（片段一）",
        "app/core/chunker.py", 108, 120,
        "超长表按行切开，但每片都补回表头，保住「故障码 | 含义」这层列语义。",
    ),
    (
        "长表格按行切分并逐片重复表头（片段二）",
        "app/core/chunker.py", 121, 133,
        "打包循环：表头先入缓冲，逐行追加，超出块大小则落盘并重新起头。",
    ),
    (
        "中文单字降权：对抗虚词噪声",
        "app/core/bm25.py", 37, 47,
        "单个中文字符按 0.3 权重计分。这是修复「P0300 召回保修条款」缺陷的关键。",
    ),
    (
        "查询词组整体命中加成（片段一）",
        "app/core/bm25.py", 197, 218,
        "查询被空格/标点切出的词组若整体出现在片段文本中，给该片段乘系数。",
    ),
    (
        "查询词组整体命中加成（片段二）",
        "app/core/bm25.py", 219, 235,
        "先加分再按相对下限过滤，避免把已加分的正确候选一起滤掉。",
    ),
    (
        "四信号重排：融合分 / 词覆盖 / 高精度串 / 章节命中（片段一）",
        "app/core/retriever.py", 140, 168,
        "先算出中文词组命中比例（跨词边界的兜底匹配）。",
    ),
    (
        "四信号重排（片段二）",
        "app/core/retriever.py", 169, 190,
        "高精度 token 命中比例：P0420 这类串命中一条就该被明显提权。",
    ),
    (
        "四信号重排（片段三）",
        "app/core/retriever.py", 191, 205,
        "章节标题命中 + 四类信号加权求和。",
    ),
    (
        "融合时向量分支的归一化（修复排序倒挂）",
        "app/core/retriever.py", 234, 258,
        "命名写着「归一化名次分」，但向量分支曾漏了归一化，导致余弦绝对值压过名次信息。",
    ),
    (
        "服务端引用越界校验",
        "app/services/rag_pipeline.py", 138, 168,
        "把回答里的 [n] 与实际注入的资料编号比对，越界写入 warnings——只做编号级校验，不做语义级。",
    ),
    (
        "无依据时直接兜底，不调用大模型",
        "app/services/rag_pipeline.py", 113, 136,
        "检索为空（或纯向量模式下相似度低于阈值）时直接返回兜底话术，从源头掐掉无依据生成。",
    ),
    (
        "提示词硬约束",
        "app/services/rag_pipeline.py", 24, 42,
        "要求只依据资料作答、逐条标注 [n]、资料不足要明说未找到并给确认渠道、保留版本限定。",
    ),
]


def read_lines(rel_path: str, start: int, end: int) -> str:
    path = PROJECT_ROOT / rel_path
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[start - 1 : end])


# --------------------------------------------------------------------------- #
# 极简语法高亮
# --------------------------------------------------------------------------- #
TOKEN_RE = re.compile(
    r"(?P<comment>#[^\n]*)"
    r"|(?P<string>\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
    r"|(?P<number>\b\d+(?:\.\d+)?\b)"
    r"|(?P<keyword>\b(?:def|class|return|if|elif|else|for|while|in|not|and|or|import|from|as|with|try|except|finally|raise|pass|None|True|False|self|async|await|lambda|yield|continue|break|global|nonlocal)\b)"
    r"|(?P<builtin>\b(?:len|set|dict|list|str|int|float|bool|sum|max|min|sorted|range|enumerate|zip|print|isinstance|getattr|hasattr|super|round|abs|any|all)\b)"
    r"|(?P<decorator>@\w+)"
)

COLORS = {
    "comment": "#6b7f6b",
    "string": "#b06a3b",
    "number": "#8a5cc4",
    "keyword": "#2b6fb3",
    "builtin": "#0f7f8f",
    "decorator": "#a0603c",
}


def highlight(code: str) -> str:
    out: list[str] = []
    pos = 0
    for match in TOKEN_RE.finditer(code):
        out.append(html.escape(code[pos : match.start()]))
        kind = match.lastgroup or ""
        out.append(
            f'<span style="color:{COLORS.get(kind, "#c9d4e0")}">{html.escape(match.group())}</span>'
        )
        pos = match.end()
    out.append(html.escape(code[pos:]))
    return "".join(out)


def code_block(code: str, label: str, note: str = "") -> str:
    note_html = f'<p class="code-note">{html.escape(note)}</p>' if note else ""
    return f"""
    <div class="code-wrap">
      <div class="code-head"><span>{html.escape(label)}</span></div>
      <pre class="code"><code>{highlight(code)}</code></pre>
      {note_html}
    </div>"""


def img_block(path: Path, caption: str) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return (
        f'<figure class="fig"><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"/>'
        f'<figcaption>{caption}</figcaption></figure>'
    )


# --------------------------------------------------------------------------- #
# 实时统计
# --------------------------------------------------------------------------- #
@dataclass
class Stats:
    py_lines: int
    app_modules: int
    endpoints: int
    docs: int
    chunks: int
    vocab: int
    dim: int


def collect_stats() -> Stats:
    py_files = [
        p
        for folder in ("app", "scripts", "tests")
        for p in (PROJECT_ROOT / folder).rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    py_lines = sum(len(p.read_text(encoding="utf-8", errors="replace").splitlines()) for p in py_files)
    app_modules = len([p for p in (PROJECT_ROOT / "app").rglob("*.py") if "__pycache__" not in p.parts])

    # 接口数量：数装饰器 @router.<method> 与 @app.<method>
    endpoints = 0
    for p in (PROJECT_ROOT / "app").rglob("*.py"):
        text = p.read_text(encoding="utf-8", errors="replace")
        endpoints += len(re.findall(r"@(?:router|app)\.(?:get|post|delete|put)\(", text))

    docs = len(
        [
            p
            for p in (PROJECT_ROOT / "data" / "documents").rglob("*")
            if p.is_file() and not p.name.startswith(".")
        ]
    )
    # 知识库规模与词典：读取本地 BM25 索引（若无则记 0）
    chunks = vocab = 0
    index_path = PROJECT_ROOT / "data" / "bm25_index.json"
    if index_path.exists():
        import json

        payload = json.loads(index_path.read_text(encoding="utf-8"))
        chunks = len(payload.get("chunks") or [])
        tokens: set[str] = set()
        for item in payload.get("chunks") or []:
            from app.core.text_utils import tokenize  # 复用项目分词，保证一致

            tokens.update(tokenize(str(item.get("text") or "")))
        vocab = len(tokens)
    return Stats(py_lines, app_modules, endpoints, docs, chunks, vocab, 512)


# --------------------------------------------------------------------------- #
# 文档内容
# --------------------------------------------------------------------------- #
def build_html(stats: Stats) -> str:
    images = PROJECT_ROOT / "docs" / "images"
    excerpt_html = "\n".join(
        code_block(read_lines(path, start, end), f"{title}　·　{path} 第 {start}–{end} 行", note)
        for title, path, start, end, note in CODE_EXCERPTS
    )

    today = date.today().strftime("%Y 年 %m 月 %d 日")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8" />
<title>AutoRAG 项目说明文档</title>
<style>
  @page {{ size: A4; margin: 16mm 15mm 16mm 15mm; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; background: #fffdf9; color: #1c1a17;
    font: 13.5px/1.75 -apple-system, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif;
    -webkit-print-color-adjust: exact; print-color-adjust: exact;
  }}
  /* 封面与目录用整页容器；正文部分自然流式分页，避免页尾大片空白 */
  .sheet {{
    width: 210mm; min-height: 297mm; padding: 16mm 15mm; margin: 0 auto;
    page-break-after: always; position: relative;
  }}
  .sheet.cover {{ display: flex; flex-direction: column; justify-content: space-between; }}
  .flow {{ padding: 0 15mm 10mm; max-width: 210mm; margin: 0 auto; }}
  .chapter {{ page-break-before: always; }}
  .chapter:first-of-type {{ page-break-before: auto; }}
  h1 {{ font-size: 30px; line-height: 1.3; margin: 0 0 6px; letter-spacing: -.5px; }}
  h2 {{
    font-size: 19px; margin: 0 0 14px; padding-bottom: 7px;
    border-bottom: 2px solid #1c1a17; letter-spacing: -.2px;
    page-break-after: avoid;
  }}
  h3 {{ font-size: 15px; margin: 20px 0 8px; page-break-after: avoid; }}
  h4 {{ font-size: 13.5px; margin: 14px 0 6px; color: #4a453d; page-break-after: avoid; }}
  p {{ margin: 0 0 9px; }}
  ul, ol {{ margin: 0 0 10px; padding-left: 22px; }}
  li {{ margin-bottom: 4px; }}
  .muted {{ color: #6d675e; }}
  .small {{ font-size: 12px; }}
  .kicker {{ font-size: 12px; letter-spacing: 2px; color: #a8663c; text-transform: uppercase; margin-bottom: 10px; }}
  .accent {{ color: #a8663c; }}
  .rule {{ height: 1px; background: #dcd6cc; margin: 16px 0; }}
  .lead {{ font-size: 15px; line-height: 1.85; color: #33302b; }}

  /* 封面 */
  .cover {{ display: flex; flex-direction: column; justify-content: space-between; }}
  .cover-meta {{ font-size: 12.5px; color: #6d675e; }}
  .cover-stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 22px; }}
  .stat {{ border: 1px solid #ded8ce; border-radius: 9px; padding: 12px 13px; background: #faf8f4; }}
  .stat b {{ display: block; font-size: 22px; line-height: 1.1; color: #1c1a17; }}
  .stat span {{ font-size: 11.5px; color: #6d675e; }}

  /* 表格 */
  table {{ width: 100%; border-collapse: collapse; margin: 10px 0 14px; font-size: 12.5px; }}
  th, td {{ border: 1px solid #ded8ce; padding: 7px 9px; text-align: left; vertical-align: top; }}
  th {{ background: #f0ece4; font-weight: 600; }}
  tr {{ page-break-inside: avoid; }}
  thead {{ display: table-header-group; }}
  td code {{ background: #f2efe9; padding: 1px 4px; border-radius: 3px; font-size: 11.5px; }}

  /* 代码 */
  .code-wrap {{ margin: 10px 0 16px; page-break-inside: auto; }}
  .code-head {{
    background: #2b2f36; color: #cfd6df; font-size: 11.5px; padding: 6px 11px;
    border-radius: 7px 7px 0 0; font-family: Consolas, "Cascadia Mono", monospace;
    page-break-after: avoid;
  }}
  pre.code {{
    margin: 0; background: #1e2228; color: #d7dee7; border-radius: 0 0 7px 7px;
    padding: 11px 13px; overflow-x: auto;
    font: 10.5px/1.5 Consolas, "Cascadia Mono", "Courier New", monospace;
    white-space: pre-wrap; word-break: break-word;
  }}
  .code-note {{ font-size: 11.5px; color: #6d675e; margin: 6px 0 0; padding-left: 2px; }}

  /* 图 */
  .fig {{ margin: 12px 0 16px; page-break-inside: avoid; }}
  .fig img {{ width: 100%; border: 1px solid #d5cfc5; border-radius: 7px; display: block; }}
  .fig figcaption {{ font-size: 11.5px; color: #6d675e; margin-top: 6px; text-align: center; }}

  /* 架构图 */
  .arch {{ margin: 6px 0 16px; }}
  .lane {{ border: 1px dashed #cfc7ba; border-radius: 10px; padding: 12px 14px; margin-bottom: 12px; background: #fbf9f5; }}
  .lane-title {{ font-size: 12px; color: #a8663c; font-weight: 600; margin-bottom: 9px; letter-spacing: .6px; }}
  .flow {{ display: flex; align-items: stretch; gap: 8px; flex-wrap: wrap; }}
  .node {{
    flex: 1 1 140px; min-width: 110px; border: 1px solid #d5cfc5; border-radius: 8px;
    padding: 9px 10px; background: #fff; font-size: 11.5px; line-height: 1.5;
  }}
  .node b {{ display: block; font-size: 12px; margin-bottom: 3px; }}
  .node span {{ color: #6d675e; font-size: 11px; }}
  .node.hot {{ border-color: #d9a86c; background: #fdf6ec; }}
  .arrow {{ align-self: center; color: #b9b1a5; font-size: 15px; }}
  .stack {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
  .box {{ border: 1px solid #ded8ce; border-radius: 9px; padding: 11px 13px; background: #fff; }}
  .box b {{ font-size: 12.5px; }}
  .box ul {{ margin: 6px 0 0; padding-left: 18px; font-size: 11.5px; color: #4a453d; }}

  /* 缺陷记录 */
  .defect {{ border-left: 3px solid #a8663c; padding: 2px 0 2px 14px; margin: 0 0 20px; page-break-inside: avoid; }}
  .defect h3 {{ margin-top: 0; }}
  .steps5 {{ font-size: 12.5px; }}
  .steps5 dt {{ font-weight: 600; color: #4a453d; margin-top: 7px; font-size: 12px; }}
  .steps5 dd {{ margin: 2px 0 0 0; }}
  .badge {{ display: inline-block; font-size: 10.5px; padding: 1px 7px; border-radius: 9px; background: #f0ece4; color: #6d675e; margin-left: 6px; }}
  .toc {{ font-size: 13px; }}
  .toc li {{ margin-bottom: 7px; }}
  .toc .pg {{ float: right; color: #8d867b; font-size: 12px; }}
</style>
</head>
<body>

<!-- ================= 封面 ================= -->
<section class="sheet cover">
  <div>
    <div class="kicker">RAG 知识库问答系统 · 项目说明文档</div>
    <h1>AutoRAG</h1>
    <p class="lead">
      面向<b>汽车售后场景</b>的检索增强生成（RAG）问答后端。<br/>
      把保养周期、故障码、保修条款等文档切分入库，检索后生成<b>带引用来源</b>的回答；
      检索不到依据时<b>明确拒答</b>，不编造参数。
    </p>
    <div class="rule"></div>
    <p class="muted small">
      技术栈：Python 3.13 · FastAPI · Chroma · sentence-transformers（BGE 中文向量）·
      自研 BM25 · 轻量重排 · Docker
    </p>
    <div class="cover-stats">
      <div class="stat"><b>{stats.endpoints}</b><span>REST 接口</span></div>
      <div class="stat"><b>{stats.py_lines:,}</b><span>Python 代码行</span></div>
      <div class="stat"><b>{stats.chunks}</b><span>知识库片段</span></div>
      <div class="stat"><b>{stats.dim}</b><span>向量维度</span></div>
    </div>
  </div>
  <div class="cover-meta">
    <div class="rule"></div>
    <p>
      项目作者：张宇翔　|　仓库：github.com/zyx4303/autorag（MIT）<br/>
      文档生成日期：{today}<br/>
      说明：本文档所有数据与截图均来自本机实际运行，未使用任何编造的指标；
      文中的代码片段由脚本从源码按行号提取，未做手工改写。
    </p>
  </div>
</section>

<!-- ================= 目录 ================= -->
<section class="chapter">
  <h2>目录</h2>
  <ol class="toc">
    <li>项目概览：一句话定位与关键数据 <span class="pg">P3</span></li>
    <li>业务背景：为什么汽车售后需要 RAG <span class="pg">P4</span></li>
    <li>系统架构：请求链路、入库链路与目录结构 <span class="pg">P5</span></li>
    <li>核心实现：从切分到带引用生成的六个关键设计 <span class="pg">P6</span></li>
    <li>缺陷排查记录：4 个真实问题的定位与修复过程 <span class="pg">P9</span></li>
    <li>运行效果：真实问答与拒答截图 <span class="pg">P13</span></li>
    <li>工程实践：可复现、可观测、可交付 <span class="pg">P14</span></li>
    <li>已知局限与下一步计划 <span class="pg">P15</span></li>
  </ol>
  <div class="rule"></div>
  <h3>这份文档想回答什么</h3>
  <ul>
    <li><b>它解决什么问题</b>——汽车售后知识分散在手册、码表、法规里，人工查找慢且容易出错。</li>
    <li><b>技术上难在哪</b>——结构化内容（表格、故障码）的切分、中文单字分词噪声、多部分提问的召回、以及"不许编造"的可验证机制。</li>
    <li><b>我是怎么做的</b>——每个技术决策背后的失败样例与验证数据。</li>
    <li><b>边界在哪</b>——哪些做不到、为什么继续调参没有意义。</li>
  </ul>
</section>

<div class="flow">
<!-- ================= 1 项目概览 ================= -->
<section class="chapter">
  <h2>1. 项目概览</h2>
  <p class="lead">
    一个可运行、可复现、边界清晰的 RAG 后端。它的核心主张是：
    <b>回答里的每一句结论都必须能追溯到你自己的文档</b>；查不到就明说查不到。
  </p>

  <h3>1.1 关键数据（本机实测）</h3>
  <table>
    <tr><th style="width:34%">指标</th><th>数值</th></tr>
    <tr><td>知识库文档 / 片段</td><td>{stats.docs} 份文档 · {stats.chunks} 个片段（示例数据集，可自由使用）</td></tr>
    <tr><td>向量模型 / 维度</td><td>BAAI/bge-small-zh-v1.5 · {stats.dim} 维（本地推理，无需云端 Key）</td></tr>
    <tr><td>关键词索引</td><td>自研 BM25，词典 {stats.vocab:,} 词，与向量库条数一致（启动时自动校验同步）</td></tr>
    <tr><td>切分参数</td><td>chunk_size=600 字符 · overlap=80 · 表格作为原子块</td></tr>
    <tr><td>检索策略</td><td>hybrid：向量召回 top20 + BM25 top20 → RRF 融合 → 四信号重排 → 取 top5</td></tr>
    <tr><td>接口数量</td><td>{stats.endpoints} 个 REST 端点（入库 / 检索 / 问答 / 流式 / 评测 / 健康检查）</td></tr>
    <tr><td>代码规模</td><td>{stats.py_lines:,} 行 Python · {stats.app_modules} 个模块 · 30 项纯逻辑断言</td></tr>
    <tr><td>检索回归表现</td><td>9 条人工构造查询：第 1 名命中 7/9，前 3 名命中 8/9（含 2 个未解决样例，见第 8 节）</td></tr>
  </table>
  <p class="small muted">
    以上数字均为本机单次真实运行结果，用于回归验证与排查对照，<b>不是准确率、QPS 或性能基准</b>，
    也没有与任何其他系统做横向对比。
  </p>

  <h3>1.2 和"直接问大模型"的区别</h3>
  <table>
    <tr><th style="width:30%">环节</th><th>直接问大模型</th><th>本项目</th></tr>
    <tr><td>知识来源</td><td>模型参数里的通用知识，无法指定</td><td>只使用你放进 <code>data/documents/</code> 的文档</td></tr>
    <tr><td>回答依据</td><td>无出处</td><td>每条结论带 <code>[n]</code> 角标，并列出文件名与章节</td></tr>
    <tr><td>引用可靠性</td><td>—</td><td>服务端校验角标是否越界，越界写入 <code>warnings</code></td></tr>
    <tr><td>查不到时</td><td>倾向于编一个看起来合理的答案</td><td>检索为空则<b>不调用大模型</b>，直接返回兜底话术与确认渠道</td></tr>
  </table>
</section>

<!-- ================= 2 业务背景 ================= -->
<section class="chapter">
  <h2>2. 业务背景</h2>

  <h3>2.1 售后场景的知识痛点</h3>
  <p>
    车主与售后顾问的日常提问高度集中在几类问题上：<b>多久保养一次</b>、<b>用什么规格的油液</b>、
    <b>这个故障码是什么意思</b>、<b>保修包不包</b>。这些答案确实都在文档里，但分散且形态各异：
  </p>
  <table>
    <tr><th style="width:26%">资料类型</th><th>典型形态</th><th>对检索的挑战</th></tr>
    <tr><td>保养周期表</td><td>Markdown 表格，行是项目、列是里程/月份</td><td>表格被切断后列语义丢失，检索出来是一堆没有表头的数字</td></tr>
    <tr><td>故障码表</td><td>几百行「码号 + 含义」的表格</td><td>码号是精确串，同义改写会漂移；单条定义容易被大表稀释</td></tr>
    <tr><td>法规条款</td><td>连续长段落，含条号与限定条件</td><td>长段落里"三包有效期"和"包修期"常出现在同一段，需保留限定</td></tr>
    <tr><td>维修手册</td><td>图文混排、带版本与车型限定</td><td>数值必须原样保留（车型/年款限定不能丢）</td></tr>
  </table>

  <h3>2.2 三个真实约束</h3>
  <ul>
    <li><b>不能编造。</b>售后场景里一个错误的扭矩值或保养间隔可能直接导致事故或保修纠纷，
      因此系统必须"不知道就说不知道"，而不是给一个像样的猜测。</li>
    <li><b>必须可追溯。</b>售后人员需要把答案拿给客户看，所以每个结论都要能指到具体文档与章节，
      否则无法作为沟通依据。</li>
    <li><b>成本要可控。</b>中小售后网点不会有 GPU 集群，方案要能在普通 CPU 机器上跑起来，
      并且能离线（本地向量模型 + 本地缓存）。</li>
  </ul>

  <h3>2.3 项目的定位与取舍</h3>
  <table>
    <tr><th style="width:24%">决策点</th><th>选择</th><th>理由</th></tr>
    <tr><td>向量模型</td><td>本地 BGE 中文小模型（512 维）</td><td>免 Key、可离线、中文效果够用；代价是首次要下载约 92MB 权重</td></tr>
    <tr><td>是否用 RAG 框架</td><td>不用 LangChain 等框架，自研切分与检索</td><td>项目的目标是把 RAG 每一环讲清楚；框架会把打分、融合、引用校验藏在内部</td></tr>
    <tr><td>是否公开部署</td><td>不公开部署，仅本地运行 + 开源代码</td><td>知识库可能包含厂商手册与法规条款，公开提供检索属于信息网络传播行为，需先确认授权</td></tr>
  </table>
</section>

<!-- ================= 3 系统架构 ================= -->
<section class="chapter">
  <h2>3. 系统架构</h2>

  <h3>3.1 请求链路（在线问答）</h3>
  <div class="arch">
    <div class="lane">
      <div class="lane-title">① 接入层</div>
      <div class="flow">
        <div class="node"><b>浏览器控制台</b><span>app/web/console.html</span></div>
        <div class="arrow">→</div>
        <div class="node"><b>FastAPI</b><span>app/main.py · 路由与依赖注入</span></div>
        <div class="arrow">→</div>
        <div class="node"><b>接口校验</b><span>app/models.py · Pydantic 契约</span></div>
      </div>
    </div>
    <div class="lane">
      <div class="lane-title">② 检索层（可单独观测）</div>
      <div class="flow">
        <div class="node hot"><b>向量召回</b><span>Chroma · cosine · top20<br/>app/core/vector_store.py</span></div>
        <div class="arrow">＋</div>
        <div class="node hot"><b>BM25 召回</b><span>自研 · 零依赖 · top20<br/>app/core/bm25.py</span></div>
        <div class="arrow">→</div>
        <div class="node"><b>RRF 融合</b><span>两路名次分归一化后相加<br/>app/core/retriever.py</span></div>
        <div class="arrow">→</div>
        <div class="node"><b>四信号重排</b><span>融合分/词覆盖/高精度串/章节命中</span></div>
      </div>
    </div>
    <div class="lane">
      <div class="lane-title">③ 生成层</div>
      <div class="flow">
        <div class="node"><b>上下文组装</b><span>片段按 [n] 编号 + 章节路径<br/>app/services/rag_pipeline.py</span></div>
        <div class="arrow">→</div>
        <div class="node"><b>兜底判断</b><span>检索为空/vendor 阈值 → 不调用模型</span></div>
        <div class="arrow">→</div>
        <div class="node"><b>LLM 调用</b><span>OpenAI 兼容 · httpx · 流式<br/>app/core/llm_client.py</span></div>
        <div class="arrow">→</div>
        <div class="node"><b>引用校验</b><span>角标越界检测 → warnings</span></div>
      </div>
    </div>
  </div>

  <h3>3.2 入库链路（离线构建）</h3>
  <div class="arch">
    <div class="lane">
      <div class="lane-title">文档 → 可检索的片段</div>
      <div class="flow">
        <div class="node"><b>加载</b><span>md/txt 内置<br/>pdf/docx 可选依赖</span></div>
        <div class="arrow">→</div>
        <div class="node hot"><b>切分</b><span>标题层级→段落→句子<br/>表格原子块</span></div>
        <div class="arrow">→</div>
        <div class="node"><b>向量化</b><span>BGE 512 维<br/>批量 + 重试</span></div>
        <div class="arrow">→</div>
        <div class="node"><b>写入 Chroma</b><span>upsert · 按来源覆盖</span></div>
        <div class="arrow">→</div>
        <div class="node"><b>重建 BM25</b><span>JSON 持久化<br/>与向量库条数校验</span></div>
      </div>
    </div>
    <div class="lane">
      <div class="lane-title">增量与一致性</div>
      <div class="flow">
        <div class="node"><b>checksum 比对</b><span>未变更文件跳过</span></div>
        <div class="arrow">→</div>
        <div class="node"><b>删除同步</b><span>磁盘删除 → 向量库与索引同步清理</span></div>
        <div class="arrow">→</div>
        <div class="node"><b>启动自检</b><span>维度一致性 / 索引同步 / 依赖版本</span></div>
      </div>
    </div>
  </div>

  <h3>3.3 目录结构（关键部分）</h3>
  <table>
    <tr><th style="width:34%">路径</th><th>职责</th></tr>
    <tr><td><code>app/api/</code></td><td>路由与依赖注入：system / ingest / qa / eval 四组接口</td></tr>
    <tr><td><code>app/core/</code></td><td>切分、BM25、向量库、检索、embedding、LLM 客户端（无业务逻辑的纯能力层）</td></tr>
    <tr><td><code>app/services/</code></td><td>入库编排、问答链路、评测服务、组件装配（业务层）</td></tr>
    <tr><td><code>scripts/</code></td><td>启动自检、入库、评测、发布检查（4 个命令行工具）</td></tr>
    <tr><td><code>tests/smoke_test.py</code></td><td>30 项纯逻辑断言：不联网、不需要 Key、不调用大模型</td></tr>
    <tr><td><code>data/documents/</code></td><td>示例知识库（虚构数据 + 自编码表 + 法规全文，均可自由使用）</td></tr>
  </table>
</section>

<!-- ================= 4 核心实现 ================= -->
<section class="chapter">
  <h2>4. 核心实现</h2>
  <p>
    下面六个设计是决定效果的关键。每一项都对应第 5 节里的一个真实缺陷——
    <b>它们不是一开始就设计出来的，而是被失败样例逼出来的。</b>
  </p>

  <h3>4.1 切分：结构感知 + 表格原子块</h3>
  <p>
    切分分三层兜底：<b>标题层级</b>（Markdown 标题与"第X章/一、"识别为语义边界，并写入章节路径）→
    <b>空行段落</b> → <b>句子</b>。在此之上加一条特殊规则：表格作为原子块，不与普通段落混合。
  </p>
  {code_block(read_lines("app/core/chunker.py", 72, 86), "app/core/chunker.py 第 72–86 行", "表格判定：表格行占比 ≥60% 即视为表格块。")}
  {code_block(read_lines("app/core/chunker.py", 108, 120), "app/core/chunker.py 第 108–120 行", "超长表按行切开，但每片都补回表头，保住「故障码 | 含义」这层列语义。")}

  <h3>4.2 检索：两路召回 + RRF 融合</h3>
  <p>
    向量召回负责"换了说法也能找到"，BM25 负责"精确串必须命中"（故障码、黏度等级、里程数）。
    两路各取 20 条候选，用 RRF 融合——<b>只用名次不用原始分数</b>，
    因为余弦相似度（0~1 量级）与 BM25 分数（理论无上界）根本不可比。
  </p>
  {code_block(read_lines("app/core/retriever.py", 234, 258), "app/core/retriever.py 第 234–258 行", "融合前对两路分别做归一化，避免量纲差异压过名次信息。")}
</section>

<section class="chapter">
  <h3>4.3 重排：四个信号一起投票</h3>
  <p>
    融合之后再做一次轻量重排，把"更像答案"的片段顶上来。四个信号各有来历：
  </p>
  <table>
    <tr><th style="width:22%">信号</th><th style="width:14%">权重</th><th>来历</th></tr>
    <tr><td>融合分</td><td>0.54</td><td>保留两路召回的排序共识</td></tr>
    <tr><td>查询词覆盖率</td><td>0.12</td><td>片段里出现了多少查询词（单字降权后）</td></tr>
    <tr><td>高精度 token</td><td>0.12</td><td>被"P0300 与 P0420 一起问"的失败样例逼出来</td></tr>
    <tr><td>中文词组命中</td><td>0.08</td><td>被"车辆失火"搜不到"气缸失火"逼出来</td></tr>
    <tr><td>章节标题命中</td><td>0.08</td><td>章节路径比正文凝练，命中即强相关</td></tr>
    <tr><td>整句命中</td><td>0.06</td><td>查询原文整体出现在片段里</td></tr>
  </table>
  {code_block(read_lines("app/core/retriever.py", 140, 168), "app/core/retriever.py 第 140–168 行", "先算出中文词组命中比例（跨词边界的兜底匹配）。")}
  {code_block(read_lines("app/core/retriever.py", 169, 190), "app/core/retriever.py 第 169–190 行", "高精度 token 命中比例：P0420 这类串命中一条就该被明显提权。")}
  {code_block(read_lines("app/core/retriever.py", 191, 205), "app/core/retriever.py 第 191–205 行", "章节标题命中 + 四类信号加权求和。")}
</section>

<section class="chapter">
  <h3>4.4 中文分词：单字 + bigram，以及它带来的噪声</h3>
  <p>
    没有引入第三方分词器（避免词典维护与额外依赖），用"单字 + 相邻双字"近似分词：
    <code>更换机油</code> → <code>更 / 换 / 机 / 油 / 更换 / 换机 / 机油</code>。
    好处是零依赖、召回稳；代价是虚词单字会贡献噪声分数——这正是第 5.1 节缺陷的根因。
  </p>
  {code_block(read_lines("app/core/bm25.py", 37, 47), "app/core/bm25.py 第 37–47 行", "单字权重常量：降权是唯一有效的抗噪手段。")}
  {code_block(read_lines("app/core/bm25.py", 197, 218), "app/core/bm25.py 第 197–218 行", "查询词组整体命中加成：先按分组切出词组，命中即乘系数。")}

  <h3>4.5 生成：带引用、可校验、会拒答</h3>
  <p>
    提示词要求"只依据资料作答、逐条标注 <code>[n]</code>、资料不足要明说未找到"，
    但<b>提示词约束不可信</b>——模型仍可能编造不存在的编号，所以服务端必须做二次校验。
  </p>
  {code_block(read_lines("app/services/rag_pipeline.py", 138, 168), "app/services/rag_pipeline.py 第 138–168 行", "引用越界校验：只做编号级校验，不做语义级（这一点在第 8 节如实说明）。")}
</section>

<section class="chapter">
  <h3>4.6 兜底：检索为空时不调用大模型</h3>
  <p>
    无依据生成是 RAG 系统最常见的幻觉来源。这里采取更决绝的做法：
    <b>检索结果为空（或纯向量模式下相似度低于阈值）时直接返回兜底话术</b>，不给模型"自由发挥"的机会。
    响应里的 <code>refused=true</code>、<code>generate_ms=0</code>、<code>model=""</code> 就是没调用模型的证据。
  </p>
  {code_block(read_lines("app/services/rag_pipeline.py", 113, 136), "app/services/rag_pipeline.py 第 113–136 行", "兜底判断：只在纯向量模式下使用相似度阈值，因为只有此时 score 是可解释的余弦值。")}
  {code_block(read_lines("app/services/rag_pipeline.py", 24, 42), "app/services/rag_pipeline.py 第 24–42 行", "系统提示词：把「不许编造」写成可执行的具体规则，而不是笼统要求。")}
</section>

<!-- ================= 5 缺陷排查 ================= -->
<section class="chapter">
  <h2>5. 缺陷排查记录（4 个真实问题）</h2>
  <p>
    这一节是整份文档的核心。下面 4 个问题都不是"写代码时想到的"，
    而是<b>跑起来之后被具体失败样例打脸</b>，再逐个定位、修复、回归验证的。
    每个问题按「现象 → 根因 → 怎么发现 → 修复 → 验证」五步记录。
  </p>

  <div class="defect">
    <h3>缺陷 1｜表格被切碎，故障码定义被稀释<span class="badge">检索排序问题</span></h3>
    <dl class="steps5">
      <dt>现象</dt>
      <dd>问「P0300 是什么意思」，第 1 名召回的是毫无关系的保修条款。</dd>

      <dt>根因</dt>
      <dd>两个问题叠加：① 表格被当作普通段落按空行切开，再被打包进 600 字的块里，
        结果 <code>| P0300 | 引擎曾经有失火现象 |</code> 这一行和 20 多条无关故障码挤在同一个向量里，
        语义被稀释；② 重叠逻辑把上一块的散文尾部拼到了表格行前面，产出
        <code>散文| P0300 | …</code> 这种行内混排的非法 Markdown，表格结构被破坏。</dd>

      <dt>怎么发现</dt>
      <dd>把含 <code>P0300</code> 的所有片段打印出来对比：进程内重新切分得到的
        <b>片段首行是表头</b>、只含同区段故障码；而索引里存的旧片段却是混排文本。
        两边一对比，问题定位到"表格没有作为独立单元处理"。</dd>

      <dt>修复</dt>
      <dd>表格改为原子块：判定（表格行占比 ≥60%）→ 不与段落混合打包 →
        超长表按行切并<b>逐片重复表头</b> → <b>当前块以表格开头时不做重叠</b>。</dd>

      <dt>验证</dt>
      <dd>修复后含 P0300 定义的片段首行为 <code>| 故障码 | 中文描述 |</code>，
        且前 3 名全部是真正含该故障码的片段（英文码表 P0300–P0399、中文码表定义表、范围表）。</dd>
    </dl>
  </div>

  <div class="defect">
    <h3>缺陷 2｜中文虚词单字压制关键词，正确答案排到第 7<span class="badge">打分模型问题</span></h3>
    <dl class="steps5">
      <dt>现象</dt>
      <dd>即使表格修好了，「P0300 是什么意思」的正确片段仍排在保修条款之后，位次第 7。</dd>

      <dt>根因</dt>
      <dd>中文按单字+bigram 切分后，查询里的「是 / 什 / 么 / 意 / 思」各自贡献一份 IDF 分数。
        这些字在语料里罕见（IDF 偏高），于是无关片段靠虚词凑出了 5 分以上，
        而真正含 <code>P0300</code> 的片段只有 2.98 分。</dd>

      <dt>怎么发现</dt>
      <dd>写了一个诊断脚本，打印 BM25 的<b>全量候选与排名</b>，并逐 token 拆解正确片段的分数构成：
        <code>p0300</code> 贡献 2.982，其余虚词贡献 0 —— 确认是"别处靠虚词得分超过它"，
        而不是"它自己没得分"。</dd>

      <dt>修复</dt>
      <dd>两步：① 单个中文字符按 <code>SINGLE_CJK_WEIGHT=0.3</code> 降权；
        ② 新增「查询词组整体命中」加成：查询按空格/标点切出的词组若整体出现在片段中，
        给该片段乘以 <code>PHRASE_BOOST</code>。</dd>

      <dt>验证</dt>
      <dd>修复后 P0300 的前 3 名全是真正含故障码的片段；同时 7 条回归查询命中率保持不降
        （没有为了修一个而弄坏其他）。</dd>
    </dl>
  </div>
</section>

<section class="chapter">
  <div class="defect">
    <h3>缺陷 3｜融合时漏了归一化，排序倒挂<span class="badge">自身逻辑不一致</span></h3>
    <dl class="steps5">
      <dt>现象</dt>
      <dd>诊断输出显示：正确片段在 BM25 通道排第 1、在向量通道也排第 1，
        但融合后掉到第 5 —— 两路都第一却排第五，逻辑上矛盾。</dd>

      <dt>根因</dt>
      <dd>融合函数的注释写着"关键词分数做归一化，使其与向量分数在同一量级"，
        但代码里 <b>只有 keyword 分支做了归一化，vector 分支直接用了余弦相似度</b>
        （0.45 量级），而名次分是 1/(60+rank) 的 0.016 量级。
        两路数量级相差 28 倍，向量通道的绝对值直接压过了名次信息。</dd>

      <dt>怎么发现</dt>
      <dd>先看候选池（确认正确片段确实在两路里），再看最终排序（发现它掉下去了），
        最后逐行读融合函数，发现注释与实现不一致。</dd>

      <dt>修复</dt>
      <dd>向量分支也按"本轮最高分"归一化后再除以 <code>rrf_k + rank</code>，
        两端量纲对齐后再相加。</dd>

      <dt>验证</dt>
      <dd>P0300 与 P0420 两个故障码的查询都能排到前面；
        「P0300 故障码是什么意思？P0420 呢？」这类多部分提问也能同时召回两个码。</dd>
    </dl>
  </div>

  <div class="defect">
    <h3>缺陷 4｜多部分提问丢失第二个实体<span class="badge">重排信号缺失</span></h3>
    <dl class="steps5">
      <dt>现象</dt>
      <dd>问「P0300 故障码是什么意思？P0420 呢？」，模型答出 P0300 后说
        「P0420 在当前知识库中未找到相关依据」——而中文码表里明确有 P0420 的定义。</dd>

      <dt>根因</dt>
      <dd>重排只看"查询词覆盖率"，两个故障码各占查询 token 的约 1/5，
        大量虚词把两者的权重拉平，含 P0420 的片段排不进 top5。
        单独问 P0420 时它能排到第 2 —— 说明不是召回缺失，而是排序问题。</dd>

      <dt>怎么发现</dt>
      <dd>对比实验：单独问 <code>P0420</code> 与组合问句，前者命中后者丢失，
        由此确定为排序而非召回问题。</dd>

      <dt>修复</dt>
      <dd>新增「高精度 token」信号：含数字或字母的短词（P0420、5w、obd2）命中率单独计分并给 0.12 权重。
        后来又被「车辆失火」这一例逼出「中文词组命中」信号（0.08 权重）。</dd>

      <dt>验证</dt>
      <dd>组合问句现在同时召回两个码的中文定义与英文对照，模型给出 4 条引用；
        9 条回归查询：第 1 名命中 7/9、前 3 名命中 8/9。</dd>
    </dl>
  </div>

  <h3>5.5 这 4 个问题的共同点</h3>
  <ul>
    <li><b>都不是靠读代码发现的</b>，而是靠"打印中间态 + 对比实验"定位的：
      全量候选排名、分数构成拆解、两路对照、组合 vs 单独提问。</li>
    <li><b>注释与实现不一致是高频陷阱</b>（缺陷 3 的注释写着归一化，代码没做），
      所以修完之后我把注释也改成了"为什么必须这么做"。</li>
    <li><b>每次修复都跑回归</b>，避免为了修一个查询而弄坏其他查询。</li>
  </ul>
</section>

<!-- ================= 6 运行效果 ================= -->
<section class="chapter">
  <h2>6. 运行效果</h2>
  <p class="muted small">
    以下截图来自本机实际运行的控制台（<code>http://127.0.0.1:8000/</code>），
    通过 Edge 无头模式在真实回答渲染完成后截取，未做任何美化或拼接。
  </p>

  <h3>6.1 正常问答：结论 + 引用 + 依据 + 命中的原文片段</h3>
  {img_block(images / "01-answer-with-citations.png", "内置控制台的真实问答结果：含 [1] 角标、依据列表、耗时拆解与命中的原文片段")}
  <p class="small">
    界面上的三个细节值得注意：① 每条结论后都有 <code>[n]</code> 角标；
    ② 「依据：」列出文件名与章节路径（来自切分时写入的 metadata）；
    ③ 下方灰色块是本次实际命中的原文片段，可直接核对答案是否忠于原文。
  </p>
</section>

<section class="chapter">
  <h3>6.2 无依据时拒答：说明资料里有什么、没有什么</h3>
  {img_block(images / "02-refusal-no-evidence.png", "负样本：询问其他品牌保养价格与未上市车型参数时的拒答结果")}
  <p class="small">
    模型没有编造价格与参数，而是明确说明"未找到相关依据"，
    并指出知识库里现有的是哪三类资料（虚构保养周期、格式示例故障码、三包条款），
    再给出确认渠道。响应中 <code>refused</code> 与 <code>warnings</code> 字段会标明本次是否走了兜底。
  </p>
  <div class="rule"></div>
  <h3>6.3 两个接口的定位差异</h3>
  <table>
    <tr><th style="width:26%">接口</th><th>用途</th><th>是否消耗 token</th></tr>
    <tr><td><code>POST /api/v1/retrieve</code></td><td>只做检索，返回片段与两路分数</td><td>否</td></tr>
    <tr><td><code>POST /api/v1/chat</code></td><td>完整问答，返回答案与引用</td><td>是</td></tr>
    <tr><td><code>POST /api/v1/chat/stream</code></td><td>SSE 流式，先推 meta（含引用）再推 token</td><td>是</td></tr>
    <tr><td><code>POST /api/v1/eval/run</code></td><td>跑评测集并按规则打分</td><td>是</td></tr>
    <tr><td><code>GET /api/v1/health</code></td><td>健康检查，可探测 LLM 连通性</td><td>可选</td></tr>
  </table>
  <p class="small muted">
    把"检索"和"生成"拆成两个接口，是为了在效果不好时能快速判断该怪召回还是怪模型——
    这是排查过程中最实用的一条设计。
  </p>
</section>

<!-- ================= 7 工程实践 ================= -->
<section class="chapter">
  <h2>7. 工程实践</h2>

  <h3>7.1 可复现：所有结论都能一条命令验证</h3>
  <table>
    <tr><th style="width:38%">脚本</th><th>作用</th><th style="width:16%">是否联网</th></tr>
    <tr><td><code>scripts/preflight.py</code></td><td>启动自检：依赖版本、配置合法性、模块导入、embedding 维度一致性、向量库与 BM25 是否同步</td><td>可选</td></tr>
    <tr><td><code>tests/smoke_test.py</code></td><td>30 项纯逻辑断言：分句分词、切分不丢内容、BM25 召回、RRF 融合、引用越界校验</td><td>否</td></tr>
    <tr><td><code>scripts/ingest.py</code></td><td>命令行入库：扫描、增量比对、切分、向量化、重建 BM25</td><td>否</td></tr>
    <tr><td><code>scripts/run_eval.py</code></td><td>批量评测并按规则打分，输出 JSON + Markdown 报告</td><td>可选</td></tr>
    <tr><td><code>scripts/publish_check.py</code></td><td>发布前检查：密钥泄露、文档授权风险、运行时产物是否会被提交</td><td>否</td></tr>
  </table>

  <h3>7.2 可观测：每个响应都带排查所需的字段</h3>
  <ul>
    <li><code>latency</code>：检索 / 生成 / 合计三段耗时，能一眼看出慢在哪一环；</li>
    <li><code>warnings</code>：引用越界、未标注角标、向量降级等异常如实上报，不静默；</li>
    <li><code>usage</code>：prompt / completion / total tokens，便于估算成本；</li>
    <li>检索结果同时带 <code>vector_score</code> 与 <code>keyword_score</code>，能看出是哪一路命中的；</li>
    <li><code>/health</code> 的 <code>notes</code> 会把所有降级（hash 向量、Key 未配置）明确列出。</li>
  </ul>

  <h3>7.3 可交付：Docker 与工程化细节</h3>
  <ul>
    <li>Dockerfile：非 root 用户运行、<code>tini</code> 转发信号、内置 HEALTHCHECK（只探 /health，不消耗 token）；</li>
    <li>docker-compose.yml：数据目录挂载到宿主机，容器销毁不丢向量库；</li>
    <li>依赖版本踩坑记录：<code>chromadb</code> 0.5.x 依赖的 <code>chroma-hnswlib</code> 只有到 cp311 的 Windows wheel，
      在 Python 3.12/3.13 上会退化成源码编译并失败；因此固定 <code>chromadb&gt;=1.0,&lt;2.0</code>（有预编译 wheel）；</li>
    <li>配置防御：非法枚举值自动纠正并写入告警，而不是启动即崩；</li>
    <li>异步一致性：Chroma 的同步阻塞调用统一用 <code>asyncio.to_thread</code> 挪出事件循环。</li>
  </ul>

  <h3>7.4 关于"数据不编造"的自律</h3>
  <ul>
    <li>README 与本文档里<b>不出现</b>准确率、召回率、QPS、并发等无法自证的指标；</li>
    <li>所有数字都标注测量条件（本机、单次、样本量），并说明它"不是"什么；</li>
    <li>未解决的失败样例照实列出，并说明为什么继续调参没有意义。</li>
  </ul>
</section>

<!-- ================= 8 局限与计划 ================= -->
<section class="chapter">
  <h2>8. 已知局限与下一步计划</h2>

  <h3>8.1 检索层面还没解决的失败样例</h3>
  <table>
    <tr><th style="width:32%">查询</th><th style="width:30%">第 1 名实际召回</th><th>原因</th></tr>
    <tr>
      <td>虚构车型A多久换一次机油</td>
      <td>README 索引.md（而非保养时间表）</td>
      <td>目录索引类文档同时提到了所有关键词，启发式信号分不清"摘要"与"正文"</td>
    </tr>
    <tr>
      <td>车辆失火是什么原因</td>
      <td>保修条款全文（正确答案在前 3 名内）</td>
      <td>"车辆"是高频搭配词，BM25 单字+双字分数仍高于真正含"失火"的片段</td>
    </tr>
  </table>
  <p>
    <b>为什么停在这里不再调权重：</b>这两例需要的是语义级精排（cross-encoder / bge-reranker），
    继续加启发式权重就是在针对样例过拟合。这也正是下一步计划的第一项。
  </p>
  <p class="small muted">
    补充一点：排序失误不等于问答失败。若把 <code>TOP_K</code> 从 5 提到 8，
    "车辆失火"这类问题在生成侧已经能答对（正确答案就在前 3）。
  </p>

  <h3>8.2 能力边界</h3>
  <table>
    <tr><th style="width:30%">维度</th><th>现状</th></tr>
    <tr><td>引用校验</td><td>只做到<b>编号级</b>（角标不越界、与实际召回一一对应）；<b>做不到语义级</b>——无法判断"这句话是否真被该片段支持"，需要 NLI 模型或 LLM-as-judge</td></tr>
    <tr><td>重排</td><td>启发式（四信号加权），不是 cross-encoder 精排</td></tr>
    <tr><td>多轮对话</td><td>未实现；接口里预留了 <code>session_id</code> 字段</td></tr>
    <tr><td>BM25 索引</td><td>单进程内存态，多副本部署会不同步，要横向扩容需换 ES/OpenSearch</td></tr>
    <tr><td>入库任务</td><td>同步长任务，大目录会长时间占用请求，生产上应改为任务队列 + 进度查询</td></tr>
    <tr><td>评测集</td><td>20 条模板已备，<code>ground_truth</code> 需按真实资料补齐后才有可对外的效果口径</td></tr>
  </table>

  <h3>8.3 下一步计划（按性价比排序）</h3>
  <ol>
    <li><b>接入 rerank 模型</b>（如 bge-reranker）替换启发式重排 —— 直接针对 8.1 的两个失败样例；</li>
    <li><b>多轮对话</b>：按 <code>session_id</code> 维护历史，并对历史问题做查询改写；</li>
    <li><b>引用高亮</b>：在控制台里把引用片段与答案句做对齐展示，降低人工核对成本；</li>
    <li><b>补齐评测集</b>：把 20 条的期望来源与关键词按真实资料填全，产出可复核的评测报告；</li>
    <li><b>入库异步化</b>：改成后台任务 + 进度查询，支持大批量文档。</li>
  </ol>

  <div class="rule"></div>
  <p class="small muted">
    文档结束。本文档中的所有数据、代码片段与截图均可在仓库中复现：
    运行 <code>python scripts/preflight.py</code> 与 <code>python tests/smoke_test.py</code>
    可验证环境与逻辑；访问内置控制台可复现第 6 节的两张截图。
  </p>
</section>

</div><!-- /flow -->
</body>
</html>
"""


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "resume" / "项目说明文档.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = collect_stats()
    html_text = build_html(stats)
    output.write_text(html_text, encoding="utf-8")
    print(f"已生成: {output}")
    print(f"  统计: {stats.py_lines} 行 Python / {stats.app_modules} 模块 / {stats.endpoints} 接口 / "
          f"{stats.docs} 文档 / {stats.chunks} 片段 / 词典 {stats.vocab}")
    print(f"  体积: {output.stat().st_size:,} 字节")
    return 0


if __name__ == "__main__":
    sys.exit(main())
