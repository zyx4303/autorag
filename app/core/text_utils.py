"""文本处理工具：分句、分词、哈希、片段摘要。"""
from __future__ import annotations

import hashlib
import re

from app.core.aliases import alias_line
from typing import Dict, List, Optional

# 中英文句末标点（保留标点）
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;])|(?<=\.)\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"[ \t\u00a0]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def stable_hash(text: str, length: int = 12) -> str:
    """内容稳定哈希，用于文档 ID 与增量判断。"""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:length]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """统一换行、压缩连续空格与空行，去掉零宽字符。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """按中英文标点切成句子，尽量不切碎（无标点的长句会原样返回）。"""
    parts: List[str] = []
    for piece in _SENTENCE_SPLIT_RE.split(text):
        if piece is None:
            continue
        piece = piece.strip()
        if piece:
            parts.append(piece)
    return parts or ([text.strip()] if text.strip() else [])


def tokenize(text: str) -> List[str]:
    """中英文混合分词（供 BM25 使用，零第三方依赖）。

    规则：
    - 英文/数字：按词切分并转小写；
    - 中文：单字 + 相邻双字（bigram），从而在无分词模型时也能近似匹配词组。
      例："更换机油" -> ["更", "换", "机", "油", "更换", "换机", "机油"]
    """
    if not text:
        return []

    lowered = text.lower()
    tokens: List[str] = []

    # 中文单字与双字
    cjk_runs = re.findall(r"[\u4e00-\u9fff]+", lowered)
    for run in cjk_runs:
        for ch in run:
            tokens.append(ch)
        for i in range(len(run) - 1):
            tokens.append(run[i : i + 2])

    # 英文数字
    for word in _TOKEN_RE.findall(lowered):
        if len(word) == 1 and not word.isdigit():
            continue  # 单个字母噪声较大
        tokens.append(word)

    return tokens


def make_snippet(text: str, max_chars: int = 160) -> str:
    """生成单行摘要，用于引用预览与调试输出。"""
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    return flat[: max_chars - 1] + "…"


def strip_markdown(text: str) -> str:
    """生成给纯文本模型看的"干净"版本：去掉常见 Markdown 标记。"""
    out = re.sub(r"```.*?```", "", text, flags=re.S)
    out = re.sub(r"`([^`]*)`", r"\1", out)
    out = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", out)
    out = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", out)
    out = re.sub(r"^\s{0,3}#{1,6}\s*", "", out, flags=re.M)
    out = re.sub(r"^\s{0,3}>\s?", "", out, flags=re.M)
    out = re.sub(r"\*\*([^*]*)\*\*", r"\1", out)
    out = re.sub(r"(?<!\*)\*([^*]*)\*(?!\*)", r"\1", out)
    out = re.sub(r"^\s*[-*+]\s+", "", out, flags=re.M)
    return normalize_text(out)


def char_ngrams(text: str, sizes: tuple = (2, 3)) -> Dict[str, int]:
    """字符 n-gram 计数，供零依赖哈希向量使用。"""
    flat = " ".join(text.lower().split())
    counts: Dict[str, int] = {}
    for size in sizes:
        if len(flat) < size:
            continue
        for i in range(len(flat) - size + 1):
            gram = flat[i : i + size]
            if gram.strip() == "":
                continue
            counts[gram] = counts.get(gram, 0) + 1
    return counts


def build_context_text(text: str, *contexts: Optional[str], with_aliases: bool = True) -> str:
    """把上下文（文档标题 / 章节路径）拼在正文前，用于**索引**而非展示。

    解决什么
    --------
    表格类内容被切成纯数据块后，会彻底失去语义锚点。实测片段
    `| 第 1 次 | 16 | 12 |`（118 字）里没有「首保」「保养」「里程」中的任何一个词，
    用户问「首保多少公里」时它 BM25 完全排不进结果，虽然向量能勉强找到（余弦全场最高）。

    把章节路径（如「二、定期保养项目表（按里程 / 时间，以先到者为准）」）拼在正文前，
    向量与词面检索就都能看到「保养」「里程」「时间」这些线索。

    为什么用「；」连接而不是直接空格：多个上下文之间需要边界，
    否则「…项目表」+「…」拼接后会产生跨边界的中文 bigram 噪声。

    别称注入（12.24 之后的第二步，见 aliases 模块）
    ------------------------------------------------
    章节路径解决的是「没有语义锚点」，解决不了**词汇鸿沟**：
    文档写「第 1 次」、用户说「首保」，两词零交集。这里按 aliases.py 的
    词表把命中片段的别称追加成一行（「（同义表述：首保）」），
    BM25 / 向量 / 精排 / 归因的校验面就都能看到用户的说法。
    别称必须是词表里可举证的同义词；片段正文与引用展示不受影响。

    重要：本函数只用于**构造索引文本**。Chroma 里 `documents` 仍存原文，
    引用展示与片段摘要不受影响；逐句归因的校验面**包含**章节路径与别称
    ——它们都是「词典级真相」，真话（首保=第 1 次保养）不该被误判成幻觉。
    """
    parts = [str(item).strip() for item in contexts if item and str(item).strip()]
    if with_aliases:
        # 别称判定看「上下文 + 正文」整体：Q001 的目标片段是纯表格
        # （「第 1 次 | 16 | 12」正文里没有「保养」二字，别称要求会落空），
        # 但它的章节路径「定期保养项目表」有 —— 两者合并才算得准。
        line = alias_line("\n".join(parts + [text]))
        if line:
            parts.append(line)
    if not parts:
        return text
    return "；".join(parts) + "\n" + text


_HR_LINE_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")


def strip_markdown_rules(text: str) -> str:
    """去掉 Markdown 的水平分割线（`---` / `***` / `___` 独占一行）。

    为什么必须在切分前去掉
    ----------------------
    分割线没有任何语义，但它会把切分边界带歪。实测（2026-10-25）：

    文档里 `## 五、保修与三包要点` 前面有一行 `---`。切分时这行被当成
    **上一章节（四、故障码速查）** 的正文保留下来，随后打包装箱时
    又把章节五的段落并了进来 —— 最终这个片段的 `section` 标成了「四」，
    正文却是章节五的内容。79 个片段里有 4 个是这种错配。

    这在只看正文时无伤大雅，但上下文增强会把 `section` 拼进索引文本
    （见 build_context_text），错标签等于往检索里灌错语义。

    只删「独占一行且由同一个 -/*/ _ 重复至少 3 次」的行，
    不会碰到表格分隔行（`|---|`）和列表项（`- 项目`）。
    """
    lines = [line for line in (text or "").split("\n") if not _HR_LINE_RE.match(line)]
    return "\n".join(lines)
