"""文本处理工具：分句、分词、哈希、片段摘要。"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List

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
