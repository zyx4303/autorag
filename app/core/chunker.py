"""文档切分：结构感知 + 递归兜底 + 相邻重叠。

策略（自上而下，逐层兜底）：
1. 结构层：识别 Markdown 标题与"第X章/一、"这类中文编号行，作为语义边界，
   同时记录每个片段所属的章节路径（写进 metadata.section，回答问题时可定位）。
2. 段落层：按空行切段；超长段落（代码块/长表格/无空行长文）按行拆。
3. 句子层：仍超长时按中英文句末标点拆句，再按 chunk_size 累积。
4. 合并层：把小于 chunk_min_chars 的碎块并入前一块（除非它是独立章节标题行）。
5. 重叠层：每个片段开头带上上一块的尾部 chunk_overlap 个字符，避免跨块语义断裂。
"""
from __future__ import annotations

import re
from typing import Dict, List

from app.config import Settings
from app.core.text_utils import normalize_text, split_sentences, stable_hash
from app.models import Chunk, SourceDocument
from app.logging_conf import get_logger

logger = get_logger(__name__)

_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_CN_HEADING_RE = re.compile(
    r"^(?:第[一二三四五六七八九十百千零两0-9]+[章节条部分篇]"
    r"|[一二三四五六七八九十]+[、.．]"
    r"|\d+(?:\.\d+){0,3}[、.．]?\s+\S)"
)
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


class _Segment:
    """切分中间态：一段文本 + 它所属的章节路径 + 在原文档中的字符区间。"""

    __slots__ = ("text", "section", "char_start", "char_end")

    def __init__(self, text: str, section: str, char_start: int, char_end: int) -> None:
        self.text = text
        self.section = section
        self.char_start = char_start
        self.char_end = char_end


def _heading_info(line: str):
    """返回 (层级, 标题文本)；不是标题则返回 None。"""
    match = _MD_HEADING_RE.match(line)
    if match:
        return len(match.group(1)), match.group(2).strip()
    stripped = line.strip()
    if stripped and len(stripped) <= 40 and _CN_HEADING_RE.match(stripped):
        return 3, stripped
    return None


def _split_oversized(block: str) -> List[str]:
    """段落仍然超长时的兜底：先按行（表格/清单友好），再按句子。"""
    parts: List[str] = []
    for line in block.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        parts.append(line)
    if len(parts) <= 1:
        return split_sentences(block)
    return parts


def _is_table_line(line: str) -> bool:
    return bool(_TABLE_ROW_RE.match(line))


def _looks_like_table(block: str) -> bool:
    """判断一段文本是否为 Markdown 表格块。"""
    lines = [line for line in block.split("\n") if line.strip()]
    if len(lines) < 2:
        return False
    table_lines = sum(1 for line in lines if _is_table_line(line))
    return table_lines / len(lines) >= 0.6


def _group_into_blocks(text: str) -> List[str]:
    """把文本按空行分段，但连续的表格行始终聚成一个整块。"""
    blocks: List[str] = []
    buffer: List[str] = []

    def flush() -> None:
        nonlocal buffer
        joined = "\n".join(buffer).strip()
        if joined:
            blocks.append(joined)
        buffer = []

    for line in text.split("\n"):
        if not line.strip():
            flush()
            continue
        buffer.append(line)
    flush()
    return blocks


def _split_table(block: str, chunk_size: int) -> List[str]:
    """把超长表格按行切开，并给每个后续分片重复表头。

    为什么必须重复表头：故障码表的列是"故障码 | 含义"，
    丢掉表头后分片就变成一堆孤立的 `| P0300 | 引擎曾经有失火现象 |`，
    虽然内容还在，但列语义丢失，人也难核对、模型也更容易误读。
    """
    lines = [line for line in block.split("\n") if line.strip()]
    if len(lines) < 3:
        return [block]

    # 前两行是表头与分隔行（|---|---|）
    header = lines[:2]
    rows = lines[2:]
    if not rows or not _is_table_line(header[1]):
        return [block]

    header_text = "\n".join(header)
    parts: List[str] = []
    buffer: List[str] = [header_text]
    buffer_len = len(header_text)
    for row in rows:
        if buffer_len + len(row) + 1 > chunk_size and len(buffer) > 1:
            parts.append("\n".join(buffer))
            buffer = [header_text]
            buffer_len = len(header_text)
        buffer.append(row)
        buffer_len += len(row) + 1
    if len(buffer) > 1:
        parts.append("\n".join(buffer))
    return parts or [block]


def _pack(segments: List[_Segment], chunk_size: int, chunk_overlap: int) -> List[_Segment]:
    """把片段列表按 chunk_size 累积打包，并加上相邻重叠。

    表格块是"原子"的，不与相邻普通段落混合：
    实测把 20 多条故障码和一段散文拼进同一个 600 字块后，
    "P0300 是什么意思" 会召回那段只含范围引用 `| P0300–P03FF |` 的表格，
    而真正写着 "P0300 | 引擎曾经有失火现象" 的片段被稀释到排不进前列。
    """
    packed: List[_Segment] = []
    buffer: List[_Segment] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        text = "\n".join(seg.text for seg in buffer).strip()
        if text:
            # 章节取该块第一个片段所属章节，避免标题与正文错配
            packed.append(
                _Segment(
                    text=text,
                    section=buffer[0].section,
                    char_start=buffer[0].char_start,
                    char_end=buffer[-1].char_end,
                )
            )
        buffer, buffer_len = [], 0

    for seg in segments:
        seg_len = len(seg.text)
        if _looks_like_table(seg.text):
            # 表格单独成块；超长表按行切分并为每片补表头
            flush()
            for table_part in _split_table(seg.text, chunk_size):
                packed.append(
                    _Segment(
                        text=table_part,
                        section=seg.section,
                        char_start=seg.char_start,
                        char_end=seg.char_end,
                    )
                )
            continue
        if buffer and buffer_len + seg_len + 1 > chunk_size:
            flush()
        buffer.append(seg)
        buffer_len += seg_len + 1
        if buffer_len >= chunk_size:
            flush()
    flush()

    # 重叠：把上一块的尾部拼到当前块开头。
    # 当前块以表格开头时绝不拼接——否则会出现 "散文| P0300 | 引擎曾经失火" 这种
    # 把表格行挤在同一行里的非法 Markdown，表格结构和可读性都会被破坏。
    if chunk_overlap > 0 and len(packed) > 1:
        with_overlap: List[_Segment] = [packed[0]]
        for prev, cur in zip(packed, packed[1:]):
            cur_starts_with_table = _is_table_line(cur.text.strip().split("\n")[0])
            tail = prev.text[-chunk_overlap:]
            tail_last_line = tail.split("\n")[-1] if tail else ""
            if cur_starts_with_table or not tail or _is_table_line(tail_last_line):
                with_overlap.append(cur)
                continue
            with_overlap.append(
                _Segment(
                    text=f"{tail}\n{cur.text}",
                    section=cur.section,
                    char_start=max(0, cur.char_start - len(tail)),
                    char_end=cur.char_end,
                )
            )
        packed = with_overlap

    return packed


def _merge_tiny(segments: List[_Segment], min_chars: int) -> List[_Segment]:
    """把过短的片段并入前一块。

    合并跨越了两个章节时会保留出处的章节路径（如 "A + B"），
    避免短片段把上一块的章节标签"带偏"，让人工核对引用时难以定位。
    """
    if min_chars <= 0:
        return segments
    merged: List[_Segment] = []
    for seg in segments:
        if merged and len(seg.text) < min_chars:
            prev = merged[-1]
            prev.text = f"{prev.text}\n{seg.text}"
            prev.char_end = seg.char_end
            if seg.section and seg.section != prev.section:
                prev.section = f"{prev.section} + {seg.section}"
            continue
        merged.append(seg)
    return merged


def chunk_document(document: SourceDocument, settings: Settings, max_chars: int = 1200) -> List[Chunk]:
    """把一份文档切成带元数据的片段列表。"""
    text = normalize_text(document.text)
    if not text:
        return []

    # 注意：这里不再对 chunk_size 做"最小值"兜底。
    # 最小值校验统一由 Settings.normalize() 负责（并会写进 /health 的 config_warnings），
    # 否则本函数再夹一次 max() 就会把用户显式配置的小值静默改掉，
    # 表现为"我在 .env 里配了 80，实际却按 120 切"这种很难排查的问题。
    chunk_size = settings.chunk_size
    chunk_overlap = settings.chunk_overlap_effective
    # 上限仍然保留：防止单块过大导致 embedding 接口超长
    chunk_size = min(chunk_size, max_chars)

    segments: List[_Segment] = []
    heading_stack: List[str] = []
    cursor = 0
    current_section = document.title or "正文"
    buffer_lines: List[str] = []
    buffer_start = 0

    def flush_buffer() -> None:
        nonlocal buffer_lines, buffer_start
        block = "\n".join(buffer_lines).strip()
        buffer_lines = []
        if not block:
            return
        # 先把连续的表格行聚成"表格块"，再按空行切段落。
        # 不这样做的话，`P0300 | 引擎曾经有失火现象` 会变成独立小段，
        # 随后被打包阶段和几十条无关故障码混进同一个向量里，语义被稀释。
        for chunk_block in _group_into_blocks(block):
            if _looks_like_table(chunk_block):
                segments.append(
                    _Segment(
                        chunk_block, current_section, buffer_start, buffer_start + len(chunk_block)
                    )
                )
                continue
            for para in re.split(r"\n\s*\n", chunk_block):
                para = para.strip()
                if not para:
                    continue
                if len(para) <= chunk_size:
                    segments.append(
                        _Segment(para, current_section, buffer_start, buffer_start + len(para))
                    )
                else:
                    offset = buffer_start
                    for piece in _split_oversized(para):
                        if len(piece) <= chunk_size:
                            segments.append(
                                _Segment(piece, current_section, offset, offset + len(piece))
                            )
                        else:
                            sub = offset
                            for sent in split_sentences(piece):
                                segments.append(
                                    _Segment(sent, current_section, sub, sub + len(sent))
                                )
                                sub += len(sent)
                        offset += len(piece) + 1

    for raw_line in text.split("\n"):
        line_start = cursor
        cursor += len(raw_line) + 1
        heading = _heading_info(raw_line)
        if heading:
            flush_buffer()
            level, title = heading
            # 维护章节栈，形成 "一级 > 二级" 的路径
            while len(heading_stack) >= level:
                heading_stack.pop()
            heading_stack.append(title)
            current_section = " > ".join(heading_stack)
            continue
        if not buffer_lines:
            buffer_start = line_start
        buffer_lines.append(raw_line)
    flush_buffer()

    # 过滤空片段
    segments = [seg for seg in segments if seg.text.strip()]
    if not segments:
        return []

    packed = _pack(segments, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    packed = _merge_tiny(packed, min_chars=settings.chunk_min_chars)

    chunks: List[Chunk] = []
    seen_ids: Dict[str, int] = {}
    for index, seg in enumerate(packed):
        base_id = f"{document.doc_id}::{index}"
        if base_id in seen_ids:  # 理论上不会发生，做一层保险
            seen_ids[base_id] += 1
            base_id = f"{base_id}#{seen_ids[base_id]}"
        seen_ids[base_id] = 0
        chunks.append(
            Chunk(
                chunk_id=base_id,
                doc_id=document.doc_id,
                source=document.source,
                title=document.title,
                section=seg.section,
                position=index,
                text=seg.text,
                char_start=seg.char_start,
                char_end=seg.char_end,
                metadata={
                    **document.metadata,
                    "checksum": document.checksum,
                    "section": seg.section,
                },
            )
        )
    logger.debug("切分完成 source=%s chunks=%d", document.source, len(chunks))
    return chunks


def build_doc_id(relative_path: str) -> str:
    """文档 ID：路径哈希，保证同一文件多次入库 ID 稳定。"""
    return f"doc_{stable_hash(relative_path, 10)}"
