"""索引侧同义别名词表：把用户的说法写进索引文本，修词汇鸿沟。

背景（README 12.14 / 12.24）
----------------------------
Q001「首保多少公里」的病根是**词汇鸿沟**：文档 79 个片段里「首保」出现 0 次，
只有「第 1 次」。12.14 试过**查询侧**同义扩展（把「首保」扩成「首次保养」再检索），
BM25 上有效但端到端排名没变——rrf_k=60 把名次差压平了。

这里的思路反过来：**在索引侧把别称写进索引文本**（「第 1 次…（同义表述：首保）」）。
用户说法与文档说法在索引里直接共存，查询侧一行不改，天然参与 BM25 / 向量 /
精排的全部打分。适合本仓库这种受控小语料。

纪律
----
- 每条别称必须是**无歧义的同义词**，且能说出演示语料里的对应物；
  12.14 明确否掉过「刹车片 ↔ 制动器」（语料里「检查前后制动器」讲的是检查周期，
  扩过去会让本应拒答的毫米阈值问题拿到看似相关的证据）。
- 别称只进**索引文本**（build_context_text 的产物），Chroma 存的原文、
  引用展示、片段正文一律不动——展示层永远干净。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# (触发词, 上下文要求, 别称列表)
# 上下文要求 = 触发词必须和这些词**同时**出现在片段里才生效，
# 用于消歧：「第 1 排」（P0171 混合气过稀）里的「第 1 次」别成一个都没有，
# 「第 1 次」必须同片段出现「保养」才算保养节点。
_ENTRIES: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = [
    ("变速器油", (), ("变速箱油",)),
    ("制动液", (), ("刹车油",)),
    ("冷却液", (), ("防冻液",)),
    ("包修期", (), ("保修期",)),
    ("第 1 次", ("保养",), ("首保", "首次保养")),
]


def alias_line(text: str) -> str:
    """按片段内容收集命中的别称，返回拼好的别称行；无命中返回空串。

    返回形如「（同义表述：首保、首次保养）」，由 build_context_text 拼进索引文本。
    """
    if not text:
        return ""
    aliases: List[str] = []
    for trigger, requires, terms in _ENTRIES:
        if trigger not in text:
            continue
        if any(req not in text for req in requires):
            continue
        for term in terms:
            if term not in aliases:
                aliases.append(term)
    if not aliases:
        return ""
    return "（同义表述：" + "、".join(aliases) + "）"


def entries() -> List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]]:
    """暴露词表（只读用途：测试、文档生成）。"""
    return list(_ENTRIES)


__all__ = ["alias_line", "entries"]
