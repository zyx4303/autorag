"""声明（Claim）模型：把文档里的散文拆成"可比较的最小事实单元"。

为什么需要它（这是本项目相对普通 RAG 的核心区别）：
普通 RAG 把文档切块后做向量检索，回答的是"哪块内容语义相近"；
但文档一旦被修改（保养周期从 8000 km 改成 10000 km），
向量检索**无法告诉你**：以前的答案里哪几句已经失效、哪些回答需要重新生成。

因此在片段之上再加一层"原子化声明"：
    主体(subject) - 属性(predicate) - 值(value) - 限定条件(qualifiers)
每条声明绑定来源文档、章节、字符区间与文档版本哈希。
有了它，"文档改动影响了什么"就变成一次可以在毫秒级完成的结构化比对。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# 声明类型：决定后续如何比较
#   interval  周期/里程类（如"更换机油：每 8000 km 或 8 个月"）—— 值可数值化比较
#   spec      规格/参数类（如"机油黏度：5W-30"）
#   threshold 阈值类（如"刹车片剩余厚度低于 3mm 需更换"）
#   condition 条件/适用范围（如"恶劣条件下缩短至 3/4"）
#   policy    条款/政策（如"包修期：3 年或 60000 km"）
#   code      故障码定义（如"P0420：催化器效率低于阈值"）
CLAIM_TYPES = ("interval", "spec", "threshold", "condition", "policy", "code")

# 比较策略：interval/threshold/policy 的数值变化属于"实质失效"；
# spec 允许同义表述；condition 通常是限定条件的增删。
NUMERIC_TYPES = {"interval", "threshold", "policy"}


@dataclass
class Claim:
    """一条原子化声明。"""

    claim_key: str                      # 稳定标识（doc + 类型 + 主体 + 属性），用于跨版本对齐
    doc_source: str = ""                # 来源文档（相对路径）
    section: str = ""                   # 章节路径
    claim_type: str = "spec"
    subject: str = ""                   # 主体，如"更换发动机机油"
    predicate: str = ""                 # 属性，如"周期"
    value: str = ""                     # 值，如"每 8,000 km 或每 8 个月"
    value_normalized: Optional[str] = None  # 归一化值（提取数值+单位），便于比较
    qualifiers: List[str] = field(default_factory=list)  # 限定条件，如 ["以先到者为准"]
    evidence: str = ""                  # 原文依据（用于人工核对，禁止无依据抽取）
    chunk_id: str = ""                  # 来自哪个片段
    char_start: int = 0
    char_end: int = 0
    doc_checksum: str = ""              # 抽取时的文档版本哈希
    extracted_at: str = ""
    model: str = ""

    def __post_init__(self) -> None:
        """自动补全归一化值。

        为什么要做成自动的：差别判定依赖 value_normalized，
        但只要有一处构造时忘记赋值，那条声明就会永远被误判为"已修改"
        （因为拿 None 去比）。把它放在模型里，就不依赖调用方记性了。
        """
        if self.value and not self.value_normalized:
            self.value_normalized = normalize_value(self.value)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def build_key(doc_source: str, claim_type: str, subject: str, predicate: str) -> str:
        raw = f"{doc_source}|{claim_type}|{subject}|{predicate}"
        return "ck_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def normalize_value(value: str) -> Optional[str]:
    """把"每 8,000 km 或每 8 个月（以先到者为准）"归一化成 "8000km|8month"。

    只做机械归一化（去空格、统一单位、抽取数字），不做语义推理——
    宁可保守：拿不准就返回 None，由上层按"值文本不同"处理。
    """
    if not value:
        return None
    text = value.lower().replace(",", "").replace("，", "")
    import re

    parts: List[str] = []
    # 里程：8000 km / 8000km / 8 万公里
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(万)?\s*(km|公里|千米)", text):
        number = float(match.group(1))
        if match.group(2):
            number *= 10000
        parts.append(f"{number:g}km")
    # 时间：8 个月 / 2 年
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(个月|月|年)", text):
        number = float(match.group(1))
        unit = "month" if match.group(2) in ("个月", "月") else "year"
        parts.append(f"{number:g}{unit}")
    # 毫米/百分比等阈值
    for match in re.finditer(r"(\d+(?:\.\d+)?)\s*(mm|毫米|%)", text):
        parts.append(f"{float(match.group(1)):g}{match.group(2)}")
    if not parts:
        return None
    return "|".join(sorted(set(parts)))


def classify_change(old: Optional[Claim], new: Optional[Claim]) -> str:
    """给出变更类型。

    返回五种之一：
      added            新增声明
      removed          删除声明
      modified         值发生实质变化（用户最需要关注的）
      qualifier_changed 只有限定条件变化，值没变
      unchanged        完全一致

    为什么要把 qualifier_changed 单独拆出来：实测发现大模型抽取限定条件时**不稳定**
    （同一个"首次"限定，两次抽取一次有一次没有），如果把它并入 modified，
    审核清单里会混进大量"值没变的假修改"，真正重要的数值变更就被淹没了。
    单独标出后，值变更与限定条件变更可以分别统计、分别处置。
    """
    if old is None and new is not None:
        return "added"
    if old is not None and new is None:
        return "removed"
    if old is None and new is None:
        return "unchanged"
    assert old is not None and new is not None

    same_qualifiers = sorted(old.qualifiers) == sorted(new.qualifiers)

    # ① 值文本完全一致
    if old.value.strip() == new.value.strip():
        return "unchanged" if same_qualifiers else "qualifier_changed"

    # ② 值文本不同但归一化后一致（"8,000 km" vs "8000km"）
    old_norm = old.value_normalized or normalize_value(old.value)
    new_norm = new.value_normalized or normalize_value(new.value)
    if old_norm and new_norm and old_norm == new_norm:
        return "unchanged" if same_qualifiers else "qualifier_changed"

    # ③ 归一化结果不同 → 实质修改
    if old_norm and new_norm:
        return "modified"

    # ④ 至少一方无法归一化 → 只能按文本判为修改
    return "modified"


def severity_of(change: str, claim_type: str) -> str:
    """评估变更严重度，用于面板排序与告警分级。

    规则（保守、可解释）：
      modified          数值型声明（周期/阈值/条款）→ high；其他 → medium
      removed           数值型 → high；其他 → medium
      qualifier_changed → medium（适用范围变了，容易被忽略，但值本身没动）
      added             → low（只是补充信息）
    """
    if change == "modified":
        return "high" if claim_type in NUMERIC_TYPES else "medium"
    if change == "removed":
        return "high" if claim_type in NUMERIC_TYPES else "medium"
    if change == "qualifier_changed":
        return "medium"
    if change == "added":
        return "low"
    return "none"
