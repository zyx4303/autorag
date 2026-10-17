"""逐句归因校验（claim-level attribution）。

解决什么问题
------------
现有校验（`QAEngine.verify_citations`）只检查**引用编号是否越界**，
即 `[7]` 超出了实际 5 条片段这种"格式错误"。

但它管不了一类更隐蔽的幻觉：
    编号合法、格式正确，但**那句话的内容其实并不在被引用的片段里**。

例如片段只说了「发动机机油每 8,000 km 或 8 个月更换」，
模型却写成「机油每 8,000 km 更换，可延长发动机寿命约 30% [1]」——
`[1]` 是合法编号，但"延长寿命 30%"这句话在片段里根本不存在。
编号校验完全看不出来，用户也很难发现。

本模块把回答拆成句子，逐句检查"这句话的关键信息在被引用的片段中是否真能找到"，
把结果透明地暴露给用户，而不是假装一切正常。

设计取舍
--------
1. **确定性、可解释**：用词覆盖而非再调一个 LLM 判定。
   模型判模型会引入新的不确定性，且无法解释"为什么判它没依据"。
   这里每个句子都能给出具体的覆盖率数值，人能复核。
2. **分级而非二值**：
   - `supported`   覆盖率 >= 阈值高线，基本可放心
   - `weak`        覆盖率中等，多半是同义改写，建议复核
   - `unsupported` 覆盖率接近 0，很可能无依据（幻觉嫌疑）
   - `no_citation` 整句没有任何引用角标
3. **只报告不拦截**：默认不改变回答内容，只加 `warnings` 并在响应的
   `attribution` 字段里给明细。是否采信由使用者判断 —— 硬性删改回答
   会破坏可读性，也可能误删正确内容。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

from app.core.text_utils import tokenize
from app.logging_conf import get_logger
from app.models import Citation, RetrievedChunk

logger = get_logger(__name__)

# 「依据：」之后是来源清单，不是正文断言，不参与归因判断
_BODY_SPLIT = re.compile(r"^[ \t]*依据[：:]", re.MULTILINE)
_CITATION = re.compile(r"\[(\d{1,2})\]")
_CJK = re.compile(r"[\u4e00-\u9fff]")

# 括号对：切分时要跳过括号内部的句末标点
_OPEN = "（(【「"
_CLOSE = "）)】」"
# 只把真正的句末标点当句子边界。
# 刻意**不包含** `；`：模型常写成「…缩短保养间隔；仅偶尔…仍按正常条件执行 [1]」，
# 分号只是句内并列，引用标在整句末尾。若按 `；` 切句，前半截会被误判成"未标引用"。
_SENT_END = "。！？!?"


def _split_sentences_safe(text: str) -> List[str]:
    """按句末标点切分，但**跳过括号内部**的标点。

    为什么不能直接用 text_utils.split_sentences：
    它把 `；` 也当句末标点，于是
    「若长期处于恶劣条件（如单次里程不足 8 km；长期 35℃ 以上高温；……），
      应按正常间隔的 3/4 缩短保养间隔 [4]」
    会在括号内的分号处被切断，导致 `[4]` 只落在最后一段，
    前面的残句被误判成"未标注引用"。实测误报率约 2/14。

    注意 `[1]` 这类引用角标也是方括号，但它成对出现、进出抵消，
    不会破坏括号深度；只有 `【` 与 `[` 共用深度计数带来的影响可忽略。
    """
    out: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in text or "":
        if ch in _OPEN:
            depth += 1
        elif ch in _CLOSE:
            depth = max(0, depth - 1)
        buf.append(ch)
        if depth == 0 and ch in _SENT_END:
            piece = "".join(buf).strip()
            if piece:
                out.append(piece)
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out

# 单个中文字符的权重：与 retriever / bm25 保持同一原则 ——
# 单字命中几乎不构成"有依据"的证据（任意两句中文都可能共享"的/是/了"）。
SINGLE_CJK_WEIGHT = 0.25

# 纯过渡语 / 版式语句，不是事实断言，跳过校验
_TRANSITIONAL = {
    "补充说明", "依据", "说明", "注", "提示", "建议", "结论",
    "建议确认渠道", "可提供的相关资料如下", "相关资料如下", "此外", "另外",
}

# 元话语（meta-statement）：模型在**谈论资料本身**而不是断言事实。
# 这类句子天然没有出处，要求它标引用会产生大量误报 ——
# 实测拒答场景下 8/16 个"无引用"全是这种话术。
_META_PATTERNS = (
    r"未找到",
    r"无法给出",
    r"无法提供",
    r"无法判断",
    r"不构成.{0,12}依据",
    r"仅为.{0,16}(含义|定义|方向|参考)",
    r"确认渠道",
    r"授权服务站",
    r"厂家客服",
    r"(手册|说明书|服务站|客服|厂家).{0,12}为准",
    r"上述内容",
    r"以上内容",
    r"本回答",
    # 描述数据性质/来源的元陈述（不是事实断言）
    r"属虚构",
    r"示例数据",
    r"虚构数据",
    r"虚构内容",
)

# 否定性陈述：模型在陈述"资料里没有 X"。
# 这个判断本身无法也不该用覆盖率验证（它描述的是缺失），跳过。
_NEGATION_PATTERNS = (
    r"未涉及",
    r"未提及",
    r"未包含",
    r"未覆盖",
    r"并没有",
    r"均未",
    r"并非",
    r"没有涉及",
    r"不含.{0,8}(价格|费用|参数|数据)",
)

SUPPORTED = "supported"
WEAK = "weak"
UNSUPPORTED = "unsupported"
NO_CITATION = "no_citation"

__all__ = [
    "SentenceAttribution",
    "AttributionReport",
    "AttributionChecker",
    "SUPPORTED",
    "WEAK",
    "UNSUPPORTED",
    "NO_CITATION",
]


@dataclass
class SentenceAttribution:
    """单个句子的归因结论。"""

    text: str
    citations: List[int] = field(default_factory=list)
    coverage: float = 0.0
    status: str = SUPPORTED
    matched_terms: int = 0
    total_terms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text[:200],
            "citations": self.citations,
            "coverage": round(self.coverage, 4),
            "status": self.status,
        }


@dataclass
class AttributionReport:
    """整段回答的归因报告。"""

    total: int = 0
    supported: int = 0
    weak: int = 0
    unsupported: int = 0
    no_citation: int = 0
    score: float = 1.0
    sentences: List[SentenceAttribution] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def checked(self) -> int:
        """参与校验的句子数（排除过渡语）。"""
        return self.total

    def to_dict(self, include_sentences: bool = True) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "total": self.total,
            "supported": self.supported,
            "weak": self.weak,
            "unsupported": self.unsupported,
            "no_citation": self.no_citation,
            "score": round(self.score, 4),
            "warnings": self.warnings,
        }
        if include_sentences:
            data["sentences"] = [
                s.to_dict()
                for s in self.sentences
                if s.status in {WEAK, UNSUPPORTED, NO_CITATION}
            ]
        return data


class AttributionChecker:
    """逐句校验回答是否有片段支撑。"""

    def __init__(
        self,
        weak_threshold: float = 0.35,
        unsupported_threshold: float = 0.12,
        min_clause_chars: int = 6,
    ) -> None:
        self._weak_threshold = max(0.01, float(weak_threshold))
        self._unsupported_threshold = max(0.0, float(unsupported_threshold))
        self._min_clause_chars = max(1, int(min_clause_chars))

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    @staticmethod
    def split_body(answer: str) -> str:
        """截掉「依据：」及其后的来源清单，只留正文断言。"""
        match = _BODY_SPLIT.search(answer or "")
        return (answer or "")[: match.start()] if match else (answer or "")

    def _split_clauses(self, sentence: str) -> List[str]:
        """把句子按中文逗号/分号拆成更小的断言单元。

        为什么必须下沉到分句级：
        实测反例 —— 「发动机机油每 8,000 km 更换 [1]，可延长发动机寿命约 30% [1]」
        前半句与片段高度吻合，后半句是彻头彻尾的幻觉，但**整句**的加权覆盖率
        仍然很高（真话的词多），句子级校验会把它判成"有依据"，直接漏检。
        拆成分句后，后半句单独计算覆盖率会掉到接近 0，就能被抓出来。

        两条边界：
        1. 括号内的分隔符不切（"（如 A；B；C）" 是一个整体）；
        2. 英文逗号若紧跟数字，视为千分位（"8,000"）不切。
        """
        out: List[str] = []
        buf: List[str] = []
        depth = 0
        chars = list(sentence or "")
        for i, ch in enumerate(chars):
            if ch in _OPEN:
                depth += 1
            elif ch in _CLOSE:
                depth = max(0, depth - 1)
            buf.append(ch)

            if depth != 0:
                continue
            is_sep = ch in "，；;"
            if not is_sep and ch == ",":
                nxt = chars[i + 1] if i + 1 < len(chars) else ""
                is_sep = not nxt.isdigit()  # 千分位不算分隔
            if is_sep:
                piece = "".join(buf[:-1]).strip(" \t。.！!？?")
                if piece:
                    out.append(piece)
                buf = []

        tail = "".join(buf).strip(" \t。.！!？?")
        if tail:
            out.append(tail)

        # 合并过短的分句：枚举型内容（"检查火花塞、点火线圈、喷油嘴"）会被
        # 逗号切成多个 3~5 字的碎片，单独算覆盖率没有意义（碎片本身不构成
        # 完整断言，覆盖率会异常偏低），合并回相邻分句更准确。
        merged: List[str] = []
        pending = ""
        for piece in out:
            pending = f"{pending}，{piece}" if pending else piece
            if len(pending) >= self._min_clause_chars:
                merged.append(pending)
                pending = ""
        if pending:
            if merged:
                merged[-1] = f"{merged[-1]}，{pending}"
            else:
                merged.append(pending)
        return merged

    @staticmethod
    def _weighted_coverage(terms: Sequence[str], passage_tokens: Set[str]) -> float:
        """句子关键词在片段中的加权覆盖率（单字降权）。

        刻意不做 IDF：这里要回答的是"这些词在片段里有没有出现"，
        而不是"这些词有多重要"。保持可解释。
        """
        if not terms:
            return 0.0
        weights = [
            SINGLE_CJK_WEIGHT if len(t) == 1 and _CJK.match(t) else 1.0 for t in terms
        ]
        total = sum(weights)
        if total <= 0:
            return 0.0
        hit = sum(w for t, w in zip(terms, weights) if t in passage_tokens)
        return hit / total

    @staticmethod
    def _is_transitional(sentence: str) -> bool:
        """判断是否是过渡语 / 元话语 / 否定陈述（不是可校验的事实断言）。"""
        core = sentence.strip().strip("：:，,。.！!？?、；; ")
        if not core:
            return True
        # 去掉列表序号后判断
        core = re.sub(r"^[\d一二三四五六七八九十]+[.、)）]\s*", "", core).strip()
        if not core:
            return True
        for word in _TRANSITIONAL:
            if core == word or (core.startswith(word) and len(core) <= len(word) + 4):
                return True
        # 元话语 / 否定陈述：描述"资料本身"而非断言事实，不要求引用
        for pattern in _META_PATTERNS:
            if re.search(pattern, core):
                return True
        for pattern in _NEGATION_PATTERNS:
            if re.search(pattern, core):
                return True
        # 太短的残句不具判断价值
        if len(core) <= 6:
            return True
        return False

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    def check(
        self,
        answer: str,
        chunks: Sequence[RetrievedChunk],
        citations: Optional[Sequence[Citation]] = None,
        match_any: bool = False,
    ) -> AttributionReport:
        """对回答做逐句归因。

        Args:
            answer: 模型返回的完整回答
            chunks: 本次召回的全部片段（用 chunk_id 或顺序号对齐引用编号）
            citations: 可选的引用元数据，仅用于兜底对齐
            match_any: 忽略 `[n]` 编号，只要**任意**片段能支撑该句就算有依据。

                Agent 场景必须开启：Agent 可能多次调用 search_kb，每次返回的
                片段编号都从 1 重新开始，`[2]` 到底指向哪一批的第 2 条无法确定。
                此时按编号对齐会张冠李戴，只能退化为"在全量已检索片段里找依据"。
                （代价是判定略宽松，但不会把有依据的句子误判成幻觉。）
        """
        report = AttributionReport()
        body = self.split_body(answer)
        if not body.strip():
            return report

        # 引用编号 -> 片段正文。build_context 是按 results 顺序编号的（从 1 开始）
        passage_by_index: Dict[int, str] = {
            i + 1: (chunk.text or "") for i, chunk in enumerate(chunks)
        }
        # 若传了 citations，用它对一遍（citations 的 index 就是展示给用户的编号）
        if citations:
            by_chunk_id = {c.chunk_id: (c.snippet or "") for c in citations}
            for c in citations:
                if c.chunk_id in by_chunk_id and c.index not in passage_by_index:
                    passage_by_index[c.index] = by_chunk_id[c.chunk_id]

        token_cache: Dict[int, Set[str]] = {
            idx: set(tokenize(text)) for idx, text in passage_by_index.items()
        }

        for raw in _split_sentences_safe(body):
            sentence = raw.strip()
            if not sentence or self._is_transitional(sentence):
                continue

            # 引用在**句子级**提取，再让各分句继承：
            # 模型常常只在句末标一次 [1]，若按分句各取各的，
            # 前半句会被误判成"未标引用"。
            indices = [int(m) for m in _CITATION.findall(sentence)]
            seen: Set[int] = set()
            uniq = [i for i in indices if not (i in seen or seen.add(i))]

            for clause in self._split_clauses(sentence):
                clean = _CITATION.sub("", clause).strip()
                if len(clean) < self._min_clause_chars:
                    continue
                terms = [t for t in tokenize(clean) if t]
                if not terms:
                    continue

                if not uniq:
                    report.sentences.append(
                        SentenceAttribution(text=clause, status=NO_CITATION)
                    )
                    report.no_citation += 1
                    report.total += 1
                    continue

                # 只要被引用的**任一**片段能覆盖，就算有依据（一句话可引多个来源）
                # match_any 时忽略编号，在全量片段里取最佳覆盖（Agent 场景专用）
                if match_any:
                    best = max(
                        (
                            self._weighted_coverage(terms, tokens)
                            for tokens in token_cache.values()
                            if tokens
                        ),
                        default=0.0,
                    )
                else:
                    best = 0.0
                    for idx in uniq:
                        tokens = token_cache.get(idx, set())
                        if tokens:
                            best = max(best, self._weighted_coverage(terms, tokens))

                if best >= self._weak_threshold:
                    status = SUPPORTED
                    report.supported += 1
                elif best >= self._unsupported_threshold:
                    status = WEAK
                    report.weak += 1
                else:
                    status = UNSUPPORTED
                    report.unsupported += 1

                report.total += 1
                report.sentences.append(
                    SentenceAttribution(
                        text=clause,
                        citations=uniq,
                        coverage=best,
                        status=status,
                    )
                )

        if report.total:
            # 得分：完全支持记 1，弱支持记 0.5，其余记 0
            report.score = (
                report.supported + 0.5 * report.weak
            ) / report.total

        self._build_warnings(report)
        return report

    # ------------------------------------------------------------------ #
    # 警告文案
    # ------------------------------------------------------------------ #
    def _build_warnings(self, report: AttributionReport) -> None:
        if report.unsupported:
            samples = [s.text[:24] for s in report.sentences if s.status == UNSUPPORTED][:2]
            report.warnings.append(
                f"有 {report.unsupported} 句内容在被引用的片段中找不到对应依据"
                f"（示例：{'；'.join(samples)}），可能是模型补充了资料外的信息，请人工核对。"
            )
        if report.no_citation:
            report.warnings.append(
                f"有 {report.no_citation} 句结论未标注任何引用来源，无法追溯依据。"
            )
        if report.total and report.score < 0.6:
            report.warnings.append(
                f"整体归因得分偏低（{report.score:.2f}），本次回答依据不足的情况较多，建议优先核对引用原文。"
            )


def build_checker(settings: Any) -> Optional[AttributionChecker]:
    """按配置构造校验器；未启用时返回 None。"""
    if not getattr(settings, "attribution_enabled", False):
        return None
    return AttributionChecker(
        weak_threshold=getattr(settings, "attribution_weak_threshold", 0.35),
        unsupported_threshold=getattr(settings, "attribution_unsupported_threshold", 0.12),
    )
