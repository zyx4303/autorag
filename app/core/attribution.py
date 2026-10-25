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

from app.core.text_utils import build_context_text, tokenize
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
    # 「示例车型 A 的数据为虚构示例」这类对数据性质的说明
    r"(数据|内容|条款|数值).{0,8}(为|是)?.{0,4}虚构",
    # 适用范围声明：也是"谈论资料本身"，词面覆盖同样验证不了。
    # 刻意要求后面紧跟「示例/虚构/文档/资料/原文/真实」，
    # 这样「仅适用于涡轮增压车型」这类真正的适用性事实仍会被校验。
    r"(仅适用|只适用|不适用|仅针对|仅覆盖).{0,6}(示例|虚构|文档|资料|原文|真实|该车型|本车型|此车型|其他车型|其他品牌|同级车型)",
    # ---- 列表引导语 / 承接语：不承载事实，是组织语言 ----
    # 「仅能提供…」「仅能确认…」「以及常规定期检查项目」
    r"^(仅|只)(能|可)(提供|确认|给出|列出|说明)",
    r"^以及.{0,16}$",
    r"(供|作|作为).{0,8}(参考|线索|方向|提示)",
    # ---- 适用边界：说明「结论不能外推到什么范围」，不是断言 ----
    r"^但?不(代表|意味着)",
    r"^但?不可.{0,10}(互相|相互|直接|一概)?(套用|适用|外推|等同)",
    # ---- 建议 / 假设 / 限制：是行动指引，不是断言 ----
    r"^[如若](需|想|要|有|遇)",
    r"^(需|应|请|要注意)(注意)?区分",
    r"^或以.{0,20}为准",
    r"^或(关注|查询|咨询|参见|浏览)",
    r"不构成.{0,14}(清单|结论|依据|诊断)",
    r"(不足以|难以|无法).{0,10}(给出|判断|确定|得出|支持|形成)",
    # ---- 评述：在比较两套依据，不是陈述资料里的事实 ----
    r"^(两者|二者|双方).{0,12}(不一致|不符|不同|有差异)",
    r"(列举|表述|措辞|数值|说法).{0,12}(不.{0,4}一致|不符|有出入|有差异)",
    # ---- 对「本回答自身结构」的指代，不是对资料的断言 ----
    r"^上述.{0,8}(项|条目|条目|编号)",
    # 疑问式主题句：把问题复述成「是否存在…」，没有可校验的事实
    r"是否.{0,20}(要求|需要|必须)$",
    # 疑问性短语（「具体适用哪一数值」）不是事实断言
    r"^(具体)?(适用|采用|取|是)?哪(一|个|些).{0,12}$",
)

# 否定性陈述：模型在陈述"资料里没有 X"。
# 这个判断本身无法也不该用覆盖率验证（它描述的是缺失），跳过。
_NEGATION_PATTERNS = (
    r"未涉及",
    r"未提及",
    r"未包含",
    r"未覆盖",
    r"未收录",
    r"未记录",
    r"并没有",
    r"均未",
    r"并非",
    r"没有涉及",
    r"不含.{0,8}(价格|费用|参数|数据)",
    # 「也没有 2027 款未上市车型的参数信息」这类承前省略的缺口陈述
    r"(也)?(没有|无).{0,18}(信息|参数|数据|说明|清单|依据|图例)",
)

# 「语料缺口」陈述：主语是资料/文档/表格，谓语是「没有 / 仅有」。
#
# 为什么必须单独识别：用词面覆盖率**在原理上无法验证「缺失」**。
# 「该表未标注其是否等同于首保」这句话即使完全正确，覆盖率也必然是 0 ——
# 因为被指的恰恰是"不存在的东西"，没有词可以匹配。
#
# 实测（2026-10-25，把过渡语判定下沉到分句级之后）：
# 全部 5 个 unsupported 句子**都是这一类**，没有一个是真正的编造。
# 若不识别，诚实说明边界的回答会被系统性地打成"依据不足"。
#
# 判据刻意同时要求「语料指称」+「缺失标记」，而不是只匹配 未/无：
# 「该车型未配备胎压监测」这种对车辆本身的否定仍应被校验，不能放过。
_CORPUS_REF = r"(资料|文档|文件|知识库|语料|原文|表格|清单|条目|章节|段落|该表|本表|上表|上述表)"
_ABSENCE_MARK = r"(未|没有|无|不含|不涉及|不包含|仅有|只有|仅含|仅包含|仅涉及|仅包括|仅涵盖|仅能|仅可|仅列|仅给)"

# 主语被省略的缺口陈述（承前省略在中英文里都常见）
_ELIDED_ABSENCE_PATTERNS = (
    r"^且?(未|没有)(对|就).{0,14}(作|做|给出|提供).{0,8}(规定|说明|解释|区分)",
    r"^且?(未|没有)(给出|提供|说明|标注|注明).{0,14}",
    r"^但?(不能|不可)(据此|作为|直接|用于|因此).{0,14}(推断|依据|结论|诊断|判断|判定|等同|套用|适用|沿用|外推|原因)",
    r"^不(能|可)作为.{0,14}(依据|结论|诊断|判断|凭证)",
)

# 分句开头的小标题（Markdown 粗体标题 / # 标题 / 中文序号小标题）。
# 模型常把「**三、需注意的差异**」和正文粘在同一个分句里（换行被规整掉了），
# 整句按标题处理会连正文断言一起丢掉，所以只剥前缀、不丢内容。
_HEADING_PREFIX_RE = re.compile(
    r"^\s*(?:#{1,6}\s*[^\n]{0,40}|\*\*[^*\n]{1,30}\*\*\s*[：:]?\s*"
    r"|[一二三四五六七八九十]+、[^\n]{0,20}\n)\s*"
)
# 整句只有标题、没有正文
_HEADING_ONLY_RE = re.compile(r"^\s*(?:#{1,6}\s*\S.{0,40}|\*\*[^*\n]{1,30}\*\*)\s*[：:]?\s*$")


def strip_clause_heading(clause: str) -> str:
    """剥掉分句开头的小标题，只保留正文断言。

    实测（2026-10-25）有一句长这样：
        `**三、需注意的差异**\n示例车型 A 的包修期里程为 100,000 km`
    前半是小标题、后半是**带数字的事实断言**。整句跳过会把断言漏掉；
    整句校验又会让标题文字拉低覆盖率。剥前缀是唯一两头都不亏的做法。
    """
    text = (clause or "").strip()
    if not text:
        return ""
    if _HEADING_ONLY_RE.match(text):
        return ""
    return _HEADING_PREFIX_RE.sub("", text, count=1).strip()


SUPPORTED = "supported"
WEAK = "weak"
UNSUPPORTED = "unsupported"
NO_CITATION = "no_citation"


def is_corpus_absence_claim(clause: str) -> bool:
    """分句是否在陈述「语料里没有 / 只有 X」。

    这类句子不能用词面覆盖率验证：被指的东西本来就不存在，没有词可匹配，
    覆盖率恒为 0。判它 unsupported 等于把「诚实说明边界」当成「编造」。

    两种句式都覆盖：
    - 有主语：「该表未标注…」「资料中仅有…」
    - 承前省略：「且未对…作专门规定」「不能作为…结论」
    """
    core = (clause or "").strip().strip("：:，,。.！!？?、；; ")
    if not core:
        return False
    if re.search(_CORPUS_REF, core) and re.search(_ABSENCE_MARK, core):
        return True
    return any(re.search(pattern, core) for pattern in _ELIDED_ABSENCE_PATTERNS)


__all__ = [
    "SentenceAttribution",
    "AttributionReport",
    "AttributionChecker",
    "SUPPORTED",
    "WEAK",
    "UNSUPPORTED",
    "NO_CITATION",
    "is_corpus_absence_claim",
    "strip_clause_heading",
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
    # 被判定为「语料缺口陈述」而跳过校验的分句数。
    # 单列出来是为了可见：它既不进分子也不进分母，
    # 如果不报，读者会以为分母就是全部句子。
    meta_skipped: int = 0
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
            "meta_skipped": self.meta_skipped,
            "score": round(self.score, 4),
            "warnings": self.warnings,
        }
        if include_sentences:
            # 输出**全部**被校验的句子，不只问题句。
            # 曾经只给 weak/unsupported/no_citation —— 对「归因不足为什么」够用，
            # 但句级引用高亮需要的是「每句有哪段依据」，supported 句恰恰是主体，
            # 过滤掉它们高亮就无从做起（实测发现）。消费方按 status 自行过滤。
            data["sentences"] = [s.to_dict() for s in self.sentences]
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
        #
        # 校验面 = 片段正文 + 章节路径（2026-10-25 修改，原因见下）。
        # 只用正文会把「写在章节标题里」的真话判成编造：实测多轮评测 MT2，
        # 模型答「…以先到者为准 [1]」—— 这句话在文档的章节标题
        # 「二、定期保养项目表（按里程 / 时间，以先到者为准）」里，
        # 但被引的片段恰好是纯表格，正文没有它。2/4 句「无依据」触发编造检测，
        # 把一个完全正确的回答强制改成了拒答。
        #
        # 为什么不怕放宽：章节路径是文档真实的标题文字，用户顺着引用本来就能看到它；
        # 幻觉要恰好编出与标题重合的词才能蒙混，成本远高于收益。
        # （不用 title：文档标题每段都一样，只会无差别放宽。）
        passage_by_index: Dict[int, str] = {
            i + 1: build_context_text(chunk.text or "", chunk.section or "")
            for i, chunk in enumerate(chunks)
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
            if not sentence:
                continue
            # 注意：这里**不能**按整句判过渡语。
            # 「资料中未找到 X；但 Y 为 Z [6]」这种句子前半句在谈资料缺口、
            # 后半句是实打实的事实断言。按整句跳过会把断言一起漏掉 ——
            # 实测 20 条评测里有 10 条因此归因为空（total=0），
            # 而那恰恰是幻觉最容易混进去的「先声明缺口、再给断言」句式。
            # 过渡语判定下沉到分句级，见下方循环。

            # 引用在**句子级**提取，再让各分句继承：
            # 模型常常只在句末标一次 [1]，若按分句各取各的，
            # 前半句会被误判成"未标引用"。
            indices = [int(m) for m in _CITATION.findall(sentence)]
            seen: Set[int] = set()
            uniq = [i for i in indices if not (i in seen or seen.add(i))]

            for raw_clause in self._split_clauses(sentence):
                # 先剥掉开头的小标题（「**三、需注意的差异**\n正文」这种粘连），
                # 只跳过纯标题，正文断言照常校验。
                clause = strip_clause_heading(raw_clause)
                if not clause:
                    report.meta_skipped += 1
                    continue
                clean = _CITATION.sub("", clause).strip()
                if len(clean) < self._min_clause_chars:
                    continue
                terms = [t for t in tokenize(clean) if t]
                if not terms:
                    continue
                # 过渡语 / 元话语 / 否定陈述在**分句**级判：只跳过真正在谈资料本身的那半句，
                # 同一句里的事实断言照常校验。
                if self._is_transitional(clause):
                    continue
                if is_corpus_absence_claim(clause):
                    report.meta_skipped += 1
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
