"""检索与生成侧的统一评测指标。

为什么要单独一个模块
--------------------
在这之前，「来源命中」和「关键词覆盖」在 `app/services/eval_service.py` 与
`scripts/run_eval.py` 里各写了一份实现。两份的判定细节一旦分叉，
服务端报告和本地脚本报告会给出不同的数字，而且很难有人发现。

本模块是**唯一口径源**：纯标准库、纯函数、无状态。
服务层和脚本层都从这里取，测试也只测这里。

口径必须诚实说明的三件事
------------------------
1. **相关性可以有两档粒度，由评测集决定**。
   - 标注了 `expected_snippets`（关键原文短语）→ **片段级**：
     相关 = 来源文件命中 **且** 片段正文含该短语，回答的是「讲这件事的那一段有没有被找出来」；
   - 没标注 → **文件级**：同文件名下的任意片段都算相关。
     这档粒度在 k>=5 时几乎必然满分（实测来源命中率 1.0 / Recall@8 1.0），
     区分不出排序好坏，只适合在没有标注时兜底。

   两档粒度的数字**不能混着比**。报告里用 `relevance_level` 标明本次用的是哪一档。

2. **nDCG 的理想排序只在召回集内计算**。标准 nDCG 的 IDCG 需要
   「全库里所有相关项」，这个值未知；这里退化为
   「把本次召回到的相关项全部排到最前」。因此 nDCG=1.0 只代表
   「召回集内部排序已经最优」，不代表没有漏召。漏召由 recall 反映，
   两个指标必须一起看，单看任何一个都会被误导。

3. **全部为规则统计**（唯一的例外是直接透传 attribution 模块给出的归因得分），
   不涉及 LLM 判定，也**不是准确率**，不与任何其他系统对比。

命名约定
--------
- `@k` 系列只看前 k 条召回结果；
- 返回 0~1 的浮点，无样本时返回 None（而不是 0.0 —— 0.0 会被误读成「全错」）。
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from app.core.text_utils import tokenize

__all__ = [
    "OK",
    "ERROR",
    "WRONG_REFUSAL",
    "MISSED_REFUSAL",
    "RETRIEVAL_MISS",
    "RANKING_LOW",
    "CITATION_OUT_OF_RANGE",
    "ATTRIBUTION_WEAK",
    "ANSWER_MISMATCH",
    "EVIDENCE_MISS",
    "FAILURE_LABELS",
    "normalize_for_match",
    "normalize_number_text",
    "keyword_coverage",
    "keyword_hits",
    "normalize_source",
    "relevance_flags",
    "relevance_level",
    "source_recall_at_k",
    "source_hit_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "token_f1",
    "ratio",
    "first_relevant_rank",
    "classify_failure",
    "distribution",
    "attribution_clause_stats",
    "is_refusal_text",
    "effective_refusal",
    "refusal_path",
    "has_substantive_answer",
]

# --------------------------------------------------------------------------- #
# 失败分型标签
# --------------------------------------------------------------------------- #
OK = "ok"
ERROR = "error"
NOT_EVALUATED = "not_evaluated"            # 关键字段缺失，本条无法判定（不是通过）
WRONG_REFUSAL = "wrong_refusal"            # 有答案却拒答
MISSED_REFUSAL = "missed_refusal"          # 该拒答却作答
RETRIEVAL_MISS = "retrieval_miss"          # 期望来源一条都没召回到
RANKING_LOW = "ranking_low"                # 召回到了，但首个相关结果排名靠后
CITATION_OUT_OF_RANGE = "citation_out_of_range"
ATTRIBUTION_WEAK = "attribution_weak"      # 答案有引用，但句子找不到依据
ANSWER_MISMATCH = "answer_mismatch"        # 检索没问题，但答案内容对不上
EVIDENCE_MISS = "evidence_miss"            # 声明未找到且没有实质内容，但该有答案

FAILURE_LABELS: Dict[str, str] = {
    OK: "通过",
    ERROR: "执行失败",
    NOT_EVALUATED: "未评测",
    WRONG_REFUSAL: "过度拒答",
    MISSED_REFUSAL: "漏拒答",
    RETRIEVAL_MISS: "检索未召回",
    RANKING_LOW: "排序靠后",
    CITATION_OUT_OF_RANGE: "引用越界",
    ATTRIBUTION_WEAK: "归因不足",
    ANSWER_MISMATCH: "答案不符",
    EVIDENCE_MISS: "依据未找到",
}

# 判定阈值（集中在此，便于报告里说明口径）
RANKING_GOOD_RANK = 3          # 首个相关结果排在前 3 视为排序可用
ANSWER_KEYWORD_FLOOR = 0.5     # 关键词覆盖率低于此值视为答案不符
SUBSTANTIVE_FLOOR = 0.3        # 关键词覆盖率低于此值视为「没有实质内容」（纯拒答）

# 归因不足（疑似编造）的判定口径。
#
# 为什么用「unsupported 占比」而不是「归因总分」：
# 总分 = (supported + 0.5×weak) / total，`no_citation` 也算进分母 ——
# 于是「模型忘了标 [n]」会压低总分，被误判成「在编造」。
# 实测教训（2026-10-25）：把 2 条高质量的部分覆盖回答误杀成拒答，
# 过度拒答率从 7% 虚高到 21%。格式问题不是幻觉，两者必须分开。
#
# 也不再采用「只要有 1 句 unsupported 就判」：实测 Q013 因 12 句里
# 2 句没匹配上（得分 0.83）被判归因不足，且这一判断会随 LLM 措辞在
# 两轮之间跳动（同一份代码同一份数据，一轮判通过、一轮判归因不足）。
# 少数句子对不上更可能是词面匹配的局限，不是系统性编造。
ATTRIBUTION_UNSUPPORTED_RATIO = 0.4   # 无依据分句占比达到此值才判「归因不足」
ATTRIBUTION_MIN_CLAUSES = 3           # 分句太少（<3）时统计不稳定，不做判定

# --------------------------------------------------------------------------- #
# 文本规范化
# --------------------------------------------------------------------------- #
# 数字之间的逗号视为千分位（8,000 / 1,234,567）
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d)")
_SPACE_RE = re.compile(r"\s+")


def normalize_number_text(text: str) -> str:
    """把全角字符折成半角，并去掉数字之间的千分位逗号。

    为什么必须做这一步：评测集里写的是 `16,000`，模型回答里常写 `16000`；
    原来的子串匹配会把这种情况判成「没答到」，形成纯噪声的假阴性。
    这也是旧版 keyword_coverage 最主要的漏判来源。
    """
    if not text:
        return ""
    # NFKC 会把全角数字/字母/标点折成半角（'１６' -> '16'，'，' -> ','）
    folded = unicodedata.normalize("NFKC", text)
    return _THOUSANDS_RE.sub("", folded)


def normalize_for_match(text: str) -> str:
    """用于关键词/子串匹配的紧致形式：折半角 + 去空白 + 小写。

    去掉空白是为了让 `12 个月` 与 `12个月` 等价；
    关键词都是短语，去空白不会造成词语粘连的误判。
    """
    return _SPACE_RE.sub("", normalize_number_text(text)).lower()


def normalize_for_snippet(text: str) -> str:
    """片段匹配用的规范化：在 normalize_for_match 基础上再忽略表格竖线。

    为什么要单独一档：关键短语常直接取自表格行，例如
    `| 更换火花塞 | 每 64,000 km |`。文档的切分/清洗环节若去掉表格竖线
    （只留内容），逐字匹配就会失败，而这并不是内容真的不同。
    竖线只是排版结构，忽略它不会放宽到误判——所有内容字符仍逐个参与匹配。
    """
    return normalize_for_match(text).replace("|", "")


def keyword_hits(expected: Sequence[str], answer: str) -> Tuple[int, int]:
    """返回 (命中数, 期望总数)。空答案时命中数为 0。"""
    words = [str(word).strip() for word in (expected or []) if str(word).strip()]
    if not words:
        return 0, 0
    haystack = normalize_for_match(answer)
    hits = sum(1 for word in words if normalize_for_match(word) in haystack)
    return hits, len(words)


def keyword_coverage(expected: Sequence[str], answer: str) -> float:
    """期望关键词在回答中的覆盖率（规范化后子串匹配）。无期望词时返回 0.0。"""
    hits, total = keyword_hits(expected, answer)
    if total == 0:
        return 0.0
    return round(hits / total, 4)


# --------------------------------------------------------------------------- #
# 检索侧：文件级相关性
# --------------------------------------------------------------------------- #
def normalize_source(source: str) -> str:
    """统一来源写法，允许评测集只写文件名。"""
    return str(source or "").replace("\\", "/").strip()


def _matches_expected(source: str, expected_norm: Set[str]) -> bool:
    source_norm = normalize_source(source)
    if source_norm in expected_norm:
        return True
    # 允许 expected 只写 basename
    return any(source_norm.endswith("/" + want) for want in expected_norm)


def relevance_flags(
    expected_sources: Sequence[str],
    retrieved: Sequence[Any],
    expected_snippets: Optional[Sequence[str]] = None,
) -> List[int]:
    """按召回顺序给出二值相关性标签（1=相关，0=不相关）。

    两档判定粒度，由 `expected_snippets` 决定：

    - **给了关键短语（片段级）**：相关 = 来源文件命中 **且** 片段正文含该短语。
      这是更严格、也更有意义的口径 —— 它回答的是「讲这件事的那一段有没有被排上来」，
      而不只是「这份文档在不在结果里」。
    - **没给（文件级）**：相关 = 来源文件命中。同文件名下的任意片段都算相关，
      粒度粗，在 k>=5 时几乎必然满分，区分不出排序好坏。

    为什么要和来源绑定：同一个短语可能在多份文档里都出现（例如 P0420 的说明在
    保养速查表和 OBD 码表里各有一份），只按短语匹配会跨文档误判。
    """
    expected_norm = {
        normalize_source(item) for item in (expected_sources or []) if normalize_source(item)
    }
    if not expected_norm:
        return [0] * len(retrieved)

    snippet_norms = [
        normalize_for_snippet(item) for item in (expected_snippets or []) if str(item).strip()
    ]
    flags: List[int] = []
    for entry in retrieved:
        source = _entry_field(entry, "source")
        if not _matches_expected(source, expected_norm):
            flags.append(0)
            continue
        if snippet_norms:
            text = normalize_for_snippet(_entry_field(entry, "text"))
            flags.append(1 if any(snip in text for snip in snippet_norms) else 0)
        else:
            flags.append(1)
    return flags


def relevance_level(expected_snippets: Optional[Sequence[str]]) -> str:
    """报告里标明本次用的是哪一档口径，避免把两种粒度的数字混着比。"""
    return "snippet" if any(str(item).strip() for item in (expected_snippets or [])) else "file"


def _entry_field(entry: Any, name: str) -> str:
    """兼容 pydantic 模型、dict、以及带属性的对象。"""
    if isinstance(entry, dict):
        return str(entry.get(name, "") or "")
    return str(getattr(entry, name, "") or "")


def source_recall_at_k(
    expected_sources: Sequence[str],
    retrieved: Sequence[Any],
    k: int,
    expected_snippets: Optional[Sequence[str]] = None,
) -> Optional[float]:
    """前 k 条召回覆盖了多少比例的期望来源。无期望来源时返回 None。

    给了 `expected_snippets` 时，「覆盖」的判定变严：
    必须有一个召回结果**既来自该来源、又含关键短语**，才算这个来源被覆盖。
    否则只是「文件出现过」，不代表能找到答案所在的那一段。
    """
    expected_norm = {
        normalize_source(item) for item in (expected_sources or []) if normalize_source(item)
    }
    if not expected_norm:
        return None
    if k <= 0:
        return 0.0

    snippet_norms = [
        normalize_for_snippet(item) for item in (expected_snippets or []) if str(item).strip()
    ]
    hit: Set[str] = set()
    for entry in list(retrieved)[:k]:
        source = normalize_source(_entry_field(entry, "source"))
        if snippet_norms:
            text = normalize_for_snippet(_entry_field(entry, "text"))
            if not any(snip in text for snip in snippet_norms):
                continue
        for want in expected_norm:
            if source == want or source.endswith("/" + want):
                hit.add(want)
    return round(len(hit) / len(expected_norm), 4)


def source_hit_at_k(
    expected_sources: Sequence[str],
    retrieved: Sequence[Any],
    k: int,
    expected_snippets: Optional[Sequence[str]] = None,
) -> bool:
    """前 k 条里是否至少覆盖了一个期望来源（口径同 source_recall_at_k）。"""
    recall = source_recall_at_k(expected_sources, retrieved, k, expected_snippets)
    return bool(recall)


def precision_at_k(flags: Sequence[int], k: int) -> Optional[float]:
    """前 k 条里相关结果的比例。分母取实际返回条数（不足 k 时用实际数）。"""
    if k <= 0:
        return None
    window = list(flags)[:k]
    if not window:
        return None
    return round(sum(window) / len(window), 4)


def first_relevant_rank(flags: Sequence[int]) -> Optional[int]:
    """首个相关结果的 1-based 排名；没有则返回 None。"""
    for index, flag in enumerate(flags, start=1):
        if flag:
            return index
    return None


def reciprocal_rank(flags: Sequence[int]) -> float:
    """MRR 的单条贡献：首个相关结果排名的倒数，没有相关结果时为 0。"""
    rank = first_relevant_rank(flags)
    return round(1.0 / rank, 4) if rank else 0.0


def ndcg_at_k(flags: Sequence[int], k: int) -> Optional[float]:
    """二值相关性下的 nDCG@k。

    IDCG 用「本次召回到的相关项全部排到最前」作为理想排序（见模块文档第 2 条）。
    因此该值衡量的是**召回集内部**的排序质量，不反映漏召。
    """
    window = list(flags)[:k]
    if not window:
        return None
    relevant_total = sum(1 for flag in flags if flag)
    if relevant_total == 0:
        return 0.0

    def gain(position: int) -> float:
        # position 为 1-based，折扣用 log2(position + 1)
        return 1.0 / math.log2(position + 1)

    dcg = sum(gain(i) for i, flag in enumerate(window, start=1) if flag)
    ideal_len = min(relevant_total, k)
    idcg = sum(gain(i) for i in range(1, ideal_len + 1))
    if idcg <= 0:
        return 0.0
    return round(min(dcg / idcg, 1.0), 4)


# --------------------------------------------------------------------------- #
# 生成侧：词面一致性（不是语义正确性）
# --------------------------------------------------------------------------- #
def token_f1(reference: str, hypothesis: str) -> Optional[float]:
    """参考答案与系统答案的 token 级 F1（多重集）。

    ⚠️ 这是**词面重合度**，对同义改写不友好：
    「每 8,000 km 更换」与「每八千公里换一次」会得 0 分，但语义一致。
    所以它只能作为「答案明显跑偏」的下限检测，不能当正确率。
    真正衡量正确性需要人工或 LLM judge，本项目暂不引入，报告里明确标注。
    """
    ref_tokens = tokenize(normalize_number_text(reference or ""))
    hyp_tokens = tokenize(normalize_number_text(hypothesis or ""))
    if not ref_tokens or not hyp_tokens:
        return None
    ref_counter = Counter(ref_tokens)
    hyp_counter = Counter(hyp_tokens)
    overlap = sum((ref_counter & hyp_counter).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


# --------------------------------------------------------------------------- #
# 汇总工具
# --------------------------------------------------------------------------- #
def ratio(numerator: int, denominator: int) -> Optional[float]:
    """安全比例：分母为 0 时返回 None。

    为什么不返回 0.0：拒答指标的分母是「负样本条数」，
    如果评测集里一条负样本都没有，0.0 会被读成「拒答全错」，
    而事实是**没有可评的样本**。None 在报告里显示为 N/A，不会误导。
    """
    if not denominator:
        return None
    return round(numerator / denominator, 4)


# --------------------------------------------------------------------------- #
# 拒答判定
# --------------------------------------------------------------------------- #
# 本项目有两条拒答路径，指标必须区分，否则会得出互相矛盾的数字：
#   hard —— 检索为空，`refuse_when_empty` 直接返回固定话术，refused=True
#   soft —— 检索非空但依据不足，走正常生成，模型自己遵守提示词说出「未找到」，refused=False
# 实测（2026-10-25 基线）：6 条负样本全部是 soft，hard 一次都没触发。
# 这意味着拒答能力实际依赖模型遵守提示词，换模型需要重新验证。
REFUSAL_MARKERS: Tuple[str, ...] = (
    "当前知识库中未找到",
    "未找到相关依据",
    "未找到与该问题相关的依据",
    "资料中未涉及",
    "资料中没有",
)


def is_refusal_text(answer: str) -> bool:
    """回答文本是否属于拒答话术。

    刻意只用固定句式，不用「无法」「未找到」这类宽泛词：
    正常回答里也可能出现「若未找到故障码，请…」，宽泛匹配会把有效回答误判成拒答。
    """
    haystack = normalize_for_match(answer)
    if not haystack:
        return False
    return any(normalize_for_match(marker) in haystack for marker in REFUSAL_MARKERS)


def has_substantive_answer(expected_keywords: Sequence[str], answer: str) -> bool:
    """答案里是否有可用信息，而不是只声明「未找到」。

    这是区分「部分覆盖」与「纯拒答」的关键。实测中这两类回答的**开头一模一样**：
        「当前知识库中未找到关于首保的依据…」+ 后续给出常规保养周期  -> 实质信息，是有效回答
        「当前知识库中未找到关于刹车片磨损限值的依据。」+ 建议确认渠道 -> 纯拒答
    只按文本特征会把前者误判成拒答（实测误判 6/14 条），必须结合「有没有给出信息」。

    判据用期望关键词覆盖率，而不是答案长度：长度会被「建议确认渠道」那几行撑起来。
    """
    if not expected_keywords:
        return False
    return keyword_coverage(expected_keywords, answer) >= SUBSTANTIVE_FLOOR


def effective_refusal(
    must_refuse: bool,
    refused_raw: bool,
    answer: str,
    substantive: bool = False,
) -> bool:
    """按样本极性判定「这条回答算不算没有作答」。

    为什么正负样本要用两套规则：同一个「声明资料缺口」的动作，在两类样本里
    含义完全相反。

    - 负样本（must_refuse=True）：目标是**声明证据不足**。只要说了「未找到依据」，
      哪怕后面补了「现有资料里只有 X」的相关证据，义务就已经履行 ——
      这种「部分披露」是最理想的负样本回答，不该因为带了实质信息就判成漏拒答。
    - 正样本（must_refuse=False）：目标是**给出答案**。只有「纯拒答」（声明缺口且
      没有任何实质信息）才算没答；「声明缺口 + 给出可用信息」是高质量回答，
      绝不能算过度拒答。

    实测教训（2026-10-25）：Q011「长期停放蓄电池会亏电吗」的正确行为是
    「声明未找到 + 披露保养表里的蓄电池检查周期」。用一套规则判定时，
    若按正样本规则，它会被记成「未拒答」；若按负样本规则，Q001 那种
    「声明缺口 + 给出真实周期」的有效回答又会被记成「过度拒答」。
    两套规则的差别就是这批数据的分歧点。
    """
    declared = bool(refused_raw) or is_refusal_text(answer)
    if not declared:
        return False
    if must_refuse:
        return True
    # 正样本：硬拒答一定算；软拒答要「没有实质信息」才算
    return bool(refused_raw) or not substantive


def refusal_path(refused: bool, answer: str, substantive: bool = False) -> str:
    """返回拒答路径：hard / soft / partial / none。

    partial 表示「先声明资料缺口、再给出可用信息」，这类回答**不是拒答**，
    单独标出来是为了让报告能区分「模型不肯答」和「模型诚实地说了边界」。
    """
    if refused:
        return "hard"
    if is_refusal_text(answer):
        return "partial" if substantive else "soft"
    return "none"


# --------------------------------------------------------------------------- #
# 失败分型
# --------------------------------------------------------------------------- #
def classify_failure(
    *,
    error: Optional[str] = None,
    must_refuse: bool = False,
    refused: Optional[bool] = None,
    substantive: bool = False,
    expected_sources: Sequence[str] = (),
    recall_at_k: Optional[float] = None,
    first_rank: Optional[int] = None,
    out_of_range: Sequence[int] = (),
    attribution_total: int = 0,
    attribution_unsupported: int = 0,
    keyword_cov: float = 1.0,
    f1: Optional[float] = None,
) -> str:
    """给一条评测结果打**一个**主标签。

    为什么只给一个主标签：多标签会让统计表失去可加性（一条算两次），
    而分型的目的恰恰是回答「主要卡在哪一环」。次要线索仍保留在逐条明细里。

    判定顺序（严格从上到下，先命中先返回）：
      1. 执行异常
      2. 该拒答的样本拿不到拒答结果（如只测召回的模式）-> 未评测，不能算通过
      3. 引用越界（格式错误，与是否拒答无关，先报）
      4. 负样本正确拒答 -> 通过（短路，不再比对词面）
      5. 拒答行为错误（这类问题最严重，先于检索判定，避免被后续条件掩盖）
      6. 检索未召回
      7. 排序靠后
      8. 归因不足（无依据分句占比 ≥ 40%，且至少 3 句）
      9. 答案不符
     10. 通过

    第 5 步必须在第 6 步之前：一条负样本如果该拒却答了，
    即使「召回」看起来正常，根因也是拒答策略而不是检索。

    ⚠️ `refused` 参数要传**按极性判定的有效拒答**，一律用
    `effective_refusal(must_refuse, refused_raw, answer, substantive)` 计算，
    不要在这里自己拼条件。历史上这里有过两次教训：

    1. 直接传 API 的 refused 字段会漏掉 soft 路径（模型自己说「未找到」）——
       实测 6 条负样本全是 soft，只传原始字段会全判成「漏拒答」，
       而拒答成功率却显示 100%，两个数字自相矛盾。
    2. 补成 `refused_raw or (有拒答话术 and not substantive)` 之后，
       负样本里「声明缺口 + 披露相关证据」这种**最理想**的回答会被判成漏拒答。
       因为同一个动作在两类样本里含义相反，必须按极性分开判（见 effective_refusal）。
    """
    if error:
        return ERROR

    # 负样本没有任何 expected_sources / expected_keywords，
    # 如果连 refused 都拿不到，后面所有判定都会落空并误判为「通过」。
    if must_refuse and refused is None:
        return NOT_EVALUATED

    # 引用越界是格式错误，与拒答与否无关，先报
    if out_of_range:
        return CITATION_OUT_OF_RANGE

    # 负样本正确拒答 = 通过，必须在此短路返回。
    # 否则会继续走后面的关键词 / F1 判定：负样本的 ground_truth 本身就写着
    # 「知识库中无该信息，应拒答」，而拒答文本是「未找到相关依据」，
    # 两者的词面 F1 很低，会被误判成 ANSWER_MISMATCH。
    # 实测中这会把 6 条负样本里的正确拒答全部错标（占评测集 30%）。
    if must_refuse and refused is True:
        return OK

    # 拒答行为判定
    if must_refuse and refused is False:
        return MISSED_REFUSAL
    if not must_refuse and refused is True:
        # 该答却拒。分两种根因，不能混为「过度拒答」：
        #   有实质内容却拒 -> 真的是策略问题
        #   没有实质内容     -> 是「证据没到位」，根因在检索/引用，不在拒答策略
        if substantive:
            return WRONG_REFUSAL
        if expected_sources:
            return EVIDENCE_MISS
        return WRONG_REFUSAL

    if expected_sources:
        if recall_at_k is not None and recall_at_k <= 0:
            return RETRIEVAL_MISS
        if first_rank is not None and first_rank > RANKING_GOOD_RANK:
            return RANKING_LOW

    # 归因不足：只看「无依据分句占比」，不看归因总分。
    # 总分里混着 no_citation（模型忘了标 [n]），那是格式问题不是编造；
    # 用总分当门槛会把「忘了标引用」判成「在编造」—— 这个坑踩过一次，见常量注释。
    # 分句少于 3 句时统计不稳定，不做判定。
    if attribution_total >= ATTRIBUTION_MIN_CLAUSES:
        unsupported_ratio = attribution_unsupported / attribution_total
        if unsupported_ratio >= ATTRIBUTION_UNSUPPORTED_RATIO:
            return ATTRIBUTION_WEAK

    if expected_sources and keyword_cov < ANSWER_KEYWORD_FLOOR:
        return ANSWER_MISMATCH

    # 为什么不拿 token F1 当判定依据：
    # 回答里只要附加了「建议确认渠道 / 其他相关资料」这类必要说明，F1 就会被显著稀释。
    # 实测 Q011 关键词覆盖 1.0（关键信息全到），F1 只有 0.16，用 F1 会把它判成「答案不符」。
    # F1 仍保留在报告里作为人工复核参考，但不参与分型。
    return OK


def attribution_clause_stats(attributions: Iterable[Any]) -> Dict[str, Any]:
    """汇总分句级归因：校验了多少句、其中多少无依据、多少只是没标引用。

    为什么单独统计：`faithfulness` 是「每条的分数再平均」，会被条目数稀释；
    这里看的是**全部句子合起来**的构成，回答的是
    「这个系统给出的断言里有多大比例找不到依据」。

    口径说明：
    - `unsupported_ratio` 才是分型（归因不足）实际使用的信号；
    - `no_citation_ratio` 是格式问题（模型忘了标 [n]），不参与分型；
    - `meta_skipped` 是「资料里没有 X」这类缺口陈述，无法用词面覆盖验证，两边都不计。
    """
    total = supported = weak = unsupported = no_citation = meta_skipped = 0
    for attribution in attributions or []:
        if not isinstance(attribution, dict):
            continue
        if not attribution.get("total"):
            continue
        total += int(attribution.get("total") or 0)
        supported += int(attribution.get("supported") or 0)
        weak += int(attribution.get("weak") or 0)
        unsupported += int(attribution.get("unsupported") or 0)
        no_citation += int(attribution.get("no_citation") or 0)
        meta_skipped += int(attribution.get("meta_skipped") or 0)
    return {
        "checked_clauses": total,
        "supported": supported,
        "weak": weak,
        "unsupported": unsupported,
        "no_citation": no_citation,
        "meta_skipped": meta_skipped,
        "unsupported_ratio": round(unsupported / total, 4) if total else None,
        "no_citation_ratio": round(no_citation / total, 4) if total else None,
    }


def distribution(labels: Iterable[str]) -> Dict[str, int]:
    """标签分布统计，键为中文可读名。"""
    counts: Dict[str, int] = {}
    for label in labels:
        name = FAILURE_LABELS.get(label, label)
        counts[name] = counts.get(name, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
