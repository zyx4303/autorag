"""问答链路：检索 -> 组装带编号的资料 -> 调用 LLM -> 解析引用 -> 返回结构化结果。

核心约束（也是简历上值得讲的点）：
1. 只依据检索到的资料作答，资料里没有就明说，并给出"建议确认渠道"；
2. 每处结论都必须带 [n] 角标，n 与返回的 citations 一一对应，禁止编造引用编号；
3. 检索为空（或向量相似度过低）时不调用大模型，直接返回兜底话术，
   避免"无依据生成"这一最常见的幻觉来源。
"""
from __future__ import annotations

import copy
import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Set, Tuple

from app.config import Settings
from app.core.attribution import AttributionChecker, build_checker
from app.core.llm_client import LLMClient, LLMError, LLMResult
from app.core.retriever import HybridRetriever
from app.core.text_utils import make_snippet
from app.logging_conf import get_logger
from app.models import Citation, Latency, RetrievedChunk, TokenUsage

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是「汽车售后知识库助手」，服务对象是车主与授权服务站的售后人员。

回答规则（必须严格遵守）：
1. 只能依据「参考资料」中的内容作答，禁止使用资料之外的常识、经验或猜测补充事实。
2. 每一条结论后面都要标注来源编号，格式为 [1]、[2]；一句话涉及多条资料时写 [1][3]。
   编号只能取自参考资料里已存在的编号，严禁编造不存在的编号。
3. 如果资料不足以回答，必须明确说明"当前知识库中未找到相关依据"，并列出建议确认渠道
   （如：以随车《用户手册》为准 / 联系授权服务站 / 拨打厂家客服热线），不要强行作答。
4. 如果资料中的数值带版本、车型、年款限定，回答时必须原样保留这些限定条件，不要扩大适用范围。
5. 不要输出与问题无关的寒暄、免责声明堆砌或多余客套。

输出格式：
- 先用一句话给出直接结论；
- 需要步骤或参数时用有序列表，每步/每项末尾标注来源编号；
- 最后单独一段以「依据：」开头，逐条列出用到的来源（格式：[n] 文件路径 · 章节）。"""

USER_PROMPT_TEMPLATE = """## 参考资料
{context}

## 用户问题
{question}

请依据上述参考资料回答，并按要求标注来源编号。"""

# 查询改写：把「那这个车呢」这类追问变成独立问题。
# 只做指代消解与省略补全，不引入新信息 —— 这是它能被校验的前提。
QUERY_REWRITE_PROMPT = """你把用户的追问改写成一条**独立、自包含**的检索问题。

规则：
1. 结合对话历史消除代词（这个 / 那个 / 它 / 同上）和被省略的主语；
2. 只做改写，不回答；不得添加对话历史里没有的新信息；
3. 历史中标注「（上轮未回答）」的问题是被拒答的——用户问它只是随手一试，
   当前追问（如「那正常保养呢？」）多半是在**换话题**。
   除非追问明确在继续那个话题，否则不要把被拒答的话题拼进改写结果；
4. 输出只写改写后的问题本身，一行，不要引号、不要任何解释；
5. 如果追问本身已经独立完整，原样输出。"""

# 历史块拼进用户消息的位置在「参考资料」之后、「用户问题」之前；
# 空历史时完全不带这个块，保证单轮请求的提示词与接入多轮之前**逐字节相同**。
HISTORY_BLOCK_TITLE = "## 对话历史（仅供理解当前问题；回答仍必须只依据参考资料）"

REFUSAL_ANSWER = (
    "当前知识库中未找到与该问题相关的依据，因此不做推测性回答。\n\n"
    "建议确认渠道：\n"
    "1. 以随车《用户手册》或《保养手册》的对应章节为准；\n"
    "2. 联系授权服务站，提供 VIN 码由售后顾问核对；\n"
    "3. 需要我回答的话，请把相关手册/工单文档放入 data/documents 目录后调用 "
    "POST /api/v1/ingest 重新入库。"
)

CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")


def refusal_on_hallucination(
    attribution: Optional[Dict[str, Any]], ratio: float
) -> Optional[str]:
    """判断一次生成是否应因「疑似编造」改为拒答，返回拒答原因（None 表示不拒答）。

    只看 `unsupported`（在被引用片段里找不到依据的句子数），**不看归因总分**。
    两条硬条件，缺一不可：
      - 至少 2 句找不到依据 —— 单句容易被「不能据此判定」这类元话语误报；
      - 占比达到 `ratio` —— 少数句子无依据仍可能是有效回答。

    刻意不用 `no_citation`：模型忘了标 `[n]` 时归因会判 no_citation、把总分拉到 0，
    但那种答案本身是有效的（实测误杀过 2 条高质量的部分覆盖回答）。
    no_citation 是格式问题，不是幻觉。

    单独抽成函数是为了能单测 —— 真实评测集里没有「编造」样本，
    这层逻辑靠跑评测是验证不出来的。
    """
    if ratio <= 0 or not isinstance(attribution, dict):
        return None
    checked = int(attribution.get("total") or 0)
    if checked < 2:
        return None
    unsupported = int(attribution.get("unsupported") or 0)
    if unsupported < 2:
        return None
    share = unsupported / checked
    if share < ratio:
        return None
    return (
        f"{unsupported}/{checked} 句在被引用的片段中找不到依据"
        f"（阈值 {ratio:.2f}），判定为编造"
    )


class QAEngine:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        llm_client: LLMClient,
        session_store: Optional[Any] = None,
        query_cache: Optional[Any] = None,
        index_state: Optional[Any] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._llm = llm_client
        # 会话历史（可选）：None 表示多轮能力未装配，answer() 会自动降级为单轮
        self._session_store = session_store
        # 查询缓存（可选）+ 索引版本（键的失效依据），见 query_cache/index_state 模块
        self._query_cache = query_cache
        self._index_state = index_state
        # 请求指标采集（可选）：记录延迟/ token/缓存命中，供 /stats 聚合
        self._metrics = metrics
        # 逐句归因校验（可选）：只报告不拦截，失败不影响主流程
        self._attribution: Optional[AttributionChecker] = build_checker(settings)

    # ------------------------------------------------------------------ #
    # 上下文组装
    # ------------------------------------------------------------------ #
    @staticmethod
    def build_context(results: Sequence[RetrievedChunk]) -> Tuple[str, List[Citation]]:
        """把召回片段拼成带编号的参考资料，并生成引用元数据表。"""
        blocks: List[str] = []
        citations: List[Citation] = []
        for index, item in enumerate(results, start=1):
            location = f"{item.source}"
            if item.section:
                location += f" · {item.section}"
            blocks.append(
                f"[{index}] 来源：{location}（片段序号 {item.position}，相关度 {item.score:.4f}）\n"
                f"{item.text}"
            )
            citations.append(
                Citation(
                    index=index,
                    chunk_id=item.chunk_id,
                    source=item.source,
                    title=item.title,
                    section=item.section,
                    position=item.position,
                    score=round(item.score, 6),
                    snippet=make_snippet(item.text, 200),
                )
            )
        return "\n\n".join(blocks), citations

    def build_messages(
        self,
        question: str,
        context: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[Dict[str, str]]:
        user_content = USER_PROMPT_TEMPLATE.format(context=context, question=question)
        if history:
            # 历史块只负责让模型理解「这个问题在问什么」，
            # 回答本身的依据仍然只能是参考资料（标题里写死了）。
            block = self.build_history_block(history)
            marker = "## 用户问题"
            user_content = user_content.replace(
                marker, f"{HISTORY_BLOCK_TITLE}\n\n{block}\n\n{marker}", 1
            )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def resolve_attribution_sources(
        attribution: Dict[str, Any],
        citations: Sequence[Citation],
    ) -> Dict[str, Any]:
        """把归因句子里的引用编号解析成来源明细（句↔片段对齐）。

        归因核心（attribution 模块）只给编号（[1][2]），前端要做
        「这句话对应哪段资料」的高亮，还得自己拿编号去 citations 里
        交叉对一遍 —— 对齐这步在这里做掉，前端拿到 `sources` 直接渲染，
        不需要理解编号体系。

        解析对象是传给归因 checker 的同一份 citations（编号体系一致）；
        编号越界理论上已被 verify_citations 拦截，这里仍防御性标注 invalid。
        """
        by_index = {c.index: c for c in citations or []}
        resolved_sentences: List[Dict[str, Any]] = []
        for sent in attribution.get("sentences") or []:
            sources: List[Dict[str, Any]] = []
            for idx in sent.get("citations") or []:
                c = by_index.get(idx)
                if c is None:
                    sources.append(
                        {"index": idx, "source": "", "section": "", "position": 0, "invalid": True}
                    )
                    continue
                sources.append(
                    {
                        "index": idx,
                        "source": c.source,
                        "section": c.section,
                        "position": c.position,
                    }
                )
            item = dict(sent)
            item["sources"] = sources
            resolved_sentences.append(item)
        out = dict(attribution)
        out["sentences"] = resolved_sentences
        return out

    @staticmethod
    def build_history_block(history: List[Dict[str, str]]) -> str:
        """把最近几轮问答压成短块。答案截到 200 字 —— 历史只需要「聊过什么」，
        不需要完整回答，截断同时控制 token 与干扰。"""
        lines: List[str] = []
        for turn in history or []:
            q = (turn.get("question") or "").strip()
            a = (turn.get("answer") or "").strip()
            if not q:
                continue
            # 拒答轮显式标注：让生成模型知道那轮没有实质回答，
            # 别把它的「答案」当成可用上下文。
            marker = "（上轮未回答）" if turn.get("refused") else ""
            lines.append(f"问：{q}{marker}")
            if a:
                lines.append(f"答：{a[:200]}{'…' if len(a) > 200 else ''}")
        return "\n".join(lines)

    @staticmethod
    def build_rewrite_messages(
        question: str, history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """查询改写的消息体。只用最近 3 轮：更早的轮次对当前追问影响很小，
        却会增加改写跑偏（被旧话题带走）的概率，还多花 token。"""
        recent = (history or [])[-3:]
        lines = ["【对话历史】"]
        for turn in recent:
            q = (turn.get("question") or "").strip()
            a = (turn.get("answer") or "").strip()
            if q:
                # 拒答轮标注出来，配合 system 提示词第 3 条：
                # 被拒答的话题多半是用户随手一试，追问往往在换话题，
                # 不该被拼进改写结果（实测「那正常保养呢？」曾被拖进
                # 上一轮的量子纠缠问题里，连累正常的追问也被拒答）。
                marker = "（上轮未回答）" if turn.get("refused") else ""
                lines.append(f"问：{q}{marker}")
            if a:
                lines.append(f"答：{a[:120]}{'…' if len(a) > 120 else ''}")
        lines.append(f"\n【当前追问】{question.strip()}")
        lines.append("\n请输出改写后的独立问题：")
        return [
            {"role": "system", "content": QUERY_REWRITE_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ]

    # ------------------------------------------------------------------ #
    # 兜底判断
    # ------------------------------------------------------------------ #
    def should_refuse(
        self, results: Sequence[RetrievedChunk], mode: str
    ) -> Tuple[bool, Optional[str]]:
        """判断是否应当直接拒答（不调用大模型）。

        :param mode: **本次请求实际生效**的检索模式，而不是 .env 里的默认值。
            因为接口允许按请求覆盖 retrieval_mode，若这里读配置默认值会出现两种错：
            - 配置是 vector、请求用 keyword/hybrid：keyword 模式下没有 vector_score，
              阈值判断会把有效召回误判成 0 分而直接拒答；
            - 配置是 keyword/hybrid、请求用 vector：阈值被完全跳过。
        """
        if not results:
            return True, "检索结果为空"

        # 第一层兜底：证据下限。
        # hybrid 的 `score` 是融合分（0.0x 量级），与 vector 模式的余弦（0~1）不同量纲，
        # 所以这里用独立配置项，而不是复用 MIN_SCORE。
        # 实测（2026-10-25）：负样本最高召回分 0.0028~0.0345，可答样本 0.0151~0.9716；
        # 取 0.01 可拦下 4/6 条最离谱的负样本且零误杀（可答样本最低 0.0151）。
        evidence_floor = float(self._settings.refusal_evidence_floor or 0.0)
        if evidence_floor > 0:
            best_fused = max((item.score or 0.0) for item in results)
            if best_fused < evidence_floor:
                return True, f"最高召回分 {best_fused:.4f} 低于证据下限 {evidence_floor}"

        # 只有"纯向量模式"下 score 才是可解释的余弦相似度，其余模式不做阈值判断
        if mode == "vector":
            best = max((item.vector_score or 0.0) for item in results)
            if best < self._settings.min_score:
                return True, f"最高向量相似度 {best:.4f} 低于阈值 {self._settings.min_score}"
        return False, None

    # ------------------------------------------------------------------ #
    # 引用校验
    # ------------------------------------------------------------------ #
    @staticmethod
    def verify_citations(
        answer: str, citations: Sequence[Citation]
    ) -> Tuple[List[Citation], List[str]]:
        """校验回答里的 [n] 是否越界；返回 (实际被引用的引用列表, 警告列表)。"""
        warnings: List[str] = []
        used_indices: Set[int] = set()
        for match in CITATION_PATTERN.finditer(answer):
            used_indices.add(int(match.group(1)))

        valid_indices = {citation.index for citation in citations}
        unknown = sorted(index for index in used_indices if index not in valid_indices)
        if unknown:
            warnings.append(
                "回答中出现了超出资料范围的引用编号 "
                + "、".join(f"[{index}]" for index in unknown)
                + "，这些编号不对应任何召回片段，请人工核对。"
            )

        if citations and not used_indices:
            warnings.append("回答未包含任何 [n] 引用角标，可能是模型未遵循引用格式，建议人工核对。")

        used = [citation for citation in citations if citation.index in used_indices]
        if not used and citations:
            # 模型没标引用时，把召回来源一并返回，便于用户自行定位
            used = list(citations)
        return used, warnings

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    async def retrieve_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> Tuple[List[RetrievedChunk], str, int]:
        started = time.perf_counter()
        results, effective_mode = await self._retriever.retrieve(query, top_k=top_k, mode=mode)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return results, effective_mode, elapsed_ms

    # ------------------------------------------------------------------ #
    # 多轮会话：历史 → 查询改写 → 落库
    # ------------------------------------------------------------------ #
    def _history(self, session_id: Optional[str]) -> List[Dict[str, str]]:
        if not session_id or self._session_store is None:
            return []
        return self._session_store.history(session_id)

    def _remember(
        self,
        session_id: Optional[str],
        question: str,
        answer_text: str,
        refused: bool = False,
    ) -> None:
        """把本轮问答落进历史。失败只记日志 —— 会话是增益功能，不能拖垮问答。"""
        if not session_id or self._session_store is None:
            return
        try:
            self._session_store.append(session_id, question, answer_text, refused=refused)
        except Exception as exc:  # noqa: BLE001
            logger.warning("会话历史写入失败（不影响本次回答）：%s", exc)

    async def _rewrite_query(
        self,
        question: str,
        history: List[Dict[str, str]],
        warnings: List[str],
    ) -> Tuple[str, bool]:
        """把追问改写成独立问题；任何失败都回退为原问题。

        返回 (检索用的查询, 是否真的改写了)。
        """
        if not history or not self._settings.query_rewrite_enabled:
            return question, False
        messages = self.build_rewrite_messages(question, history)
        try:
            result: LLMResult = await self._llm.chat(messages)
        except LLMError as exc:
            warnings.append(f"查询改写失败，已按原问题检索：{exc}")
            return question, False
        except Exception as exc:  # noqa: BLE001
            # 这里必须是宽捕获：改写只是检索前的增益步骤，
            # 任何异常都应该回退为原问题继续回答，而不是把 /chat 一起带崩。
            warnings.append(f"查询改写异常，已按原问题检索：{type(exc).__name__}: {exc}")
            return question, False

        rewritten = (result.text or "").strip()
        # 剥掉模型可能顺手加的引号，只取第一行 —— 提示词要求单行输出
        if rewritten:
            rewritten = rewritten.splitlines()[0].strip().strip('"“”‘’')
        # 合理性护栏：太短（改写丢了内容）或太长（把答案当问题写回来）都不可信
        if not rewritten or len(rewritten) < 4 or len(rewritten) > 200:
            warnings.append("查询改写结果异常，已按原问题检索")
            return question, False
        if rewritten == question:
            # 模型原样返回了追问 —— 没有提供新信息，不该在响应里伪装成「已改写」
            return question, False
        return rewritten, True

    def _record_chat(self, outcome: Dict[str, Any]) -> None:
        """把一次 /chat 的结果记进指标采集。失败静默——观测不能影响主流程。"""
        if self._metrics is None:
            return
        try:
            latency = outcome.get("latency")
            usage = outcome.get("usage")
            self._metrics.record(
                endpoint="chat",
                total_ms=getattr(latency, "total_ms", None),
                retrieve_ms=getattr(latency, "retrieve_ms", None),
                generate_ms=getattr(latency, "generate_ms", None),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
                cache_hit=bool(outcome.get("cache_hit")),
                refused=bool(outcome.get("refused")),
            )
        except Exception:  # noqa: BLE001
            logger.warning("指标记录失败（已忽略）", exc_info=True)

    # ------------------------------------------------------------------ #
    # 查询缓存（L5）：键里带索引版本，入库即失效；多轮与 debug 不缓存。
    # 正确性边界详见 query_cache / index_state 模块文档。
    # ------------------------------------------------------------------ #
    def _cache_key(
        self,
        question: str,
        top_k: Optional[int],
        mode: Optional[str],
    ) -> Optional[str]:
        """缓存键；返回 None 表示该请求不该进缓存。"""
        if self._query_cache is None or not self._settings.query_cache_enabled:
            return None
        version = self._index_state.version if self._index_state is not None else 0
        effective_k = int(top_k or self._settings.top_k)
        return f"v{version}|{mode or 'auto'}|k{effective_k}|{(question or '').strip()}"

    @staticmethod
    def _cacheable(session_id: Optional[str], debug: bool) -> bool:
        # 多轮答案依赖会话历史；debug 是评测/调试路径——
        # 评测的 --repeat 要测的是真实方差，第二轮全命中缓存的话方差就是假的。
        return not session_id and not debug

    async def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        debug: bool = False,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """完整问答，返回 dict（与 ChatResponse 字段对齐）。

        - debug=True 时在返回值的 retrieved 字段里带上召回明细；
        - 返回值额外带一个内部字段 retrieved_raw（RetrievedChunk 列表），
          便于评测脚本复用同一次检索结果，避免重复调用向量库。该字段不会进 HTTP 响应。
        """
        total_started = time.perf_counter()
        question = (question or "").strip()
        warnings: List[str] = []

        # 查询缓存：命中直接返回，不进检索和生成。
        # 延迟字段重算（旧值是上次请求的，透传会误导调用方）。
        cache_key: Optional[str] = None
        if self._cacheable(session_id, debug):
            cache_key = self._cache_key(question, top_k, mode)
            if cache_key:
                cached = self._query_cache.get(cache_key)
                if cached is not None:
                    logger.info("查询缓存命中：%s", question[:40])
                    response = copy.deepcopy(cached)
                    response["cache_hit"] = True
                    response["latency"] = Latency(
                        retrieve_ms=0,
                        generate_ms=0,
                        total_ms=int((time.perf_counter() - total_started) * 1000),
                    )
                    self._record_chat(response)
                    return response

        # 多轮：先用历史把追问改写成独立问题，再拿改写结果去检索。
        # 生成时仍用用户的原话 + 历史块 —— 模型需要看到用户怎么问的。
        history = self._history(session_id)
        retrieval_query, did_rewrite = await self._rewrite_query(question, history, warnings)

        results, effective_mode, retrieve_ms = await self.retrieve_only(
            retrieval_query, top_k=top_k, mode=mode
        )
        debug_rows = HybridRetriever.to_debug(results, include_text=True) if debug else []

        refuse, reason = self.should_refuse(results, effective_mode)
        if refuse and self._settings.refuse_when_empty:
            logger.info("触发兜底回答：%s", reason)
            total_ms = int((time.perf_counter() - total_started) * 1000)
            # 拒答也进历史：下一轮改写时模型需要知道「上一轮没答上」
            self._remember(session_id, question, REFUSAL_ANSWER, refused=True)
            refusal_response = {
                "question": question,
                "answer": REFUSAL_ANSWER,
                "citations": [],
                "retrieved": debug_rows,
                "retrieved_raw": results,
                "refused": True,
                "refusal_reason": reason,
                "retrieval_mode": effective_mode,
                "model": "",
                "session_id": session_id or "",
                "rewritten_query": retrieval_query if did_rewrite else "",
                "used_history": bool(history),
                "latency": Latency(
                    retrieve_ms=retrieve_ms, generate_ms=0, total_ms=total_ms
                ),
                "usage": None,
                "warnings": [f"未生成回答：{reason}"] + warnings,
                "cache_hit": False,
            }
            # 拒答是确定性的（由检索分数决定），同样可以缓存——
            # 下次同样的问题省掉一次检索
            if cache_key:
                self._query_cache.put(cache_key, copy.deepcopy(refusal_response))
            self._record_chat(refusal_response)
            return refusal_response

        context, citations = self.build_context(results)
        messages = self.build_messages(question, context, history=history)

        generate_started = time.perf_counter()
        try:
            result: LLMResult = await self._llm.chat(messages)
        except LLMError as exc:
            generate_ms = int((time.perf_counter() - generate_started) * 1000)
            total_ms = int((time.perf_counter() - total_started) * 1000)
            logger.error("生成回答失败：%s", exc)
            llm_failure_response = {
                "question": question,
                "answer": f"[生成失败] {exc}",
                "citations": citations,
                "retrieved": debug_rows,
                "retrieved_raw": results,
                "refused": False,
                "retrieval_mode": effective_mode,
                "model": self._llm.model,
                "session_id": session_id or "",
                "rewritten_query": retrieval_query if did_rewrite else "",
                "used_history": bool(history),
                "latency": Latency(
                    retrieve_ms=retrieve_ms, generate_ms=generate_ms, total_ms=total_ms
                ),
                "usage": None,
                "warnings": warnings + [f"LLM 调用失败：{exc}"],
                # 瞬时失败不写缓存：下次请求重新尝试，而不是把错误固化下来
                "cache_hit": False,
            }
            self._record_chat(llm_failure_response)
            return llm_failure_response
        generate_ms = int((time.perf_counter() - generate_started) * 1000)

        used_citations, citation_warnings = self.verify_citations(result.text, citations)
        warnings.extend(citation_warnings)

        # 逐句归因校验：专门抓"引用编号合法、但内容并不在被引用片段里"的幻觉。
        # 只产出报告与 warnings，不修改 answer 正文（改正文会破坏可读性且可能误删）。
        attribution: Optional[Dict[str, Any]] = None
        if self._attribution is not None:
            try:
                report = self._attribution.check(result.text, results, citations)
                # 顺手把句子里的引用编号解析成来源明细（句↔片段对齐），
                # 前端做高亮不再需要拿编号自己交叉引用。
                attribution = self.resolve_attribution_sources(
                    report.to_dict(), citations
                )
                warnings.extend(report.warnings)
            except Exception as exc:  # noqa: BLE001 - 校验是增强项，失败不影响回答
                logger.warning("归因校验失败（已跳过）：%s", exc)

        # 第二层兜底：编造检测。模型可能"没拒答却给了没有依据的答案"，
        # 这一层由服务端独立判断，不依赖模型是否遵守提示词。
        #
        # ⚠️ 只看 unsupported（找不到依据的句子），**不看归因总分**。
        # 原因（实测踩过）：模型忘了标 `[n]` 时归因判 no_citation、总分会被拉到 0，
        # 但那种答案是有效的。第一版用「总分 < 0.5」当门槛，把 Q015（"资料里没有专门清单，
        # 以下按保养表整理"+具体检查项）和 Q016（给出相关故障码）两条高质量的部分覆盖回答
        # 误杀成拒答，过度拒答率从 7% 虚增到 21%。no_citation 是格式问题，不是幻觉。
        refusal_reason: Optional[str] = None
        answer_text = result.text
        final_citations = used_citations
        hallucination_reason = refusal_on_hallucination(
            attribution, float(self._settings.refusal_unsupported_ratio or 0.0)
        )
        if hallucination_reason:
            refusal_reason = hallucination_reason
            warnings.append(
                "已按归因校验改为拒答（原回答前 80 字："
                f"{result.text.strip()[:80]}…）。"
            )
            logger.info("拒答兜底触发：%s", hallucination_reason)
            answer_text = REFUSAL_ANSWER
            final_citations = []

        total_ms = int((time.perf_counter() - total_started) * 1000)

        usage = None
        if result.usage:
            usage = TokenUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
            )

        self._remember(session_id, question, answer_text)
        success_response = {
            "question": question,
            "answer": answer_text,
            "citations": final_citations,
            "retrieved": debug_rows,
            "retrieved_raw": results,
            "refused": bool(refusal_reason),
            "refusal_reason": refusal_reason,
            "retrieval_mode": effective_mode,
            "model": result.model or self._llm.model,
            "session_id": session_id or "",
            "rewritten_query": retrieval_query if did_rewrite else "",
            "used_history": bool(history),
            "latency": Latency(
                retrieve_ms=retrieve_ms, generate_ms=generate_ms, total_ms=total_ms
            ),
            "usage": usage,
            "warnings": warnings,
            "attribution": attribution,
            "cache_hit": False,
        }
        if cache_key:
            self._query_cache.put(cache_key, copy.deepcopy(success_response))
        self._record_chat(success_response)
        return success_response

    async def stream_answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        debug: bool = False,
        session_id: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], AsyncIterator[str]]:
        """流式问答。返回 (元信息字典, 文本增量异步迭代器)。

        元信息里包含 citations 与召回明细，通过 SSE 的 meta 事件先发给前端；
        文本增量逐个 yield，前端拼接后即可得到与 /chat 一致的答案。
        多轮行为与 answer() 一致；区别是落历史要在文本流完后做，
        所以这里包了一层迭代器。
        """
        total_started = time.perf_counter()
        question = (question or "").strip()
        warnings: List[str] = []

        history = self._history(session_id)
        retrieval_query, did_rewrite = await self._rewrite_query(question, history, warnings)

        results, effective_mode, retrieve_ms = await self.retrieve_only(
            retrieval_query, top_k=top_k, mode=mode
        )
        debug_rows = HybridRetriever.to_debug(results, include_text=True) if debug else []
        citations: List[Citation] = []
        context = ""

        refuse, reason = self.should_refuse(results, effective_mode)
        if refuse and self._settings.refuse_when_empty:
            async def refusal_stream() -> AsyncIterator[str]:
                yield REFUSAL_ANSWER

            self._remember(session_id, question, REFUSAL_ANSWER, refused=True)
            meta = {
                "question": question,
                "citations": [],
                "retrieved": debug_rows,
                "refused": True,
                "refusal_reason": reason,
                "retrieval_mode": effective_mode,
                "model": "",
                "session_id": session_id or "",
                "rewritten_query": retrieval_query if did_rewrite else "",
                "used_history": bool(history),
                "latency": {
                    "retrieve_ms": retrieve_ms,
                    "total_ms": int((time.perf_counter() - total_started) * 1000),
                },
                "warnings": [f"未生成回答：{reason}"],
            }
            return meta, refusal_stream()

        context, citations = self.build_context(results)
        messages = self.build_messages(question, context, history=history)

        meta = {
            "question": question,
            "citations": [citation.model_dump() for citation in citations],
            "retrieved": debug_rows,
            "refused": False,
            "retrieval_mode": effective_mode,
            "model": self._llm.model,
            "session_id": session_id or "",
            "rewritten_query": retrieval_query if did_rewrite else "",
            "used_history": bool(history),
            "warnings": (["召回片段为空，回答可能无依据"] if not citations else []) + warnings,
        }

        if not (session_id and self._session_store is not None):
            return meta, self._llm.chat_stream(messages)

        # 流式没有"最终答案"这个时点，包一层迭代器：文本流完再落历史
        async def remembered_stream() -> AsyncIterator[str]:
            parts: List[str] = []
            async for chunk in self._llm.chat_stream(messages):
                parts.append(chunk)
                yield chunk
            self._remember(session_id, question, "".join(parts))

        return meta, remembered_stream()
