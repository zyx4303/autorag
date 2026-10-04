"""问答链路：检索 -> 组装带编号的资料 -> 调用 LLM -> 解析引用 -> 返回结构化结果。

核心约束（也是简历上值得讲的点）：
1. 只依据检索到的资料作答，资料里没有就明说，并给出"建议确认渠道"；
2. 每处结论都必须带 [n] 角标，n 与返回的 citations 一一对应，禁止编造引用编号；
3. 检索为空（或向量相似度过低）时不调用大模型，直接返回兜底话术，
   避免"无依据生成"这一最常见的幻觉来源。
"""
from __future__ import annotations

import re
import time
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Set, Tuple

from app.config import Settings
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

REFUSAL_ANSWER = (
    "当前知识库中未找到与该问题相关的依据，因此不做推测性回答。\n\n"
    "建议确认渠道：\n"
    "1. 以随车《用户手册》或《保养手册》的对应章节为准；\n"
    "2. 联系授权服务站，提供 VIN 码由售后顾问核对；\n"
    "3. 需要我回答的话，请把相关手册/工单文档放入 data/documents 目录后调用 "
    "POST /api/v1/ingest 重新入库。"
)

CITATION_PATTERN = re.compile(r"\[(\d{1,2})\]")


class QAEngine:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        llm_client: LLMClient,
    ) -> None:
        self._settings = settings
        self._retriever = retriever
        self._llm = llm_client

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

    def build_messages(self, question: str, context: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(context=context, question=question),
            },
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

    async def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        debug: bool = False,
    ) -> Dict[str, Any]:
        """完整问答，返回 dict（与 ChatResponse 字段对齐）。

        - debug=True 时在返回值的 retrieved 字段里带上召回明细；
        - 返回值额外带一个内部字段 retrieved_raw（RetrievedChunk 列表），
          便于评测脚本复用同一次检索结果，避免重复调用向量库。该字段不会进 HTTP 响应。
        """
        total_started = time.perf_counter()
        question = (question or "").strip()
        warnings: List[str] = []

        results, effective_mode, retrieve_ms = await self.retrieve_only(
            question, top_k=top_k, mode=mode
        )
        debug_rows = HybridRetriever.to_debug(results) if debug else []

        refuse, reason = self.should_refuse(results, effective_mode)
        if refuse and self._settings.refuse_when_empty:
            logger.info("触发兜底回答：%s", reason)
            total_ms = int((time.perf_counter() - total_started) * 1000)
            return {
                "question": question,
                "answer": REFUSAL_ANSWER,
                "citations": [],
                "retrieved": debug_rows,
                "retrieved_raw": results,
                "refused": True,
                "retrieval_mode": effective_mode,
                "model": "",
                "latency": Latency(
                    retrieve_ms=retrieve_ms, generate_ms=0, total_ms=total_ms
                ),
                "usage": None,
                "warnings": [f"未生成回答：{reason}"] + warnings,
            }

        context, citations = self.build_context(results)
        messages = self.build_messages(question, context)

        generate_started = time.perf_counter()
        try:
            result: LLMResult = await self._llm.chat(messages)
        except LLMError as exc:
            generate_ms = int((time.perf_counter() - generate_started) * 1000)
            total_ms = int((time.perf_counter() - total_started) * 1000)
            logger.error("生成回答失败：%s", exc)
            return {
                "question": question,
                "answer": f"[生成失败] {exc}",
                "citations": citations,
                "retrieved": debug_rows,
                "retrieved_raw": results,
                "refused": False,
                "retrieval_mode": effective_mode,
                "model": self._llm.model,
                "latency": Latency(
                    retrieve_ms=retrieve_ms, generate_ms=generate_ms, total_ms=total_ms
                ),
                "usage": None,
                "warnings": warnings + [f"LLM 调用失败：{exc}"],
            }
        generate_ms = int((time.perf_counter() - generate_started) * 1000)

        used_citations, citation_warnings = self.verify_citations(result.text, citations)
        warnings.extend(citation_warnings)
        total_ms = int((time.perf_counter() - total_started) * 1000)

        usage = None
        if result.usage:
            usage = TokenUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                total_tokens=result.total_tokens,
            )

        return {
            "question": question,
            "answer": result.text,
            "citations": used_citations,
            "retrieved": debug_rows,
            "retrieved_raw": results,
            "refused": False,
            "retrieval_mode": effective_mode,
            "model": result.model or self._llm.model,
            "latency": Latency(
                retrieve_ms=retrieve_ms, generate_ms=generate_ms, total_ms=total_ms
            ),
            "usage": usage,
            "warnings": warnings,
        }

    async def stream_answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        mode: Optional[str] = None,
        debug: bool = False,
    ) -> Tuple[Dict[str, Any], AsyncIterator[str]]:
        """流式问答。返回 (元信息字典, 文本增量异步迭代器)。

        元信息里包含 citations 与召回明细，通过 SSE 的 meta 事件先发给前端；
        文本增量逐个 yield，前端拼接后即可得到与 /chat 一致的答案。
        """
        total_started = time.perf_counter()
        question = (question or "").strip()

        results, effective_mode, retrieve_ms = await self.retrieve_only(
            question, top_k=top_k, mode=mode
        )
        debug_rows = HybridRetriever.to_debug(results) if debug else []
        citations: List[Citation] = []
        context = ""

        refuse, reason = self.should_refuse(results, effective_mode)
        if refuse and self._settings.refuse_when_empty:
            async def refusal_stream() -> AsyncIterator[str]:
                yield REFUSAL_ANSWER

            meta = {
                "question": question,
                "citations": [],
                "retrieved": debug_rows,
                "refused": True,
                "retrieval_mode": effective_mode,
                "model": "",
                "latency": {
                    "retrieve_ms": retrieve_ms,
                    "total_ms": int((time.perf_counter() - total_started) * 1000),
                },
                "warnings": [f"未生成回答：{reason}"],
            }
            return meta, refusal_stream()

        context, citations = self.build_context(results)
        messages = self.build_messages(question, context)

        meta = {
            "question": question,
            "citations": [citation.model_dump() for citation in citations],
            "retrieved": debug_rows,
            "refused": False,
            "retrieval_mode": effective_mode,
            "model": self._llm.model,
            "warnings": [] if citations else ["召回片段为空，回答可能无依据"],
        }
        return meta, self._llm.chat_stream(messages)
