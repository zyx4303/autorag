"""声明抽取引擎：用 LLM 把文档片段转成结构化声明。

三个工程要点：
1. **强制 JSON 输出**：提示词里给出严格 schema 与示例，解析失败时做一次修复重试，
   仍失败则跳过并记录原因（绝不把解析失败的文本硬塞进库里）。
2. **增量抽取**：只对"内容哈希发生变化"的片段调用 LLM。
   一份 87 片段的文档改一行，实际只需 1 次 LLM 调用——这是成本能控住的关键。
3. **禁止无依据抽取**：要求每条声明的 evidence 必须是片段原文的连续子串，
   服务端做子串校验，不通过就丢弃该条（防止模型"顺手补充常识"）。
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.core.llm_client import LLMClient, LLMError
from app.core.text_utils import stable_hash
from app.guard.claims import Claim, CLAIM_TYPES, normalize_value
from app.logging_conf import get_logger
from app.models import Chunk

logger = get_logger(__name__)

EXTRACT_SYSTEM_PROMPT = """你是知识库结构化抽取器。任务：从给定的文本片段中抽取"原子化声明"。

一条声明 = 一个最小可比较的事实，格式为：主体 | 属性 | 值 | 限定条件。

抽取规则（必须严格遵守）：
1. 只抽取片段中**明确写出**的事实，禁止依据常识补充、禁止推断、禁止改写数值。
2. 一条声明只表达一个事实。同一个句子包含两个事实时，拆成两条。
3. 数值必须原样保留（含千分位、单位、区间、比较符号）。
4. 限定条件（车型、年款、气候、里程/时间先后规则等）放进 qualifiers 数组；
   若片段声明了适用范围，必须提取，不得省略。
5. evidence 字段必须是**片段原文中连续出现的子串**（用于人工核对）。
6. 若片段里没有任何可抽取的事实（如纯说明性文字、目录、免责声明），返回空数组。

claim_type 只能取以下之一：
- interval   周期/里程类（如"更换机油：每 8000 km 或 8 个月"）
- spec       规格参数类（如"机油黏度：5W-30"）
- threshold  阈值类（如"刹车片低于 3mm 需更换"）
- condition  条件/适用范围（如"恶劣条件按正常间隔的 3/4 执行"）
- policy     条款政策类（如"包修期：3 年或 60000 km"）
- code       故障码定义（如"P0420：催化器效率低于阈值"）

严格输出 JSON，不要任何解释文字，格式：
{"claims":[{"claim_type":"interval","subject":"更换发动机机油","predicate":"周期",
"value":"每 8,000 km 或每 8 个月","qualifiers":["以先到者为准"],
"evidence":"更换发动机机油 | 每 8,000 km 或每 8 个月（以先到者为准）"}]}"""

EXTRACT_USER_TEMPLATE = """来源文档：{source}
章节路径：{section}

待抽取片段：
\"\"\"
{chunk_text}
\"\"\"

请按规则抽取原子化声明，直接输出 JSON。"""


class ClaimExtractor:
    def __init__(self, llm: LLMClient, max_retries: int = 1) -> None:
        self._llm = llm
        self._max_retries = max(1, max_retries)
        self.stats = {"llm_calls": 0, "chunks_with_claims": 0, "chunks_empty": 0,
                      "claims_extracted": 0, "claims_dropped_no_evidence": 0,
                      "parse_failures": 0, "reused_chunks": 0}

    # ------------------------------------------------------------------ #
    async def extract_from_chunk(self, chunk: Chunk, doc_checksum: str) -> List[Claim]:
        """从单个片段抽取声明（会调用 LLM）。"""
        messages = [
            {"role": "system", "content": EXTRACT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": EXTRACT_USER_TEMPLATE.format(
                    source=chunk.source,
                    section=chunk.section or "（无）",
                    chunk_text=chunk.text[:4000],
                ),
            },
        ]

        payload: Optional[Dict[str, Any]] = None
        last_error = ""
        for attempt in range(1, self._max_retries + 1):
            try:
                self.stats["llm_calls"] += 1
                result = await self._llm.chat(
                    messages, temperature=0.0, max_tokens=1600
                )
                payload = self._parse_json(result.text)
                if payload is not None:
                    break
                last_error = "返回内容不是合法 JSON"
            except LLMError as exc:
                last_error = str(exc)
                logger.warning("声明抽取调用失败（第 %d 次）：%s", attempt, exc)
            if attempt < self._max_retries:
                # 重试时把上一次的原始返回附上，要求"仅修正格式"
                messages.append({"role": "assistant", "content": result.text[:1500]})
                messages.append(
                    {"role": "user", "content": "上面的输出不是合法 JSON，请只输出修正后的 JSON，不要解释。"}
                )

        if payload is None:
            self.stats["parse_failures"] += 1
            logger.warning("声明抽取解析失败，跳过片段 %s：%s", chunk.chunk_id, last_error)
            return []

        raw_claims = payload.get("claims")
        if not isinstance(raw_claims, list):
            self.stats["parse_failures"] += 1
            return []
        if not raw_claims:
            self.stats["chunks_empty"] += 1
            return []

        claims: List[Claim] = []
        for item in raw_claims:
            claim = self._to_claim(item, chunk, doc_checksum)
            if claim is not None:
                claims.append(claim)

        if claims:
            self.stats["chunks_with_claims"] += 1
            self.stats["claims_extracted"] += len(claims)
        else:
            self.stats["chunks_empty"] += 1
        return claims

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """稳健解析：容忍 ```json 包裹、前后废话、尾随逗号。"""
        if not text:
            return None
        cleaned = text.strip()
        # 去掉 markdown 代码围栏
        fence = re.search(r"```(?:json)?\s*(.+?)```", cleaned, flags=re.S)
        if fence:
            cleaned = fence.group(1).strip()
        # 截取第一个 { 到最后一个 }
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # 去掉对象/数组尾随逗号后再试一次
            fixed = re.sub(r",\s*([}\]])", r"\1", cleaned)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                return None

    def _to_claim(self, item: Any, chunk: Chunk, doc_checksum: str) -> Optional[Claim]:
        """把 LLM 返回的一条记录转成 Claim，并做证据校验。"""
        if not isinstance(item, dict):
            return None
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        value = str(item.get("value") or "").strip()
        if not (subject and value):
            return None

        claim_type = str(item.get("claim_type") or "spec").strip().lower()
        if claim_type not in CLAIM_TYPES:
            claim_type = "spec"

        qualifiers = item.get("qualifiers")
        if isinstance(qualifiers, str):
            qualifiers = [qualifiers]
        qualifier_list = [str(q).strip() for q in (qualifiers or []) if str(q).strip()]

        evidence = str(item.get("evidence") or "").strip()

        # 证据校验：evidence 必须是片段原文的连续子串（忽略空白差异）
        if evidence:
            compact_chunk = re.sub(r"\s+", "", chunk.text)
            compact_evidence = re.sub(r"\s+", "", evidence)
            if compact_evidence and compact_evidence not in compact_chunk:
                self.stats["claims_dropped_no_evidence"] += 1
                logger.debug("丢弃无依据声明（evidence 不在原文中）：%s", evidence[:60])
                return None
        else:
            # 没有 evidence 的声明不可核对，直接丢弃
            self.stats["claims_dropped_no_evidence"] += 1
            return None

        char_start = chunk.text.find(evidence[:40])
        if char_start < 0:
            char_start = 0

        return Claim(
            claim_key=Claim.build_key(chunk.source, claim_type, subject, predicate or "值"),
            doc_source=chunk.source,
            section=chunk.section,
            claim_type=claim_type,
            subject=subject,
            predicate=predicate or "值",
            value=value,
            value_normalized=normalize_value(value),
            qualifiers=qualifier_list,
            evidence=evidence,
            chunk_id=chunk.chunk_id,
            char_start=chunk.char_start + char_start,
            char_end=chunk.char_start + char_start + len(evidence),
            doc_checksum=doc_checksum,
            extracted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            model=self._llm.model,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def chunk_content_hash(text: str) -> str:
        """片段内容哈希：用于增量抽取判断（内容没变就不重复调 LLM）。"""
        return stable_hash(re.sub(r"\s+", "", text), 16)


def plan_incremental(
    chunks: List[Chunk],
    known_hashes: Dict[str, str],
) -> Tuple[List[Chunk], List[Chunk]]:
    """把片段分成"需要重新抽取"和"可直接复用旧声明"两组。

    known_hashes: {chunk_id: 内容哈希}（来自上一次抽取时记录的哈希）
    """
    to_extract: List[Chunk] = []
    reusable: List[Chunk] = []
    for chunk in chunks:
        current = ClaimExtractor.chunk_content_hash(chunk.text)
        if known_hashes.get(chunk.chunk_id) == current:
            reusable.append(chunk)
        else:
            to_extract.append(chunk)
    return to_extract, reusable
