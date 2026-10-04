"""LLM 调用：OpenAI 兼容 /v1/chat/completions（同步一次性 + 流式）。

刻意不引入 openai SDK，只用 httpx，好处：
- 依赖更薄，任何"OpenAI 兼容"服务都能接（DeepSeek / 百炼 / 硅基流动 / vLLM / Ollama / OpenAI）；
- 异常信息完全可控，方便把 401/404 这类配置错误翻译成中文提示。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from app.config import Settings, is_placeholder
from app.logging_conf import get_logger

logger = get_logger(__name__)


class LLMError(RuntimeError):
    """LLM 调用失败（鉴权、限流、超时、返回结构异常等）。"""


@dataclass
class LLMResult:
    text: str = ""
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    finish_reason: str = ""

    @property
    def prompt_tokens(self) -> Optional[int]:
        value = self.usage.get("prompt_tokens")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def completion_tokens(self) -> Optional[int]:
        value = self.usage.get("completion_tokens")
        return int(value) if isinstance(value, (int, float)) else None

    @property
    def total_tokens(self) -> Optional[int]:
        value = self.usage.get("total_tokens")
        return int(value) if isinstance(value, (int, float)) else None


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(timeout=settings.llm_timeout_seconds)

    # ------------------------------------------------------------------ #
    @property
    def model(self) -> str:
        return self._settings.llm_model

    @property
    def url(self) -> str:
        return f"{self._settings.llm_base_url.rstrip('/')}/chat/completions"

    @property
    def configured(self) -> bool:
        """是否已配置可用的 LLM 密钥（未配置时问答接口会明确报错，而不是瞎编答案）。"""
        return not is_placeholder(self._settings.llm_api_key)

    async def aclose(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------ #
    def _headers(self) -> Dict[str, str]:
        if not self.configured:
            raise LLMError(
                "LLM_API_KEY 未配置（当前为空或仍是 [待补充]）。请在 .env 中填写后重启服务。"
            )
        return {
            "Authorization": f"Bearer {self._settings.llm_api_key}",
            "Content-Type": "application/json",
        }

    def _payload(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
        stream: bool,
    ) -> Dict[str, Any]:
        return {
            "model": self._settings.llm_model,
            "messages": messages,
            "temperature": (
                self._settings.llm_temperature if temperature is None else float(temperature)
            ),
            "max_tokens": int(max_tokens or self._settings.llm_max_tokens),
            "stream": stream,
        }

    @staticmethod
    def _explain_status(status: int, body: str) -> str:
        if status == 401:
            return f"LLM 鉴权失败(401)：API Key 无效或未通过。返回内容：{body[:200]}"
        if status == 404:
            return (
                f"LLM 接口 404：请检查 LLM_BASE_URL 是否包含 /v1、以及 LLM_MODEL 名称是否正确。"
                f"返回内容：{body[:200]}"
            )
        if status == 429:
            return f"LLM 触发限流(429)：请降低并发或稍后重试。返回内容：{body[:200]}"
        return f"LLM 调用失败({status})：{body[:300]}"

    # ------------------------------------------------------------------ #
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResult:
        """一次性返回完整回答。"""
        headers = self._headers()
        payload = self._payload(messages, temperature, max_tokens, stream=False)
        last_error: Optional[str] = None

        for attempt in range(1, max(1, self._settings.llm_max_retries) + 1):
            try:
                response = await self._client.post(self.url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                last_error = f"LLM 请求超时（{self._settings.llm_timeout_seconds}s）：{exc}"
                logger.warning("%s（第 %d 次）", last_error, attempt)
            except httpx.HTTPError as exc:
                last_error = f"LLM 网络异常：{exc}"
                logger.warning("%s（第 %d 次）", last_error, attempt)
            else:
                if response.status_code < 400:
                    return self._parse_response(response.json())
                body = response.text
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    raise LLMError(self._explain_status(response.status_code, body))
                last_error = self._explain_status(response.status_code, body)
                logger.warning("%s（第 %d 次）", last_error, attempt)

            if attempt < max(1, self._settings.llm_max_retries):
                await asyncio.sleep(min(2 ** (attempt - 1), 8))

        raise LLMError(last_error or "LLM 调用失败，且无具体错误信息")

    @staticmethod
    def _parse_response(data: Dict[str, Any]) -> LLMResult:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError(f"LLM 返回结构异常（缺少 choices）：{str(data)[:200]}")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if content is None:
            raise LLMError(f"LLM 返回内容为空：{str(choices[0])[:200]}")
        return LLMResult(
            text=str(content).strip(),
            model=str(data.get("model", "")),
            usage=dict(data.get("usage") or {}),
            finish_reason=str(choices[0].get("finish_reason", "")),
        )

    # ------------------------------------------------------------------ #
    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[str]:
        """流式返回增量文本（仅 yield 内容增量，不 yield 元数据）。"""
        headers = self._headers()
        payload = self._payload(messages, temperature, max_tokens, stream=True)

        try:
            async with self._client.stream(
                "POST", self.url, json=payload, headers=headers
            ) as response:
                if response.status_code >= 400:
                    body = (await response.aread()).decode("utf-8", errors="ignore")
                    raise LLMError(self._explain_status(response.status_code, body))
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if data == "[DONE]":
                        break
                    try:
                        parsed = json.loads(data)
                    except json.JSONDecodeError:
                        logger.debug("跳过无法解析的流式分片：%s", data[:120])
                        continue
                    for choice in parsed.get("choices") or []:
                        delta = choice.get("delta") or {}
                        piece = delta.get("content")
                        if piece:
                            yield str(piece)
        except httpx.TimeoutException as exc:
            raise LLMError(f"LLM 流式请求超时：{exc}") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM 流式请求网络异常：{exc}") from exc

    # ------------------------------------------------------------------ #
    async def health_check(self) -> Dict[str, Any]:
        """轻量连通性检查：只做一次极短的真实请求，不伪造结果。"""
        if not self.configured:
            return {"ok": False, "detail": "LLM_API_KEY 未配置（空或 [待补充]）"}
        try:
            result = await self.chat(
                messages=[{"role": "user", "content": "ping"}],
                temperature=0.0,
                max_tokens=1,
            )
        except LLMError as exc:
            return {"ok": False, "detail": str(exc)[:300]}
        return {"ok": True, "detail": f"连通正常，model={result.model or self.model}"}
