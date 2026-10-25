"""Cross-encoder 精排（可选增强）。

为什么需要这一层
----------------
`HybridRetriever._rerank()` 做的是**启发式**重排：关键词覆盖率、整串命中、
高精度 token、章节命中。它可解释、零依赖，但本质仍是"字面匹配 + 规则加权"，
无法判断"这段话是否真的回答了这个问题"。

项目实测记录里留下的两个失败样例，病因其实是同一个：
  问「车辆失火是什么原因」时，保修条款片段因为含"车/辆/失/火/原/因"这些单字
  而拿到很高的词覆盖率，把真正写着「P0300 | 检测到随机或多缸失火」的故障码片段
  挤到后面。继续加权重只会针对样例过拟合 —— 这已经是启发式的天花板。

Cross-encoder 把 `(query, passage)` **一起**送进模型做联合编码，直接输出
"这段话对该问题的相关性"。实测在本项目的故障码/保养数据上区分明显：
  问「车辆失火是什么原因」：正确片段 logit -0.10，保修条款 -5.38，明显分开。

为什么不用 sentence-transformers 的 CrossEncoder
------------------------------------------------
实测踩坑：`CrossEncoder.predict()` 对 `num_labels=1` 的模型**会自动套一层
sigmoid** 再返回。如果调用方再 sigmoid 一次（本模块最初的写法），
0.475 和 0.013 会被二次压缩成 0.617 和 0.503 —— 区分度几乎归零，
表现为"精排后排名完全不变"，很容易误判成模型不行。

这里直接用 `transformers` 的 `AutoModelForSequenceClassification`，
拿**原始 logit** 自己转概率，行为完全可控，也不依赖第三方封装的版本差异。

工程取舍
--------
1. **懒加载**：首次检索时才加载权重（Embedding 之外再多 1GB，不能拖慢启动）。
2. **线程池**：CPU 推理是同步阻塞的，必须走 `asyncio.to_thread`。
3. **失败降级**：加载或推理任一步失败都退回启发式结果，绝不让检索 500。
4. **只精排头部**：只对前 `top_n` 个候选打分，控制耗时。
"""
from __future__ import annotations

import asyncio
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from app.config import Settings
from app.core.embedding import (  # 复用现有的 HF 镜像/证书处理，避免重复实现
    _apply_hf_endpoint as apply_hf_endpoint,
    _configure_hf_environment as configure_hf_environment,
)
from app.logging_conf import get_logger

logger = get_logger(__name__)

DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-base"

__all__ = ["RerankerError", "CrossEncoderReranker", "build_reranker"]


class RerankerError(RuntimeError):
    """精排模型加载或推理失败。"""


def _sigmoid(value: float) -> float:
    """把 cross-encoder 的原始 logit 压到 0~1，便于展示与设阈值。

    单调变换不改变排序；只是让 `score` 字段对人类可读、可跨查询比较。
    显式处理符号避免 math.exp 溢出。
    """
    try:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)
    except OverflowError:  # pragma: no cover - 极端值兜底
        return 0.0 if value < 0 else 1.0


def quantize_linear_layers(model: Any, torch_module: Any) -> Any:
    """对模型的 Linear 层做 int8 动态量化，返回量化后的模型。

    单独抽成模块级函数是为了可测：单测用一个几 KB 的小模型验证
    「Linear 被替换成量化层」，不用加载 1.1GB 的真实精排模型。

    为什么选动态量化而不是 ONNX 导出：零导出风险（XLMRoberta 导出
    ONNX 要额外依赖与校验），收益实测已足够——CPU 上 2491ms → 885ms。
    """
    return torch_module.ao.quantization.quantize_dynamic(
        model, {torch_module.nn.Linear}, dtype=torch_module.qint8
    )


class CrossEncoderReranker:
    """基于 transformers 序列分类头的语义精排。

    名字保留 CrossEncoder，因为它在检索语境里就是这个含义
    （双塔 embedding 之外，把 query 与 passage 拼在一起编码）。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        device: str = "cpu",
        max_length: int = 512,
        batch_size: int = 16,
        hf_endpoint: str = "",
        local_files_only: bool = False,
        quantize: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device or "cpu"
        self._max_length = max(64, int(max_length))
        self._batch_size = max(1, int(batch_size))
        self._quantize = bool(quantize)
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._thread_lock = threading.Lock()
        self._load_error: Optional[str] = None

        # 与 LocalSentenceTransformerEmbedder 同样的坑：.env 里的 HF_ENDPOINT
        # 只会进 Settings 对象，不会自动写进 os.environ，而 huggingface_hub
        # 只认环境变量（且在 import 时把端点固化）。这里必须显式同步。
        if hf_endpoint:
            apply_hf_endpoint(hf_endpoint)
        self._local_files_only = bool(local_files_only) or os.getenv(
            "HF_HUB_OFFLINE", ""
        ).strip().lower() in {"1", "true", "yes"}

    # ------------------------------------------------------------------ #
    # 模型加载
    # ------------------------------------------------------------------ #
    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def _ensure_model(self) -> Tuple[Any, Any]:
        """懒加载 tokenizer + 序列分类模型；并发调用时只加载一次。"""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        with self._thread_lock:
            if self._model is not None and self._tokenizer is not None:
                return self._model, self._tokenizer

            configure_hf_environment()
            try:
                import torch  # type: ignore
                from transformers import (  # type: ignore
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )
            except ImportError as exc:
                raise RerankerError(
                    "精排需要可选依赖，请执行： pip install -r requirements-local.txt"
                ) from exc

            logger.info("正在加载精排模型（首次需下载约 1.1GB 权重）：%s", self._model_name)
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self._model_name, local_files_only=self._local_files_only
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    self._model_name, local_files_only=self._local_files_only
                )
                model.to(self._device)
                model.eval()
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                raise RerankerError(
                    f"加载精排模型失败：{self._load_error}\n"
                    "常见原因与处理方式：\n"
                    "  1) 未下载权重且处于离线模式（HF_HUB_OFFLINE=1）→ "
                    "先联网下载，或把 RERANK_MODEL 指向本地目录；\n"
                    "  2) 无法访问 huggingface.co → 设置 HF_ENDPOINT=https://hf-mirror.com；\n"
                    "  3) Windows 上 HF 缓存 symlink 失败导致 snapshots 为空 → "
                    "用 snapshot_download(local_dir=...) 落到普通目录，再把 RERANK_MODEL 指过去；\n"
                    "  4) 依赖缺失 → pip install -r requirements-local.txt"
                ) from exc

            self._torch = torch
            # int8 动态量化：加载后立刻做（一次性 3s 成本），推理全程受益。
            # 实测 CPU 上 2491ms → 885ms，top-8 成员集与 fp32 100% 一致。
            if self._quantize:
                model = quantize_linear_layers(model, torch)
                logger.info("精排模型已完成 int8 动态量化（Linear 层）")
            self._model = model
            self._tokenizer = tokenizer
            logger.info(
                "精排模型加载完成：%s（device=%s，max_length=%d）",
                self._model_name,
                self._device,
                self._max_length,
            )
            return model, tokenizer

    # ------------------------------------------------------------------ #
    # 推理
    # ------------------------------------------------------------------ #
    def _score_sync(self, query: str, texts: List[str]) -> List[float]:
        """返回**原始 logit**（不做 sigmoid）。

        这里刻意不走 sentence-transformers 的 CrossEncoder.predict()：
        它会对 num_labels=1 的模型自动 sigmoid，调用方再做一次就会把
        区分度压平（实测 0.475/0.013 被压成 0.617/0.503）。
        """
        model, tokenizer = self._ensure_model()
        torch = self._torch
        if torch is None:  # pragma: no cover - 理论上不会发生
            raise RerankerError("torch 未就绪")

        scores: List[float] = []
        # 自己分批，避免长片段全量padding导致显存/内存峰值过高
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            encoded = tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
            # num_labels=1 → shape (batch, 1)
            scores.extend(float(v) for v in logits.view(-1).tolist())
        return scores

    async def rerank(
        self,
        query: str,
        items: List[Dict[str, Any]],
        top_n: int = 20,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """对候选片段做语义精排。

        Args:
            query: 原始查询
            items: 已按启发式分数排好序的候选（来自 `_fuse`）
            top_n: 只对前 N 个候选调用模型，控制 CPU 推理成本
            text_key: 片段正文字段名

        Returns:
            重排后的列表。**失败时原样返回**（降级，不抛异常给调用方）。
        """
        if not items:
            return items

        limit = max(1, int(top_n))
        head = items[:limit]
        tail = items[limit:]

        texts = [str(item.get(text_key, "") or "") for item in head]
        if not any(texts):
            return items

        try:
            raw_scores = await asyncio.to_thread(self._score_sync, query, texts)
        except Exception as exc:  # noqa: BLE001 - 精排失败必须降级
            self._load_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            logger.warning("精排失败，回退到启发式顺序：%s", self._load_error)
            return items

        for item, raw in zip(head, raw_scores):
            # 保留原启发式分，便于 /retrieve 调试时对比两种排序的差异
            item["heuristic_score"] = float(item.get("score", 0.0))
            item["rerank_raw"] = round(float(raw), 4)
            item["rerank_score"] = round(_sigmoid(float(raw)), 6)
            # 最终 score 用精排概率：与启发式分同处 0~1 区间，语义清晰
            item["score"] = item["rerank_score"]

        # 按原始 logit 排序（与按 sigmoid 排序等价，但避免精度损失）
        head.sort(key=lambda x: float(x.get("rerank_raw", 0.0)), reverse=True)
        return head + tail

    # ------------------------------------------------------------------ #
    # 可观测
    # ------------------------------------------------------------------ #
    def describe(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "model": self._model_name,
            "device": self._device,
            "max_length": self._max_length,
            "batch_size": self._batch_size,
            "loaded": self.loaded,
            "load_error": self._load_error,
        }


def build_reranker(settings: Settings) -> Optional[CrossEncoderReranker]:
    """按配置构造精排器；未启用时返回 None。"""
    if not getattr(settings, "rerank_enabled", False):
        return None

    model_name = (getattr(settings, "rerank_model", "") or DEFAULT_RERANK_MODEL).strip()
    # 占位符或空值回退默认，避免启动就崩
    if not model_name or model_name == "[待补充]":
        model_name = DEFAULT_RERANK_MODEL
    # 允许写相对路径（相对项目根），便于把模型放进项目内 models/
    if not os.path.isabs(model_name) and os.path.isdir(model_name):
        model_name = os.path.abspath(model_name)

    return CrossEncoderReranker(
        model_name=model_name,
        device=getattr(settings, "rerank_device", "cpu") or "cpu",
        max_length=getattr(settings, "rerank_max_length", 512),
        batch_size=getattr(settings, "rerank_batch_size", 16),
        hf_endpoint=os.getenv("HF_ENDPOINT", "").strip(),
        quantize=bool(getattr(settings, "rerank_quantize", True)),
    )
