"""异步入库任务管理：POST /ingest?async_mode 立即返回 job_id，进度可查。

为什么需要它
------------
同步入库把整个 HTTP 请求挂住（重建 ~2.7s，语料更大时线性变差），
客户端超时了服务端还在跑，结果既拿不到也不知道跑没跑完。
异步模式把「提交」和「执行」分开：提交立即返回 job_id，
进度/结果通过 GET /ingest/status/{job_id} 查询。

设计取舍：

- **同一时刻只允许一个入库任务**。入库是索引级的变更操作，
  两个并发入库会造成注册表与向量库状态错乱——这不是性能问题，
  是正确性问题，所以直接拒绝（409）而不是排队。
- **任务跑在事件循环上**（asyncio.create_task）。入库里的 CPU 段
  （切分、本地向量化）仍会短暂阻塞事件循环——这是同步模式就有的行为，
  异步模式先解决「请求挂住」的问题；线程级卸载是下一步的事。
- **任务记录保存在内存**，进程重启即失。入库的持久状态在 registry
  和向量库里，任务记录只是观测窗口；保留最近 N 条用于查询。
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Dict, List, Optional

from app.logging_conf import get_logger

logger = get_logger(__name__)

# 保留最近多少条已结束的任务记录（防内存无界增长）
_MAX_FINISHED_JOBS = 32


class IngestJobManager:
    """管理后台入库任务：提交、进度查询、单飞约束。"""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._running_id: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------ #
    # 提交
    # ------------------------------------------------------------------ #
    def start(
        self,
        ingestion: Any,
        rebuild: bool = False,
        reset_registry: bool = False,
        only_sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """提交一个后台入库任务。已有任务在跑时抛 RuntimeError（调用方转 409）。"""
        if self._running_id is not None:
            raise RuntimeError(
                f"已有入库任务在执行（job={self._running_id}），"
                "入库是索引级变更，不允许并发。请等它结束后再提交。"
            )

        job_id = uuid.uuid4().hex[:12]
        job: Dict[str, Any] = {
            "job_id": job_id,
            "status": "queued",
            "params": {
                "rebuild": rebuild,
                "reset_registry": reset_registry,
                "paths": only_sources,
            },
            "progress": {"phase": "queued"},
            "result": None,
            "error": None,
            "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": None,
        }
        self._jobs[job_id] = job
        self._running_id = job_id

        task = asyncio.create_task(self._run(job, ingestion))
        # 任务结束时把 running 标记清掉；异常在 _run 内部已捕获，这里只兜底
        task.add_done_callback(lambda _t: None)
        logger.info("已提交异步入库任务 %s（rebuild=%s）", job_id, rebuild)
        return {"job_id": job_id, "status": "queued"}

    async def _run(self, job: Dict[str, Any], ingestion: Any) -> None:
        job_id = job["job_id"]

        def on_progress(data: Dict[str, Any]) -> None:
            # 进度回调在 ingest 内部被同步调用，直接更新记录即可
            job["progress"] = data

        job["status"] = "running"
        try:
            response = await ingestion.ingest(
                rebuild=job["params"]["rebuild"],
                reset_registry=job["params"]["reset_registry"],
                only_sources=job["params"]["paths"],
                progress_cb=on_progress,
            )
            job["status"] = "done"
            job["result"] = response.model_dump()
        except Exception as exc:  # noqa: BLE001 - 任务失败要落记录，不能只留下一个悬 task
            job["status"] = "failed"
            job["error"] = f"{type(exc).__name__}: {exc}"
            logger.exception("异步入库任务 %s 失败", job_id)
        finally:
            job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._running_id = None
            self._prune_finished()

    # ------------------------------------------------------------------ #
    # 查询
    # ------------------------------------------------------------------ #
    def status(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        """按提交时间倒序列出任务摘要（不含 files 明细，控制响应体积）。"""
        ordered = sorted(self._jobs.values(), key=lambda j: j["submitted_at"], reverse=True)
        return [
            {
                "job_id": j["job_id"],
                "status": j["status"],
                "submitted_at": j["submitted_at"],
                "finished_at": j["finished_at"],
                "progress": j["progress"],
            }
            for j in ordered
        ]

    @property
    def running_job_id(self) -> Optional[str]:
        return self._running_id

    def _prune_finished(self) -> None:
        """已结束任务超过上限时，淘汰最老的（queued/running 不会被淘汰）。"""
        finished = [jid for jid, j in self._jobs.items() if j["status"] in {"done", "failed"}]
        # self._jobs 是插入序，finished 也按提交序排列，从头淘汰即最老的
        excess = len(finished) - _MAX_FINISHED_JOBS
        for jid in finished[:max(0, excess)]:
            self._jobs.pop(jid, None)


__all__ = ["IngestJobManager"]
