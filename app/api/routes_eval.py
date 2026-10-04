"""评测接口：查看评测集概况、运行评测集、查看历史报告。

注意：/eval/run 会真实调用 LLM，按评测集条数消耗 token。
"""
from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, status

from app.api.deps import get_eval_service, verify_api_key
from app.models import EvalRunRequest, EvalRunResponse
from app.services.eval_service import EvaluationService

router = APIRouter(prefix="/eval", tags=["eval"])


@router.get("/dataset", summary="查看评测集概况（只读，不改动任何数据）")
async def dataset_summary(
    path: str | None = None,
    service: EvaluationService = Depends(get_eval_service),
    _: None = Depends(verify_api_key),
) -> dict:
    try:
        return service.dataset_summary(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/run", response_model=EvalRunResponse, summary="运行评测集（会调用 LLM）")
async def run_eval(
    payload: EvalRunRequest = Body(default_factory=EvalRunRequest),
    service: EvaluationService = Depends(get_eval_service),
    _: None = Depends(verify_api_key),
) -> EvalRunResponse:
    try:
        summary, report = await service.run(
            dataset_path=payload.dataset_path,
            top_k=payload.top_k,
            retrieval_mode=payload.retrieval_mode,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    saved_to = service.save_report(report)
    summary.note = (
        f"{summary.note} 完整报告已保存到 {saved_to}（含每条问题的召回来源、引用编号、"
        "耗时与 token 用量，可逐条人工复核）。"
    )
    return summary


@router.get("/reports", summary="列出历史评测报告（只读）")
async def list_reports() -> dict:
    return {"reports": EvaluationService.list_reports()}
