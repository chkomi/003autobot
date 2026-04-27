"""백테스트 잡 관련 엔드포인트 (plan v2 §5.2 — 4단계 구현).

연결된 서비스:
  - BacktestService → RQ 큐 (Redis 있을 때) 또는 asyncio BackgroundTask (폴백)
  - DB(backtest_jobs): 상태 추적
  - DB(backtest_results): 완료된 지표

흐름:
  POST /backtests     → 202 Accepted + job_id
  GET  /backtests/{id}→ 상태(queued|running|succeeded|failed) + 결과
"""
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Path, Request
from loguru import logger

from api.schemas.backtests import BacktestJob, BacktestRequest, BacktestResult
from api.schemas.common import Envelope
from api.services.backtest_service import BacktestService

router = APIRouter(prefix="/backtests", tags=["backtests"])


# ── 의존성: BacktestService 싱글턴 ──────────────────────────────

def get_backtest_svc(request: Request) -> BacktestService:
    return request.app.state.backtest_svc

BacktestSvcDep = Annotated[BacktestService, Depends(get_backtest_svc)]


# ── 백테스트 생성 ─────────────────────────────────────────────────

@router.post(
    "",
    response_model=Envelope[BacktestResult],
    status_code=202,
    summary="백테스트 job 생성",
    description=(
        "지정 심볼과 기간에 대한 백테스트를 비동기로 실행한다. "
        "202 Accepted 와 함께 job_id를 반환하며, "
        "GET /backtests/{job_id} 로 상태와 결과를 조회할 수 있다."
    ),
)
async def create_backtest(
    req: BacktestRequest,
    background_tasks: BackgroundTasks,
    svc: BacktestSvcDep,
):
    try:
        job = await svc.create_job(req, background_tasks)
    except Exception as exc:
        logger.exception(f"[create_backtest] 오류: {exc}")
        raise HTTPException(status_code=500, detail=f"job 생성 실패: {exc}") from exc

    return Envelope(
        data=BacktestResult(job=job, metrics=None, error=None),
        generated_at=datetime.now(timezone.utc),
    )


# ── 백테스트 상태/결과 조회 ──────────────────────────────────────

@router.get(
    "/{job_id}",
    response_model=Envelope[BacktestResult],
    summary="백테스트 상태/결과 조회",
    description=(
        "job_id 로 백테스트 상태(queued|running|succeeded|failed)와 "
        "완료된 경우 성과 지표를 반환한다."
    ),
)
async def get_backtest(
    svc: BacktestSvcDep,
    job_id: str = Path(..., min_length=1, description="create_backtest 응답의 job_id"),
):
    result = await svc.get_job(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"job_id '{job_id}' 를 찾을 수 없습니다.")

    return Envelope(
        data=result,
        generated_at=datetime.now(timezone.utc),
    )
