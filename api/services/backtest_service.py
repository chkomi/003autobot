"""백테스트 job 생성/조회 서비스 (plan v2 §3.3 + §5.2).

실행 경로:
  Redis 있을 때 → RQ 큐에 enqueue → rq worker 프로세스가 처리
  Redis 없을 때 → asyncio BackgroundTask (API 프로세스 내 실행)

상태 저장:
  - DB(backtest_jobs): queued / running / succeeded / failed
  - DB(backtest_results): 완료된 결과 지표

사용 예:
    svc = BacktestService(redis_url="redis://localhost:6379/0")
    job = await svc.create_job(request, background_tasks)
    result = await svc.get_job(job.job_id)
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import BackgroundTasks
from loguru import logger
from sqlalchemy import select

from datetime import date as date_type

from api.db.base import get_session_factory
from api.db.models import BacktestJob as BacktestJobModel
from api.db.models import BacktestResult as BacktestResultModel

from api.schemas.backtests import (
    BacktestJob,
    BacktestMetrics,
    BacktestRequest,
    BacktestResult,
)

# ── RQ 조건부 임포트 ────────────────────────────────────────────

try:
    from rq import Queue as RQQueue, Retry as RQRetry
    from redis import Redis as RQRedis
    _RQ_AVAILABLE = True
except ImportError:
    _RQ_AVAILABLE = False


class BacktestService:
    """백테스트 job 생성/조회 서비스."""

    QUEUE_NAME = "backtest"

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._redis_url = redis_url
        self._queue: Optional["RQQueue"] = None

        if redis_url and _RQ_AVAILABLE:
            try:
                conn = RQRedis.from_url(redis_url)
                self._queue = RQQueue(
                    name=self.QUEUE_NAME,
                    connection=conn,
                    default_timeout=3600,  # 백테스트 최대 1시간
                )
                logger.info(f"[BacktestService] RQ 큐 연결 완료: {redis_url}")
            except Exception as exc:
                logger.warning(f"[BacktestService] RQ 연결 실패 — asyncio 폴백: {exc}")
                self._queue = None
        elif redis_url and not _RQ_AVAILABLE:
            logger.warning("[BacktestService] rq 패키지 미설치 — asyncio 폴백으로 실행됩니다.")

    # ── job 생성 ──────────────────────────────────────────────────

    async def create_job(
        self,
        req: BacktestRequest,
        background_tasks: BackgroundTasks,
    ) -> BacktestJob:
        """새 백테스트 job을 DB에 등록하고 큐에 넣는다."""
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        factory = get_session_factory()
        async with factory() as session:
            db_job = BacktestJobModel(
                job_id=job_id,
                symbol=req.symbol,
                start_date=req.start_date,   # date 객체 그대로 (Date 컬럼)
                end_date=req.end_date,        # date 객체 그대로 (Date 컬럼)
                strategy_version=req.strategy_version,
                params=json.dumps(req.params) if req.params else None,
                status="queued",
                progress=0.0,
                created_at=now,
                updated_at=now,
            )
            session.add(db_job)
            await session.commit()

        logger.info(f"[BacktestService] job 등록: {job_id} {req.symbol} {req.start_date}~{req.end_date}")

        # ── 실행 방식 선택 ───────────────────────────────────────
        if self._queue is not None:
            # RQ 큐에 enqueue
            from api.workers.backtest_worker import run_backtest_job
            # Fix #8: 재시도 로직 추가 (30s → 60s → 120s 간격, 최대 3회)
            self._queue.enqueue(
                run_backtest_job,
                kwargs=dict(
                    job_id=job_id,
                    symbol=req.symbol,
                    start_date=str(req.start_date),
                    end_date=str(req.end_date),
                    strategy_version=req.strategy_version,
                    params_override=req.params or {},
                ),
                job_id=job_id,
                retry=RQRetry(max=3, interval=[30, 60, 120]),
            )
            logger.info(f"[BacktestService] RQ 큐에 enqueue 완료: {job_id}")
        else:
            # asyncio BackgroundTask (Redis 없을 때)
            from api.workers.backtest_worker import _execute_backtest
            background_tasks.add_task(
                _execute_backtest,
                job_id=job_id,
                symbol=req.symbol,
                start_date=str(req.start_date),
                end_date=str(req.end_date),
                params_override=req.params or {},
            )
            logger.info(f"[BacktestService] BackgroundTask로 실행: {job_id}")

        return BacktestJob(
            job_id=job_id,
            status="queued",
            symbol=req.symbol,
            start_date=req.start_date,
            end_date=req.end_date,
            strategy_version=req.strategy_version,
            created_at=now,
            updated_at=now,
            progress=0.0,
        )

    # ── job 조회 ──────────────────────────────────────────────────

    async def get_job(self, job_id: str) -> Optional[BacktestResult]:
        """job_id로 상태와 결과를 조회한다. 없으면 None 반환."""
        factory = get_session_factory()
        async with factory() as session:
            db_job = await session.scalar(
                select(BacktestJobModel).where(BacktestJobModel.job_id == job_id)
            )
            if db_job is None:
                return None

            # Fix #11: DB 컬럼이 Date 타입이므로 SQLAlchemy가 date 객체를 직접 반환
            # 하위호환을 위해 문자열로 저장된 경우도 안전하게 처리
            def _to_date(v):
                if isinstance(v, str):
                    return date_type.fromisoformat(v)
                return v

            job_schema = BacktestJob(
                job_id=db_job.job_id,
                status=db_job.status,
                symbol=db_job.symbol,
                start_date=_to_date(db_job.start_date),
                end_date=_to_date(db_job.end_date),
                strategy_version=db_job.strategy_version,
                created_at=db_job.created_at,
                updated_at=db_job.updated_at,
                progress=db_job.progress,
            )

            # 결과가 있으면 함께 반환
            db_result = await session.scalar(
                select(BacktestResultModel).where(
                    BacktestResultModel.job_id == job_id
                )
            )
            metrics: Optional[BacktestMetrics] = None
            if db_result:
                metrics = BacktestMetrics(
                    total_return=db_result.total_return,
                    annual_return=db_result.annual_return,
                    win_rate=db_result.win_rate,
                    profit_factor=db_result.profit_factor,
                    sharpe_ratio=db_result.sharpe_ratio,
                    max_drawdown=db_result.max_drawdown,
                    trade_count=db_result.trade_count,
                    avg_holding_hours=db_result.avg_holding_hours,
                )

            return BacktestResult(
                job=job_schema,
                metrics=metrics,
                error=db_job.error_message,
            )
