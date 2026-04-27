"""백테스트 RQ job 함수 (plan v2 §3.3 + §4).

이 모듈은 두 가지 경로로 실행된다:
  1. RQ worker 프로세스 (Redis 있을 때): rq worker 가 큐에서 꺼내 실행
  2. asyncio BackgroundTask (Redis 없을 때): API 프로세스 내 백그라운드 실행

진입점:
    run_backtest_job(job_id, symbol, start_date, end_date,
                     strategy_version, params_override)

DB 상태 흐름:
    queued → running → succeeded | failed
"""
from __future__ import annotations

import asyncio
import json
import traceback
from datetime import date as date_type
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from config.settings import OKXConfig
from config.strategy_params import AllStrategyParams
from data.okx_rest_client import OKXRestClient
from backtest.backtester import Backtester
from backtest.data_loader import HistoricalDataLoader
from backtest.performance import PerformanceReport

# OKX REST API 호출 간 최소 대기 시간 (rate-limit 보호)
_FETCH_SLEEP_SEC: float = 0.5


# ── DB 상태 업데이트 헬퍼 ────────────────────────────────────────

async def _set_status(
    job_id: str,
    status: str,
    progress: float = 0.0,
    error_message: Optional[str] = None,
) -> None:
    """백테스트 job의 상태를 DB에 업데이트한다.

    DB 오류가 발생하면 예외를 그대로 전파한다.
    호출부에서 별도 try/except 로 처리해야 한다.
    """
    from api.db.base import get_session_factory
    from api.db.models import BacktestJob
    from sqlalchemy import update

    factory = get_session_factory()
    async with factory() as session:
        await session.execute(
            update(BacktestJob)
            .where(BacktestJob.job_id == job_id)
            .values(
                status=status,
                progress=progress,
                error_message=error_message,
                updated_at=datetime.now(timezone.utc),
            )
        )
        await session.commit()


async def _save_result(
    job_id: str,
    report: PerformanceReport,
    start_date: str,
    end_date: str,
) -> None:
    """PerformanceReport를 backtest_results 테이블에 저장한다."""
    from api.db.base import get_session_factory
    from api.db.models import BacktestResult
    from sqlalchemy import select

    factory = get_session_factory()
    async with factory() as session:
        # 중복 방지: 이미 있으면 skip
        existing = await session.scalar(
            select(BacktestResult).where(BacktestResult.job_id == job_id)
        )
        if existing:
            return

        result = BacktestResult(
            job_id=job_id,
            total_return=report.total_pnl_pct,
            annual_return=_annualize(report.total_pnl_pct, start_date, end_date),
            win_rate=report.win_rate,
            profit_factor=report.profit_factor,
            sharpe_ratio=report.sharpe_ratio,
            max_drawdown=report.max_drawdown_pct,
            trade_count=report.total_trades,
            avg_holding_hours=report.avg_trade_duration_hours or 0.0,
            artifact_path=None,  # Phase 6에서 S3/R2 경로 추가
        )
        session.add(result)
        await session.commit()


def _annualize(total_return_pct: float, start_date: str, end_date: str) -> float:
    """실제 백테스트 기간으로 연간화 수익률을 계산한다.

    백테스트 기간이 0이면 total_return_pct 를 그대로 반환한다.
    """
    if total_return_pct <= -1.0:
        return -1.0
    try:
        d0 = date_type.fromisoformat(start_date)
        d1 = date_type.fromisoformat(end_date)
        years = max((d1 - d0).days / 365.25, 1 / 365.25)  # 최소 1일
    except (ValueError, AttributeError):
        years = 1.33  # 파싱 실패 시 기본값 (약 16개월)
    return (1 + total_return_pct) ** (1 / years) - 1


# ── 실제 백테스트 실행 로직 ──────────────────────────────────────

async def _execute_backtest(
    job_id: str,
    symbol: str,
    start_date: str,
    end_date: str,
    params_override: Optional[dict] = None,
) -> None:
    """백테스트를 실제로 실행하는 비동기 코루틴."""
    logger.info(f"[backtest] 시작 job_id={job_id} symbol={symbol} {start_date}~{end_date}")

    # ── Fix #3: client를 try 블록 진입 전에 None으로 초기화 ────────
    # finally에서 client가 미정의 상태로 참조하는 NameError 방지
    client: Optional[OKXRestClient] = None

    try:
        await _set_status(job_id, "running", progress=0.05)

        # OKX 클라이언트 초기화
        okx_cfg = OKXConfig()
        client = OKXRestClient(okx_cfg)

        # 전략 파라미터 로드 (YAML lru_cache로 1회만 파일 I/O)
        params = AllStrategyParams.from_yaml()
        if params_override:
            logger.debug(f"[backtest] 파라미터 오버라이드: {params_override}")
            # TODO: 필요 시 params_override를 dataclass에 반영

        loader = HistoricalDataLoader(client)

        # ── Fix #9: 타임프레임 간 API 호출 사이 슬립으로 rate-limit 보호 ──
        await _set_status(job_id, "running", progress=0.10)
        logger.info(f"[backtest] 4H 데이터 수집 중...")
        df_4h = await loader.fetch_full_history(symbol, "4h", start_date, end_date)

        await asyncio.sleep(_FETCH_SLEEP_SEC)
        await _set_status(job_id, "running", progress=0.30)
        logger.info(f"[backtest] 1H 데이터 수집 중...")
        df_1h = await loader.fetch_full_history(symbol, "1h", start_date, end_date)

        await asyncio.sleep(_FETCH_SLEEP_SEC)
        await _set_status(job_id, "running", progress=0.55)
        logger.info(f"[backtest] 15M 데이터 수집 중...")
        df_15m = await loader.fetch_full_history(symbol, "15m", start_date, end_date)

        await _set_status(job_id, "running", progress=0.70)
        logger.info(
            f"[backtest] 데이터 수집 완료 "
            f"4h={len(df_4h)} 1h={len(df_1h)} 15m={len(df_15m)}"
        )

        if len(df_4h) < 220:
            raise ValueError(
                f"4H 캔들 부족: {len(df_4h)}개 (EMA200 워밍업 최소 220 필요)"
            )

        # 백테스트 실행
        backtester = Backtester(
            params=params,
            initial_balance=1000.0,
            leverage=3,
            max_position_pct=0.10,
        )
        report: PerformanceReport = backtester.run(df_4h, df_1h, df_15m)

        await _set_status(job_id, "running", progress=0.90)
        logger.info(f"[backtest] 완료: {report.summary()}")

        # 결과 저장 (실제 기간 기반 연간화 포함)
        await _save_result(job_id, report, start_date, end_date)
        await _set_status(job_id, "succeeded", progress=1.0)

        logger.info(f"[backtest] job_id={job_id} 성공")

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error(f"[backtest] job_id={job_id} 실패: {exc}\n{tb}")
        # ── Fix #4: _set_status 실패 자체도 별도로 처리 ───────────
        try:
            await _set_status(
                job_id,
                "failed",
                progress=0.0,
                error_message=str(exc)[:512],
            )
        except Exception as db_exc:
            logger.error(
                f"[backtest] job_id={job_id} DB 상태 업데이트도 실패 "
                f"(job이 'running'으로 남을 수 있음): {db_exc}"
            )
    finally:
        # ── Fix #3: client가 None이어도 안전하게 종료 ────────────
        if client is not None:
            try:
                await client._exchange.close()
            except Exception:
                pass


# ── RQ 진입점 (동기 래퍼) ────────────────────────────────────────

def run_backtest_job(
    job_id: str,
    symbol: str,
    start_date: str,
    end_date: str,
    strategy_version: Optional[str] = None,
    params_override: Optional[dict] = None,
) -> None:
    """RQ 워커에서 호출하는 동기 진입점.

    asyncio 이벤트 루프가 없는 워커 프로세스에서도 안전하게 실행된다.
    """
    logger.info(
        f"[RQ worker] run_backtest_job start "
        f"job_id={job_id} symbol={symbol} {start_date}~{end_date}"
    )
    asyncio.run(
        _execute_backtest(
            job_id=job_id,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            params_override=params_override,
        )
    )
