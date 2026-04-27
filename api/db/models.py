"""SQLAlchemy ORM 모델 (plan v2 §3.4).

테이블 구성:
  기존 SQLite 테이블 미러링:
    - trades          — 거래 이력
    - daily_pnl       — 일별 PnL
    - equity_snapshots— 자산 스냅샷
    - bot_events      — 봇 이벤트 로그
    - candle_cache    — OHLCV 캐시

  신규 API 전용 테이블:
    - symbols         — 심볼 메타데이터
    - score_snapshots — 시점별 점수 이력 (리더보드용)
    - backtest_jobs   — 백테스트 작업 큐
    - backtest_results— 백테스트 결과

시간 표준 (plan v2 §16):
  - 저장: 모든 타임스탬프 UTC
  - PostgreSQL: TIMESTAMP WITH TIME ZONE
  - SQLite: TEXT (ISO 8601, UTC)
"""
from __future__ import annotations

from datetime import date as date_type
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from api.db.base import Base


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────
# 기존 SQLite 테이블 미러링
# ─────────────────────────────────────────────────────────────────

class Trade(Base):
    """거래 이력 — 기존 SQLite ``trades`` 테이블과 1:1 대응."""
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(8))          # LONG | SHORT
    status: Mapped[str] = mapped_column(String(16), index=True) # OPEN | CLOSED | ...

    entry_price: Mapped[Optional[float]] = mapped_column(Float)
    exit_price: Mapped[Optional[float]] = mapped_column(Float)
    quantity: Mapped[Optional[float]] = mapped_column(Float)
    leverage: Mapped[Optional[int]] = mapped_column(Integer)
    stop_loss: Mapped[Optional[float]] = mapped_column(Float)
    take_profit_1: Mapped[Optional[float]] = mapped_column(Float)
    take_profit_2: Mapped[Optional[float]] = mapped_column(Float)

    pnl_usdt: Mapped[Optional[float]] = mapped_column(Float)
    pnl_pct: Mapped[Optional[float]] = mapped_column(Float)

    entry_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    exit_reason: Mapped[Optional[str]] = mapped_column(String(64))

    signal_confidence: Mapped[Optional[float]] = mapped_column(Float)
    atr_at_entry: Mapped[Optional[float]] = mapped_column(Float)
    sl_algo_id: Mapped[Optional[str]] = mapped_column(String(32))
    tp_algo_id: Mapped[Optional[str]] = mapped_column(String(32))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now_utc
    )

    __table_args__ = (
        Index("ix_trades_symbol_status", "symbol", "status"),
        Index("ix_trades_entry_time", "entry_time"),
    )


class DailyPnl(Base):
    """일별 PnL — 기존 SQLite ``daily_pnl`` 테이블과 1:1 대응."""
    __tablename__ = "daily_pnl"

    date: Mapped[str] = mapped_column(String(10), primary_key=True)  # YYYY-MM-DD KST
    pnl_usdt: Mapped[float] = mapped_column(Float, default=0.0)
    trade_count: Mapped[int] = mapped_column(Integer, default=0)
    win_count: Mapped[int] = mapped_column(Integer, default=0)
    peak_equity: Mapped[Optional[float]] = mapped_column(Float)
    min_equity: Mapped[Optional[float]] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now_utc, onupdate=_now_utc
    )


class EquitySnapshot(Base):
    """자산 스냅샷 — 기존 SQLite ``equity_snapshots`` 테이블과 1:1 대응."""
    __tablename__ = "equity_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    equity: Mapped[float] = mapped_column(Float)
    free: Mapped[Optional[float]] = mapped_column(Float)
    used: Mapped[Optional[float]] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now_utc, index=True
    )


class BotEvent(Base):
    """봇 이벤트 로그 — 기존 SQLite ``bot_events`` 테이블과 1:1 대응."""
    __tablename__ = "bot_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(32), index=True)
    level: Mapped[str] = mapped_column(String(16))
    message: Mapped[str] = mapped_column(Text)
    event_metadata: Mapped[Optional[str]] = mapped_column("metadata", Text)  # JSON string
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now_utc, index=True
    )


class CandleCache(Base):
    """OHLCV 캔들 캐시 — 기존 SQLite ``candle_cache`` 테이블과 1:1 대응."""
    __tablename__ = "candle_cache"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32))
    timeframe: Mapped[str] = mapped_column(String(8))
    timestamp: Mapped[str] = mapped_column(String(32))   # ISO 8601 UTC
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_candle"),
        Index("ix_candle_symbol_tf_ts", "symbol", "timeframe", "timestamp"),
    )


# ─────────────────────────────────────────────────────────────────
# 신규 API 전용 테이블
# ─────────────────────────────────────────────────────────────────

class Symbol(Base):
    """심볼 메타데이터 레지스트리."""
    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    base: Mapped[str] = mapped_column(String(16))    # BTC
    quote: Mapped[str] = mapped_column(String(16))   # USDT
    contract_type: Mapped[str] = mapped_column(String(16), default="SWAP")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    qty_precision: Mapped[int] = mapped_column(Integer, default=4)
    min_qty: Mapped[float] = mapped_column(Float, default=0.001)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now_utc
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now_utc, onupdate=_now_utc
    )


class ScoreSnapshot(Base):
    """시점별 점수 스냅샷 — 리더보드 이력 및 Phase 4 캐시 대체 기반."""
    __tablename__ = "score_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    long_score: Mapped[float] = mapped_column(Float)
    short_score: Mapped[float] = mapped_column(Float)
    neutral: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(8))        # LONG | SHORT | NEUTRAL
    signal_strength: Mapped[str] = mapped_column(String(16))

    # 6개 차원 점수
    score_trend: Mapped[float] = mapped_column(Float)
    score_momentum: Mapped[float] = mapped_column(Float)
    score_volume: Mapped[float] = mapped_column(Float)
    score_volatility: Mapped[float] = mapped_column(Float)
    score_sentiment: Mapped[float] = mapped_column(Float)
    score_macro: Mapped[float] = mapped_column(Float)

    last_price: Mapped[Optional[float]] = mapped_column(Float)
    change_24h: Mapped[Optional[float]] = mapped_column(Float)

    calculated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now_utc, index=True
    )

    __table_args__ = (
        Index("ix_score_symbol_calc_at", "symbol", "calculated_at"),
    )


class BacktestJob(Base):
    """백테스트 작업 큐 (plan v2 §5.2, Phase 4 RQ 연동 기반)."""
    __tablename__ = "backtest_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    start_date: Mapped[date_type] = mapped_column(Date)   # Fix #11: DATE 타입
    end_date: Mapped[date_type] = mapped_column(Date)
    strategy_version: Mapped[Optional[str]] = mapped_column(String(32))
    params: Mapped[Optional[str]] = mapped_column(Text)   # JSON

    status: Mapped[str] = mapped_column(
        String(16), default="queued", index=True
    )  # queued | running | succeeded | failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[Optional[str]] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now_utc
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now_utc, onupdate=_now_utc
    )


class BacktestResult(Base):
    """백테스트 결과 지표 저장."""
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(
        String(64), unique=True, index=True
    )  # backtest_jobs.job_id FK (명시적 FK는 Phase 3 마이그레이션에서 추가)

    total_return: Mapped[float] = mapped_column(Float)
    annual_return: Mapped[float] = mapped_column(Float)
    win_rate: Mapped[float] = mapped_column(Float)
    profit_factor: Mapped[float] = mapped_column(Float)
    sharpe_ratio: Mapped[float] = mapped_column(Float)
    max_drawdown: Mapped[float] = mapped_column(Float)
    trade_count: Mapped[int] = mapped_column(Integer)
    avg_holding_hours: Mapped[float] = mapped_column(Float)

    # 원시 데이터(거래 목록 등)는 Blob/S3에 저장, 여기선 경로만 기록
    artifact_path: Mapped[Optional[str]] = mapped_column(String(512))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now_utc
    )
