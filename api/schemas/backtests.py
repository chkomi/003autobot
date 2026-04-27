"""백테스트 관련 스키마 (plan v2 §5.2)."""
from datetime import date, datetime
from typing import Optional

from pydantic import BaseModel, Field


class BacktestRequest(BaseModel):
    symbol: str
    start_date: date
    end_date: date
    strategy_version: Optional[str] = None
    params: dict = Field(default_factory=dict)


class BacktestJob(BaseModel):
    job_id: str
    status: str  # queued | running | succeeded | failed
    symbol: str
    start_date: date
    end_date: date
    strategy_version: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    progress: float = 0.0  # 0.0 ~ 1.0


class BacktestMetrics(BaseModel):
    total_return: float
    annual_return: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    trade_count: int
    avg_holding_hours: float


class BacktestResult(BaseModel):
    job: BacktestJob
    metrics: Optional[BacktestMetrics] = None
    error: Optional[str] = None
