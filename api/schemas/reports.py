"""리포트 관련 스키마 (plan v2 §5.3)."""
from datetime import date
from typing import Optional

from pydantic import BaseModel


class ReportOverview(BaseModel):
    period: str  # daily | weekly | monthly
    base_date: date
    total_pnl: float
    total_return_pct: float
    best_symbol: Optional[str] = None
    worst_symbol: Optional[str] = None
    risk_state: str  # OK | WARN | ALERT
    next_actions: list[str] = []


class OpsHealth(BaseModel):
    status: str  # ok | degraded | down
    data_collection_fresh_sec: Optional[int] = None
    score_snapshot_age_sec: Optional[int] = None
    last_backtest_finished_at: Optional[str] = None
    ws_disconnect_count_5m: Optional[int] = None
    notes: list[str] = []
