"""시장/점수 관련 스키마 (plan v2 §5.1)."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ScoreBreakdown(BaseModel):
    trend: float = Field(..., ge=0, le=100)
    momentum: float = Field(..., ge=0, le=100)
    volume: float = Field(..., ge=0, le=100)
    volatility: float = Field(..., ge=0, le=100)
    sentiment: float = Field(..., ge=0, le=100)
    macro: float = Field(..., ge=0, le=100)


class SymbolScore(BaseModel):
    symbol: str
    long_score: float = Field(..., ge=0, le=100)
    short_score: float = Field(..., ge=0, le=100)
    neutral: float = Field(..., ge=0, le=100)
    direction: str  # LONG | SHORT | NEUTRAL
    signal_strength: str  # STRONG | MODERATE | NEUTRAL | ...
    breakdown: ScoreBreakdown
    calculated_at: datetime


class LeaderboardEntry(BaseModel):
    rank: int
    symbol: str
    long_score: float
    short_score: float
    direction: str
    last_price: Optional[float] = None
    change_24h: Optional[float] = None


class SymbolDetail(BaseModel):
    symbol: str
    score: SymbolScore
    rationale: list[str] = Field(
        default_factory=list,
        description="점수 산출 근거 텍스트 (plan v2 §5.1)",
    )
