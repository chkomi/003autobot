"""
리스크 관리자 — 진입 전 안전 게이트.
모든 거래는 이 클래스의 pre_trade_check를 통과해야 한다.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from loguru import logger

from config.settings import TradingConfig, RiskConfig
from core.exceptions import (
    DailyLossLimitError,
    InsufficientBalanceError,
    MaxDrawdownError,
    MaxPositionsError,
)
from database.db_manager import DatabaseManager
from database.models import EquitySnapshot, TradeRecord
from risk.position_sizer import PositionSize, PositionSizer
from strategy.signal_aggregator import SignalResult


class RiskCheckResult(Enum):
    APPROVED = "APPROVED"
    REJECTED_DAILY_LOSS = "REJECTED_DAILY_LOSS"
    REJECTED_MAX_DRAWDOWN = "REJECTED_MAX_DRAWDOWN"
    REJECTED_MAX_POSITIONS = "REJECTED_MAX_POSITIONS"
    REJECTED_INSUFFICIENT_BALANCE = "REJECTED_INSUFFICIENT_BALANCE"
    REJECTED_INVALID_SIGNAL = "REJECTED_INVALID_SIGNAL"
    REJECTED_COOLDOWN = "REJECTED_COOLDOWN"


@dataclass
class RiskCheckReport:
    result: RiskCheckResult
    reason: str
    position_size: Optional[PositionSize] = None


class RiskManager:
    """사전 리스크 검증 및 포지션 사이즈 결정"""

    def __init__(
        self,
        trading_cfg: TradingConfig,
        risk_cfg: RiskConfig,
        db: DatabaseManager,
    ):
        self._t = trading_cfg
        self._r = risk_cfg
        self._db = db
        self._sizer = PositionSizer(trading_cfg, risk_cfg)
        self._peak_equity: float = 0.0  # 인메모리 고점 추적

    async def pre_trade_check(
        self,
        signal: SignalResult,
        balance: float,
        open_positions: list[TradeRecord],
    ) -> RiskCheckReport:
        """진입 전 리스크 검증. 통과 시 포지션 사이즈도 계산해 반환한다."""

        # 1. 시그널 유효성
        if not signal.is_actionable:
            return RiskCheckReport(
                result=RiskCheckResult.REJECTED_INVALID_SIGNAL,
                reason="시그널이 없음 (FLAT)",
            )

        # 2. 최대 동시 포지션 수
        open_count = len(open_positions)
        if open_count >= self._t.max_open_positions:
            return RiskCheckReport(
                result=RiskCheckResult.REJECTED_MAX_POSITIONS,
                reason=f"최대 포지션 초과: {open_count}/{self._t.max_open_positions}",
            )

        # 같은 방향 이미 포지션이 있으면 진입하지 않음 (단일 방향 전략)
        same_dir = [p for p in open_positions if p.direction == signal.direction]
        if same_dir:
            return RiskCheckReport(
                result=RiskCheckResult.REJECTED_MAX_POSITIONS,
                reason=f"{signal.direction} 방향 포지션이 이미 열려 있음",
            )

        # 3. 일일 손실 한도
        daily = await self._db.get_daily_pnl()
        daily_loss_limit = balance * self._r.daily_loss_limit_pct
        if daily.pnl_usdt < -daily_loss_limit:
            return RiskCheckReport(
                result=RiskCheckResult.REJECTED_DAILY_LOSS,
                reason=f"일일 손실 한도 초과: {daily.pnl_usdt:.2f} USDT (한도: -{daily_loss_limit:.2f})",
            )

        # 4. 최대 드로다운
        peak = await self._get_peak_equity(balance)
        if peak > 0:
            drawdown = (peak - balance) / peak
            if drawdown >= self._r.max_drawdown_pct:
                return RiskCheckReport(
                    result=RiskCheckResult.REJECTED_MAX_DRAWDOWN,
                    reason=f"최대 드로다운 초과: {drawdown:.2%} (한도: {self._r.max_drawdown_pct:.2%})",
                )

        # 4.5 연속 손실 쿨다운
        consecutive_losses = await self._db.fetch_consecutive_losses()
        if consecutive_losses >= 5:
            return RiskCheckReport(
                result=RiskCheckResult.REJECTED_DAILY_LOSS,
                reason=f"연속 {consecutive_losses}패 — 24시간 쿨다운 필요",
            )
        # 3연패 이상: 최소 confidence 강화 (0.7 이상만 허용)
        if consecutive_losses >= 3 and signal.confidence < 0.7:
            return RiskCheckReport(
                result=RiskCheckResult.REJECTED_INVALID_SIGNAL,
                reason=f"연속 {consecutive_losses}패 — 신뢰도 {signal.confidence:.2f} < 0.7 차단",
            )

        # 드로다운 비율 계산 (포지션 사이징에 사용)
        current_drawdown = 0.0
        if peak > 0 and peak > balance:
            current_drawdown = (peak - balance) / peak

        # 5. 포지션 사이즈 계산
        try:
            # 과거 성과 데이터 조회
            stats = await self._db.fetch_trade_stats()
            win_rate = None
            avg_win_r = None
            avg_loss_r = None

            if stats.get("total", 0) >= 10:  # 최소 10회 거래 후 Kelly 적용
                total = stats["total"]
                wins = stats.get("wins", 0) or 0
                win_rate = wins / total if total > 0 else 0.5
                # pnl_pct는 이미 비율값(예: 0.02 = 2%)이므로 entry_price로 나눌 필요 없음
                avg_win_r = abs(stats.get("avg_win_pct", 0.02) or 0.02)
                avg_loss_r = abs(stats.get("avg_loss_pct", 0.01) or 0.01)

            pos_size = self._sizer.calculate(
                symbol=signal.symbol,
                balance_usdt=balance,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                win_rate=win_rate,
                avg_win_ratio=avg_win_r,
                avg_loss_ratio=avg_loss_r,
                drawdown_pct=current_drawdown,
            )
        except InsufficientBalanceError as e:
            return RiskCheckReport(
                result=RiskCheckResult.REJECTED_INSUFFICIENT_BALANCE,
                reason=str(e),
            )

        logger.info(
            f"리스크 검증 PASS: {signal.direction} | "
            f"수량: {pos_size.quantity} {signal.symbol} | "
            f"증거금: ${pos_size.margin_required:.2f}"
        )
        return RiskCheckReport(
            result=RiskCheckResult.APPROVED,
            reason="통과",
            position_size=pos_size,
        )

    async def record_equity(self, balance: float, free: float, used: float) -> None:
        """자산 스냅샷을 기록하고 고점을 갱신한다."""
        snap = EquitySnapshot(equity=balance, free=free, used=used)
        await self._db.insert_equity_snapshot(snap)
        if balance > self._peak_equity:
            self._peak_equity = balance

    async def check_max_drawdown(self, current_equity: float) -> bool:
        """최대 드로다운 초과 여부. True이면 봇을 정지해야 함."""
        peak = await self._get_peak_equity(current_equity)
        if peak == 0:
            return False
        drawdown = (peak - current_equity) / peak
        if drawdown >= self._r.max_drawdown_pct:
            logger.critical(
                f"최대 드로다운 초과: {drawdown:.2%} (한도: {self._r.max_drawdown_pct:.2%})"
            )
            return True
        return False

    async def _get_peak_equity(self, current: float) -> float:
        """고점 자산 반환 (인메모리 캐시 우선, 없으면 DB 조회)"""
        if self._peak_equity == 0:
            self._peak_equity = await self._db.fetch_peak_equity()
        return max(self._peak_equity, current)
