"""
스탑 관리자.
트레일링 스탑 로직과 TP1 도달 후 포지션 관리를 담당한다.
"""
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from config.strategy_params import RiskParams
from database.models import TradeRecord


@dataclass
class StopUpdate:
    """스탑 업데이트 결과"""
    should_update: bool
    new_stop_price: float
    reason: str = ""


class StopManager:
    """트레일링 스탑 및 TP 레벨 관리"""

    def __init__(self, params: RiskParams):
        self._p = params

    def check_trailing_stop(
        self,
        trade: TradeRecord,
        current_price: float,
        atr_value: float,
    ) -> StopUpdate:
        """트레일링 스탑을 계산한다.

        호출자(trade_lifecycle)가 TP1 도달 후에만 호출한다는 가정 하에 동작한다.
        따라서 가격이 TP1 레벨을 되돌려와도 트레일링을 계속 갱신한다.

        조건:
        - 롱: 새 SL = 현재가 - trailing_atr_multiplier × ATR. 기존 SL보다 높고, 변경폭이
              min_trail_step_ratio × ATR 이상일 때만 업데이트.
        - 숏: 새 SL = 현재가 + trailing_atr_multiplier × ATR. 기존 SL보다 낮고, 변경폭이
              min_trail_step_ratio × ATR 이상일 때만 업데이트.
        """
        if trade.stop_loss is None or trade.entry_price is None:
            return StopUpdate(should_update=False, new_stop_price=0.0)

        trail_distance = atr_value * self._p.trailing_atr_multiplier
        min_step = atr_value * getattr(self._p, "min_trail_step_ratio", 0.15)

        if trade.direction == "LONG":
            new_stop = current_price - trail_distance
            if new_stop > trade.stop_loss and (new_stop - trade.stop_loss) >= min_step:
                logger.debug(
                    f"트레일링 스탑 업데이트 (LONG): ${trade.stop_loss:.2f} → ${new_stop:.2f}"
                )
                return StopUpdate(
                    should_update=True,
                    new_stop_price=new_stop,
                    reason=f"트레일링: 현재가 ${current_price:.2f} - ATR ${trail_distance:.2f}",
                )

        elif trade.direction == "SHORT":
            new_stop = current_price + trail_distance
            if new_stop < trade.stop_loss and (trade.stop_loss - new_stop) >= min_step:
                logger.debug(
                    f"트레일링 스탑 업데이트 (SHORT): ${trade.stop_loss:.2f} → ${new_stop:.2f}"
                )
                return StopUpdate(
                    should_update=True,
                    new_stop_price=new_stop,
                    reason=f"트레일링: 현재가 ${current_price:.2f} + ATR ${trail_distance:.2f}",
                )

        return StopUpdate(should_update=False, new_stop_price=trade.stop_loss or 0.0)

    def is_stop_hit(self, trade: TradeRecord, current_price: float) -> bool:
        """현재 가격이 손절 가격에 도달했는지 확인."""
        if trade.stop_loss is None:
            return False
        if trade.direction == "LONG":
            return current_price <= trade.stop_loss
        return current_price >= trade.stop_loss

    def is_tp1_hit(self, trade: TradeRecord, current_price: float) -> bool:
        """현재 가격이 TP1에 도달했는지 확인."""
        if trade.take_profit_1 is None:
            return False
        if trade.direction == "LONG":
            return current_price >= trade.take_profit_1
        return current_price <= trade.take_profit_1

    def is_tp2_hit(self, trade: TradeRecord, current_price: float) -> bool:
        """현재 가격이 TP2에 도달했는지 확인."""
        if trade.take_profit_2 is None:
            return False
        if trade.direction == "LONG":
            return current_price >= trade.take_profit_2
        return current_price <= trade.take_profit_2

    def is_time_limit_hit(self, trade: TradeRecord, max_hours: int) -> bool:
        """포지션 최대 보유 시간 초과 여부 확인."""
        if trade.entry_time is None:
            return False
        from datetime import datetime, timezone
        try:
            entry = datetime.fromisoformat(trade.entry_time)
            if entry.tzinfo is None:
                entry = entry.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            elapsed = (now - entry).total_seconds() / 3600
            return elapsed >= max_hours
        except (ValueError, TypeError) as e:
            logger.debug(f"시간 제한 파싱 실패: {e}")
            return False

    def calc_unrealized_pnl(self, trade: TradeRecord, current_price: float) -> float:
        """미실현 손익 (USDT) 계산."""
        if trade.entry_price is None:
            return 0.0
        if trade.direction == "LONG":
            return (current_price - trade.entry_price) * trade.quantity * trade.leverage
        return (trade.entry_price - current_price) * trade.quantity * trade.leverage

    def propose_adaptive_sl(
        self,
        trade: TradeRecord,
        current_atr: float,
    ) -> StopUpdate:
        """TP1 도달 전 ATR 변화에 따라 SL 거리를 조정해 새 SL을 제안한다.

        진입 시점 ATR 대비 현재 ATR이 크게 커지면 SL을 넓혀 노이즈 피격을 줄이고,
        크게 작아지면 SL을 좁혀 이익을 보호한다. 본절 이상으로는 넘어가지 않는다
        (TP1 전 영역은 트레일링 대상이 아님 — 진입가를 넘어선 이동은 하지 않는다).
        """
        if (
            trade.stop_loss is None
            or trade.entry_price is None
            or trade.atr_at_entry is None
            or trade.atr_at_entry <= 0
        ):
            return StopUpdate(should_update=False, new_stop_price=trade.stop_loss or 0.0)

        base_distance = abs(trade.entry_price - trade.stop_loss)
        if base_distance <= 0:
            return StopUpdate(should_update=False, new_stop_price=trade.stop_loss)

        adjusted_distance = self.adaptive_stop_distance(
            entry_atr=trade.atr_at_entry,
            current_atr=current_atr,
            base_sl_distance=base_distance,
        )
        if trade.direction == "LONG":
            new_stop = trade.entry_price - adjusted_distance
            # 본절 이상으로 끌어올리지 않는다 (TP1 전이기 때문)
            new_stop = min(new_stop, trade.entry_price * 0.999)
        else:
            new_stop = trade.entry_price + adjusted_distance
            new_stop = max(new_stop, trade.entry_price * 1.001)

        min_step = current_atr * getattr(self._p, "min_trail_step_ratio", 0.15)
        if abs(new_stop - trade.stop_loss) < min_step:
            return StopUpdate(should_update=False, new_stop_price=trade.stop_loss)

        return StopUpdate(
            should_update=True,
            new_stop_price=new_stop,
            reason=f"적응형 SL: ATR {trade.atr_at_entry:.2f}→{current_atr:.2f}",
        )

    def adaptive_stop_distance(
        self,
        entry_atr: float,
        current_atr: float,
        base_sl_distance: float,
    ) -> float:
        """변동성 변화에 따라 SL 거리를 조정한다.

        ATR이 진입 시점 대비 확대되면 SL을 넓히고,
        축소되면 트레일링을 강화한다.

        Args:
            entry_atr: 진입 시점의 ATR
            current_atr: 현재 ATR
            base_sl_distance: 기본 SL 거리 (진입가 기준)

        Returns:
            조정된 SL 거리
        """
        if entry_atr <= 0:
            return base_sl_distance

        ratio = current_atr / entry_atr

        if ratio > 1.5:
            # ATR 50% 이상 확대: SL을 비례 확대 (최대 2배)
            scale = min(ratio, 2.0)
            adjusted = base_sl_distance * scale
            logger.debug(f"적응형 SL: ATR 확대 {ratio:.2f}x → SL {scale:.2f}x 확대")
        elif ratio < 0.7:
            # ATR 30% 이상 축소: SL을 좁혀 이익 보호
            scale = max(ratio, 0.5)
            adjusted = base_sl_distance * scale
            logger.debug(f"적응형 SL: ATR 축소 {ratio:.2f}x → SL {scale:.2f}x 축소")
        else:
            adjusted = base_sl_distance

        return adjusted
