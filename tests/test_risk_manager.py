"""
리스크 관리자 단위 테스트.
DB는 인메모리 SQLite를 사용한다.
"""
import pytest
import pytest_asyncio

from config.settings import TradingConfig, RiskConfig
from risk.position_sizer import PositionSizer
from risk.stop_manager import StopManager
from config.strategy_params import RiskParams
from database.models import TradeRecord


class TestPositionSizer:
    @pytest.fixture
    def sizer(self):
        return PositionSizer(TradingConfig(), RiskConfig())

    def test_basic_calculation(self, sizer):
        result = sizer.calculate(
            balance_usdt=1000.0,
            entry_price=50000.0,
            stop_price=49500.0,  # 1% SL
        )
        assert result.quantity > 0
        assert result.margin_required <= 1000.0 * 0.10  # 최대 10%
        assert result.leverage == 5

    def test_min_quantity(self, sizer):
        """잔고가 매우 적어도 최소 수량 보장"""
        result = sizer.calculate(
            balance_usdt=100.0,
            entry_price=50000.0,
            stop_price=49000.0,
        )
        assert result.quantity >= 0.001

    def test_max_position_cap(self, sizer):
        """포지션이 max_position_pct를 넘지 않아야 함"""
        result = sizer.calculate(
            balance_usdt=10000.0,
            entry_price=50000.0,
            stop_price=49990.0,  # 매우 작은 SL → 큰 포지션 계산됨
        )
        max_margin = 10000.0 * 0.10
        assert result.margin_required <= max_margin * 1.01  # 1% 오차 허용


class TestStopManager:
    @pytest.fixture
    def stop_mgr(self):
        return StopManager(RiskParams())

    def _make_long_trade(self, entry=50000.0, sl=49000.0, tp1=52000.0) -> TradeRecord:
        t = TradeRecord(symbol="BTC/USDT:USDT", direction="LONG", quantity=0.01, leverage=5)
        t.entry_price = entry
        t.stop_loss = sl
        t.take_profit_1 = tp1
        return t

    def _make_short_trade(self, entry=50000.0, sl=51000.0, tp1=48000.0) -> TradeRecord:
        t = TradeRecord(symbol="BTC/USDT:USDT", direction="SHORT", quantity=0.01, leverage=5)
        t.entry_price = entry
        t.stop_loss = sl
        t.take_profit_1 = tp1
        return t

    def test_sl_hit_long(self, stop_mgr):
        trade = self._make_long_trade()
        assert stop_mgr.is_stop_hit(trade, 48999.0)
        assert not stop_mgr.is_stop_hit(trade, 49001.0)

    def test_sl_hit_short(self, stop_mgr):
        trade = self._make_short_trade()
        assert stop_mgr.is_stop_hit(trade, 51001.0)
        assert not stop_mgr.is_stop_hit(trade, 50999.0)

    def test_tp1_hit_long(self, stop_mgr):
        trade = self._make_long_trade()
        assert stop_mgr.is_tp1_hit(trade, 52000.0)
        assert not stop_mgr.is_tp1_hit(trade, 51999.0)

    def test_tp1_hit_short(self, stop_mgr):
        trade = self._make_short_trade()
        assert stop_mgr.is_tp1_hit(trade, 48000.0)
        assert not stop_mgr.is_tp1_hit(trade, 48001.0)

    def test_trailing_stop_updates_long(self, stop_mgr):
        """TP1 이후 가격 상승 시 트레일링 스탑 이동 (ATR × trailing_atr_multiplier)"""
        trade = self._make_long_trade()
        # 가격이 TP1 이상으로 올라감
        update = stop_mgr.check_trailing_stop(trade, current_price=53000.0, atr_value=1000.0)
        assert update.should_update
        expected = 53000.0 - 1000.0 * stop_mgr._p.trailing_atr_multiplier
        assert update.new_stop_price == expected

    def test_trailing_stop_updates_even_if_price_below_tp1(self, stop_mgr):
        """호출자가 TP1 이후라고 판단해 호출하면, 가격이 TP1 아래로 되돌려와도 트레일링은 갱신."""
        trade = self._make_long_trade(entry=50000.0, sl=49000.0, tp1=52000.0)
        # 한 번 TP1 위에서 트레일링으로 SL을 51,000(가정)까지 올렸다고 치고
        trade.stop_loss = 51000.0
        # 가격이 51,600으로 내려와도, 현재가-ATR*trail > old_stop이면 업데이트
        mult = stop_mgr._p.trailing_atr_multiplier
        current = 51000.0 + 1000.0 * mult + 200  # old_stop 위로 200 + ATR*mult
        update = stop_mgr.check_trailing_stop(trade, current_price=current, atr_value=1000.0)
        assert update.should_update

    def test_trailing_stop_min_step(self, stop_mgr):
        """변경폭이 ATR × min_trail_step_ratio 미만이면 재배치하지 않는다."""
        trade = self._make_long_trade()
        trade.stop_loss = 51000.0
        mult = stop_mgr._p.trailing_atr_multiplier
        min_step = 1000.0 * stop_mgr._p.min_trail_step_ratio
        # new_stop 이 old_stop 보다 min_step 이하로만 올라가는 current_price
        current = trade.stop_loss + 1000.0 * mult + (min_step * 0.5)
        update = stop_mgr.check_trailing_stop(trade, current_price=current, atr_value=1000.0)
        assert not update.should_update

    def test_propose_adaptive_sl_widens_when_atr_expands(self, stop_mgr):
        trade = self._make_long_trade(entry=50000.0, sl=49000.0)
        trade.atr_at_entry = 500.0
        update = stop_mgr.propose_adaptive_sl(trade, current_atr=1000.0)
        assert update.should_update
        # ATR 2배 확대 → SL 거리도 확대되어 stop 가격은 더 낮아짐
        assert update.new_stop_price < 49000.0

    def test_propose_adaptive_sl_never_crosses_breakeven(self, stop_mgr):
        trade = self._make_long_trade(entry=50000.0, sl=49000.0)
        trade.atr_at_entry = 2000.0  # 진입 시 ATR이 매우 컸음
        # 지금 ATR이 극단 축소 → SL을 좁히려 하지만 본절 이상으로는 안 올라감
        update = stop_mgr.propose_adaptive_sl(trade, current_atr=200.0)
        if update.should_update:
            assert update.new_stop_price < trade.entry_price

    def test_unrealized_pnl_long(self, stop_mgr):
        trade = self._make_long_trade(entry=50000.0)
        pnl = stop_mgr.calc_unrealized_pnl(trade, 51000.0)
        # (51000 - 50000) * 0.01 * 5 = 50.0 USDT
        assert abs(pnl - 50.0) < 0.01

    def test_unrealized_pnl_short(self, stop_mgr):
        trade = self._make_short_trade(entry=50000.0)
        pnl = stop_mgr.calc_unrealized_pnl(trade, 49000.0)
        # (50000 - 49000) * 0.01 * 5 = 50.0 USDT
        assert abs(pnl - 50.0) < 0.01
