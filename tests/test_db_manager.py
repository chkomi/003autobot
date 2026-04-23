"""
DatabaseManager 단위 테스트.
인메모리 SQLite를 사용한다.
"""
import pytest
import pytest_asyncio

from database.db_manager import DatabaseManager
from database.models import TradeRecord, BotEvent


@pytest_asyncio.fixture
async def db(tmp_path):
    db = await DatabaseManager.create(tmp_path / "test.db")
    yield db
    await db.close()


def _sample_trade(**kwargs) -> TradeRecord:
    defaults = dict(
        symbol="BTC/USDT:USDT",
        direction="LONG",
        quantity=0.001,
        leverage=5,
        entry_price=50000.0,
        stop_loss=49500.0,
        take_profit_1=51000.0,
        entry_time="2026-01-01T00:00:00+00:00",
        signal_confidence=0.85,
    )
    defaults.update(kwargs)
    return TradeRecord(**defaults)


class TestInsertAndFetch:

    @pytest.mark.asyncio
    async def test_insert_trade(self, db):
        trade = _sample_trade()
        await db.insert_trade(trade)

        open_trades = await db.fetch_open_trades()
        assert len(open_trades) == 1
        assert open_trades[0].trade_id == trade.trade_id
        assert open_trades[0].symbol == "BTC/USDT:USDT"

    @pytest.mark.asyncio
    async def test_update_trade(self, db):
        trade = _sample_trade()
        await db.insert_trade(trade)

        trade.status = "CLOSED"
        trade.exit_price = 51000.0
        trade.pnl_usdt = 5.0
        trade.pnl_pct = 0.02
        trade.exit_time = "2026-01-01T06:00:00+00:00"
        trade.exit_reason = "TP1"
        await db.update_trade(trade)

        open_trades = await db.fetch_open_trades()
        assert len(open_trades) == 0

        closed = await db.fetch_closed_trades(limit=10)
        assert len(closed) == 1
        assert closed[0].pnl_usdt == 5.0

    @pytest.mark.asyncio
    async def test_multiple_trades(self, db):
        t1 = _sample_trade(direction="LONG")
        t2 = _sample_trade(direction="SHORT")
        await db.insert_trade(t1)
        await db.insert_trade(t2)

        open_trades = await db.fetch_open_trades()
        assert len(open_trades) == 2

    @pytest.mark.asyncio
    async def test_fetch_closed_empty(self, db):
        closed = await db.fetch_closed_trades(limit=10)
        assert closed == []


class TestDailyPnl:

    @pytest.mark.asyncio
    async def test_daily_pnl_no_trades(self, db):
        daily = await db.get_daily_pnl()
        assert daily.trade_count == 0
        assert daily.pnl_usdt == 0.0


class TestEvents:

    @pytest.mark.asyncio
    async def test_log_and_fetch_events(self, db):
        event = BotEvent(
            event_type="ORDER",
            level="INFO",
            message="테스트 이벤트",
        )
        await db.log_event(event)

        events = await db.fetch_recent_events(limit=10)
        assert len(events) >= 1
