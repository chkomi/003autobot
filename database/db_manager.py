"""
SQLite DB 관리자 (aiosqlite 기반 비동기).
모든 DB 읽기/쓰기는 이 클래스를 통한다.
"""
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from loguru import logger

from core.exceptions import DatabaseError
from database.models import (
    BotEvent,
    DailyPnL,
    EquitySnapshot,
    TradeRecord,
)


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class DatabaseManager:
    def __init__(self, db_path: Path):
        self._db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    @classmethod
    async def create(cls, db_path: Path) -> "DatabaseManager":
        """DB 연결 및 마이그레이션 실행. 앱 시작 시 한 번 호출."""
        db_path.parent.mkdir(parents=True, exist_ok=True)
        manager = cls(db_path)
        await manager._connect()
        await manager._migrate()
        cls._secure_file(db_path)
        logger.info(f"DB 초기화 완료: {db_path}")
        return manager

    @staticmethod
    def _secure_file(path: Path) -> None:
        """민감 파일의 권한을 소유자 전용(0o600)으로 설정한다."""
        try:
            if path.exists():
                path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except OSError as e:
            logger.warning(f"파일 권한 설정 실패 ({path}): {e}")

    async def _connect(self) -> None:
        self._conn = await aiosqlite.connect(str(self._db_path))
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")

    async def _migrate(self) -> None:
        migrations_dir = Path(__file__).parent / "migrations"
        # 001: 기본 스키마 (idempotent, CREATE IF NOT EXISTS)
        sql = (migrations_dir / "001_initial_schema.sql").read_text(encoding="utf-8")
        await self._conn.executescript(sql)
        await self._conn.commit()

        # 002+: ALTER 계열은 실패 시 무시 (이미 적용된 컬럼)
        for mig in sorted(migrations_dir.glob("0*.sql")):
            if mig.name.startswith("001_"):
                continue
            stmts = [s.strip() for s in mig.read_text(encoding="utf-8").split(";") if s.strip() and not s.strip().startswith("--")]
            for stmt in stmts:
                try:
                    await self._conn.execute(stmt)
                except aiosqlite.OperationalError as e:
                    logger.debug(f"마이그레이션 {mig.name} 스킵: {e}")
            await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("DB 연결 종료")

    # ── Trade CRUD ──────────────────────────────────────────────

    async def insert_trade(self, trade: TradeRecord) -> None:
        sql = """
        INSERT INTO trades (
            trade_id, symbol, direction, status,
            entry_price, exit_price, quantity, leverage,
            stop_loss, take_profit_1, take_profit_2,
            pnl_usdt, pnl_pct,
            entry_time, exit_time, exit_reason,
            signal_confidence, atr_at_entry,
            sl_algo_id, tp_algo_id, created_at
        ) VALUES (
            :trade_id, :symbol, :direction, :status,
            :entry_price, :exit_price, :quantity, :leverage,
            :stop_loss, :take_profit_1, :take_profit_2,
            :pnl_usdt, :pnl_pct,
            :entry_time, :exit_time, :exit_reason,
            :signal_confidence, :atr_at_entry,
            :sl_algo_id, :tp_algo_id, :created_at
        )
        """
        try:
            await self._conn.execute(sql, trade.to_dict())
            await self._conn.commit()
        except aiosqlite.Error as e:
            raise DatabaseError(f"거래 기록 실패: {e}") from e

    async def update_trade(self, trade: TradeRecord) -> None:
        sql = """
        UPDATE trades SET
            status         = :status,
            exit_price     = :exit_price,
            pnl_usdt       = :pnl_usdt,
            pnl_pct        = :pnl_pct,
            exit_time      = :exit_time,
            exit_reason    = :exit_reason,
            stop_loss      = :stop_loss,
            take_profit_1  = :take_profit_1,
            take_profit_2  = :take_profit_2,
            quantity       = :quantity,
            sl_algo_id     = :sl_algo_id,
            tp_algo_id     = :tp_algo_id
        WHERE trade_id = :trade_id
        """
        try:
            await self._conn.execute(sql, trade.to_dict())
            await self._conn.commit()
        except aiosqlite.Error as e:
            raise DatabaseError(f"거래 업데이트 실패: {e}") from e

    async def fetch_open_trades(self) -> list[TradeRecord]:
        async with self._conn.execute(
            "SELECT * FROM trades WHERE status = 'OPEN' ORDER BY entry_time"
        ) as cur:
            rows = await cur.fetchall()
            return [TradeRecord.from_row(dict(r)) for r in rows]

    async def fetch_trade_by_id(self, trade_id: str) -> Optional[TradeRecord]:
        async with self._conn.execute(
            "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
        ) as cur:
            row = await cur.fetchone()
            return TradeRecord.from_row(dict(row)) if row else None

    async def fetch_closed_trades(self, limit: int = 100) -> list[TradeRecord]:
        async with self._conn.execute(
            "SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY exit_time DESC LIMIT ?",
            (limit,),
        ) as cur:
            rows = await cur.fetchall()
            return [TradeRecord.from_row(dict(r)) for r in rows]

    # ── Daily P&L ───────────────────────────────────────────────

    async def get_daily_pnl(self, date: Optional[str] = None) -> DailyPnL:
        date = date or _today()
        async with self._conn.execute(
            "SELECT * FROM daily_pnl WHERE date = ?", (date,)
        ) as cur:
            row = await cur.fetchone()
            if row:
                d = dict(row)
                return DailyPnL(
                    date=d["date"],
                    pnl_usdt=d["pnl_usdt"] or 0,
                    trade_count=d["trade_count"] or 0,
                    win_count=d["win_count"] or 0,
                    peak_equity=d.get("peak_equity"),
                    min_equity=d.get("min_equity"),
                    updated_at=d.get("updated_at", ""),
                )
        return DailyPnL(date=date)

    async def update_daily_pnl(
        self,
        pnl_delta: float,
        is_win: bool,
        equity: Optional[float] = None,
        date: Optional[str] = None,
    ) -> None:
        date = date or _today()
        current = await self.get_daily_pnl(date)

        new_pnl = current.pnl_usdt + pnl_delta
        new_count = current.trade_count + 1
        new_wins = current.win_count + (1 if is_win else 0)

        if equity is not None:
            peak = max(current.peak_equity or equity, equity)
            low = min(current.min_equity or equity, equity)
        else:
            peak = current.peak_equity
            low = current.min_equity

        sql = """
        INSERT INTO daily_pnl (date, pnl_usdt, trade_count, win_count, peak_equity, min_equity, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(date) DO UPDATE SET
            pnl_usdt    = excluded.pnl_usdt,
            trade_count = excluded.trade_count,
            win_count   = excluded.win_count,
            peak_equity = excluded.peak_equity,
            min_equity  = excluded.min_equity,
            updated_at  = excluded.updated_at
        """
        await self._conn.execute(sql, (date, new_pnl, new_count, new_wins, peak, low))
        await self._conn.commit()

    # ── Equity Snapshot ─────────────────────────────────────────

    async def insert_equity_snapshot(self, snap: EquitySnapshot) -> None:
        await self._conn.execute(
            "INSERT INTO equity_snapshots (equity, free, used, created_at) VALUES (?, ?, ?, ?)",
            (snap.equity, snap.free, snap.used, snap.created_at),
        )
        await self._conn.commit()

    async def fetch_peak_equity(self) -> float:
        """전체 기간 최고 자산 (드로다운 계산 기준)"""
        async with self._conn.execute(
            "SELECT MAX(equity) FROM equity_snapshots"
        ) as cur:
            row = await cur.fetchone()
            return float(row[0]) if row and row[0] else 0.0

    # ── Bot Events ──────────────────────────────────────────────

    async def log_event(self, event: BotEvent) -> None:
        await self._conn.execute(
            """INSERT INTO bot_events (event_type, level, message, metadata, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                event.event_type,
                event.level,
                event.message,
                event.metadata_json(),
                event.created_at,
            ),
        )
        await self._conn.commit()

    async def fetch_recent_events(self, limit: int = 50) -> list[dict]:
        async with self._conn.execute(
            "SELECT * FROM bot_events ORDER BY created_at DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    # ── 통계 조회 ────────────────────────────────────────────────

    async def fetch_trade_stats(self) -> dict:
        """전체 거래 통계 요약"""
        sql = """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS wins,
            SUM(CASE WHEN pnl_usdt <= 0 THEN 1 ELSE 0 END) AS losses,
            SUM(pnl_usdt) AS total_pnl,
            AVG(CASE WHEN pnl_usdt > 0 THEN pnl_usdt END) AS avg_win,
            AVG(CASE WHEN pnl_usdt <= 0 THEN pnl_usdt END) AS avg_loss,
            AVG(CASE WHEN pnl_pct > 0 THEN pnl_pct END) AS avg_win_pct,
            AVG(CASE WHEN pnl_pct <= 0 THEN pnl_pct END) AS avg_loss_pct,
            MAX(pnl_usdt) AS best_trade,
            MIN(pnl_usdt) AS worst_trade
        FROM trades WHERE status = 'CLOSED'
        """
        async with self._conn.execute(sql) as cur:
            row = await cur.fetchone()
            return dict(row) if row else {}

    async def fetch_consecutive_losses(self) -> int:
        """최근 연속 손실 횟수를 반환한다."""
        sql = """
        SELECT pnl_usdt FROM trades
        WHERE status = 'CLOSED'
        ORDER BY exit_time DESC
        LIMIT 20
        """
        async with self._conn.execute(sql) as cur:
            rows = await cur.fetchall()
            count = 0
            for row in rows:
                if dict(row)["pnl_usdt"] is not None and dict(row)["pnl_usdt"] <= 0:
                    count += 1
                else:
                    break
            return count
