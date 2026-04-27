"""
SQLite DB 관리자 (aiosqlite 기반 비동기).
모든 DB 읽기/쓰기는 이 클래스를 통한다.
"""
import json
import os
import stat
from datetime import datetime, timedelta, timezone
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


# KST = UTC+9. 일일 손실 한도는 한국 시간 자정(00:00 KST) 기준으로 리셋된다.
_KST = timezone(timedelta(hours=9))


def _today() -> str:
    """오늘 날짜를 KST(UTC+9) 기준으로 반환한다. (YYYY-MM-DD)"""
    return datetime.now(_KST).strftime("%Y-%m-%d")


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

        # 002+: executescript로 전체 SQL 실행 (CREATE IF NOT EXISTS는 idempotent,
        # ALTER 재실행은 OperationalError를 try/except로 무시)
        for mig in sorted(migrations_dir.glob("0*.sql")):
            if mig.name.startswith("001_"):
                continue
            sql_text = mig.read_text(encoding="utf-8")
            try:
                await self._conn.executescript(sql_text)
                await self._conn.commit()
            except aiosqlite.OperationalError as e:
                logger.debug(f"마이그레이션 {mig.name} 스킵 (이미 적용됨): {e}")
                # ALTER 등 일부 statement만 실패한 경우, 트랜잭션 롤백 후 statement 단위로 재시도
                try:
                    await self._conn.rollback()
                except aiosqlite.Error:
                    pass
                stmts = [s.strip() for s in sql_text.split(";") if s.strip() and not s.strip().startswith("--")]
                for stmt in stmts:
                    try:
                        await self._conn.execute(stmt)
                    except aiosqlite.OperationalError as inner_e:
                        logger.debug(f"마이그레이션 {mig.name} statement 스킵: {inner_e}")
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
        """전체 기간 거래 통계 요약 (누적)"""
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

    async def fetch_trade_stats_for_date(self, date_str: Optional[str] = None) -> dict:
        """특정 KST 날짜의 거래 통계 요약.

        date_str: 'YYYY-MM-DD' (KST 기준). None이면 오늘(KST).
        exit_time은 UTC ISO 문자열로 저장되므로, KST 날짜 → UTC 범위로 변환해 필터링한다.
        KST 자정 00:00 = UTC 전날 15:00
        """
        date_str = date_str or _today()
        # KST 날짜 → UTC 범위
        from datetime import date as _date_type
        kst_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=_KST)
        utc_start = (kst_date).astimezone(timezone.utc)
        utc_end = (kst_date + timedelta(days=1)).astimezone(timezone.utc)
        utc_start_str = utc_start.strftime("%Y-%m-%dT%H:%M:%S")
        utc_end_str = utc_end.strftime("%Y-%m-%dT%H:%M:%S")

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
        FROM trades
        WHERE status = 'CLOSED'
          AND exit_time >= ?
          AND exit_time < ?
        """
        async with self._conn.execute(sql, (utc_start_str, utc_end_str)) as cur:
            row = await cur.fetchone()
            return dict(row) if row else {}

    # ── 월간 메트릭 (CAPR 목표 추적) ────────────────────────────

    async def get_or_init_monthly_metric(
        self,
        month_str: str,
        start_equity: float,
        target_pct: float,
    ) -> dict:
        """월간 메트릭을 조회하거나, 없으면 초기화해 반환한다.

        month_str: 'YYYY-MM' (KST)
        start_equity: 월 첫날 시작 자산
        target_pct: 해당 월 목표 수익률
        """
        async with self._conn.execute(
            "SELECT * FROM monthly_metrics WHERE month = ?", (month_str,)
        ) as cur:
            row = await cur.fetchone()
            if row:
                return dict(row)

        # 신규 월 초기화
        await self._conn.execute(
            """INSERT OR IGNORE INTO monthly_metrics
               (month, start_equity, end_equity, pnl_usdt, trade_count, win_count,
                target_pct, actual_pct, goal_achieved, updated_at)
               VALUES (?, ?, ?, 0, 0, 0, ?, 0, 0, datetime('now'))""",
            (month_str, start_equity, start_equity, target_pct),
        )
        await self._conn.commit()
        return {
            "month": month_str,
            "start_equity": start_equity,
            "end_equity": start_equity,
            "pnl_usdt": 0.0,
            "trade_count": 0,
            "win_count": 0,
            "target_pct": target_pct,
            "actual_pct": 0.0,
            "goal_achieved": 0,
        }

    async def update_monthly_metric(
        self,
        month_str: str,
        end_equity: float,
        daily_pnl: float,
        daily_trades: int,
        daily_wins: int,
        daily_best: Optional[float] = None,
        daily_worst: Optional[float] = None,
    ) -> None:
        """월간 메트릭을 업데이트한다 (매일 KST 자정 호출)."""
        async with self._conn.execute(
            "SELECT * FROM monthly_metrics WHERE month = ?", (month_str,)
        ) as cur:
            row = await cur.fetchone()
            if not row:
                return
            m = dict(row)

        start_eq = m.get("start_equity") or end_equity
        actual_pct = (end_equity - start_eq) / start_eq if start_eq > 0 else 0.0
        target_pct = m.get("target_pct") or 0.0
        goal_achieved = 1 if actual_pct >= target_pct else 0

        new_best = max(filter(None, [m.get("best_day_pnl"), daily_best or daily_pnl or 0]))
        new_worst = min(filter(None, [m.get("worst_day_pnl"), daily_worst or daily_pnl or 0]))

        await self._conn.execute(
            """UPDATE monthly_metrics SET
               end_equity    = ?,
               pnl_usdt      = pnl_usdt + ?,
               trade_count   = trade_count + ?,
               win_count     = win_count + ?,
               best_day_pnl  = ?,
               worst_day_pnl = ?,
               actual_pct    = ?,
               goal_achieved = ?,
               updated_at    = datetime('now')
             WHERE month = ?""",
            (
                end_equity,
                daily_pnl,
                daily_trades,
                daily_wins,
                new_best,
                new_worst,
                actual_pct,
                goal_achieved,
                month_str,
            ),
        )
        await self._conn.commit()

    async def fetch_monthly_metrics(self, limit: int = 12) -> list[dict]:
        """최근 N개월 메트릭을 최신순으로 반환한다."""
        async with self._conn.execute(
            "SELECT * FROM monthly_metrics ORDER BY month DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def fetch_portfolio_stats(
        self,
        initial_capital: float,
        start_date: Optional[str] = None,
    ) -> dict:
        """포트폴리오 전체 통계 (누적 수익률, CAGR 추정, 승월/패월 등).

        initial_capital: .env의 GOAL_INITIAL_CAPITAL
        start_date: 'YYYY-MM-DD' — None이면 최초 equity_snapshot 날짜 사용
        """
        # 최신 자산
        async with self._conn.execute(
            "SELECT equity FROM equity_snapshots ORDER BY created_at DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
            current_equity = float(row[0]) if row else initial_capital

        # 시작일 결정
        if not start_date:
            async with self._conn.execute(
                "SELECT MIN(created_at) FROM equity_snapshots"
            ) as cur:
                row = await cur.fetchone()
                start_date = (row[0] or "")[:10] if row else _today()

        # 경과 일수
        try:
            d0 = datetime.strptime(start_date, "%Y-%m-%d")
            elapsed_days = max((datetime.now() - d0).days, 1)
        except ValueError:
            elapsed_days = 1

        cumulative_return = (current_equity - initial_capital) / initial_capital
        years = elapsed_days / 365.25
        try:
            cagr = (current_equity / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0
        except (ZeroDivisionError, ValueError):
            cagr = 0.0

        # 월간 메트릭 집계
        monthly = await self.fetch_monthly_metrics(limit=24)
        win_months = sum(1 for m in monthly if (m.get("actual_pct") or 0) > 0)
        loss_months = sum(1 for m in monthly if (m.get("actual_pct") or 0) < 0)
        achieved_months = sum(1 for m in monthly if m.get("goal_achieved") == 1)

        return {
            "initial_capital": initial_capital,
            "current_equity": current_equity,
            "cumulative_return": cumulative_return,
            "cagr": cagr,
            "elapsed_days": elapsed_days,
            "start_date": start_date,
            "win_months": win_months,
            "loss_months": loss_months,
            "achieved_months": achieved_months,
            "total_months": len(monthly),
        }

    # ── 포트폴리오 이정표 ─────────────────────────────────────────

    async def get_hit_milestones(self) -> list[float]:
        """이미 달성한 이정표 수익률 목록을 반환한다."""
        async with self._conn.execute(
            "SELECT milestone_pct FROM portfolio_milestones ORDER BY milestone_pct"
        ) as cur:
            rows = await cur.fetchall()
            return [float(r[0]) for r in rows]

    async def record_milestone(self, milestone_pct: float, equity: float) -> None:
        """새 이정표를 기록한다 (중복 방지)."""
        await self._conn.execute(
            """INSERT OR IGNORE INTO portfolio_milestones
               (milestone_pct, equity_at_hit, achieved_at)
               VALUES (?, ?, datetime('now'))""",
            (milestone_pct, equity),
        )
        await self._conn.commit()

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
