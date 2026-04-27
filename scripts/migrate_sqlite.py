#!/usr/bin/env python3
"""SQLite → PostgreSQL 일회성 데이터 이관 스크립트 (plan v2 §15).

사용법:
    # 1) 환경변수 설정
    export AUTOBOT_API_DATABASE_URL=postgresql+asyncpg://autobot:autobot@localhost:5432/autobot

    # 2) PostgreSQL에 테이블 생성 (아직 안 했다면)
    alembic upgrade head

    # 3) 데이터 이관
    python scripts/migrate_sqlite.py --sqlite data_cache/trading.db

    # 4) dry-run (실제로 넣지 않고 개수만 확인)
    python scripts/migrate_sqlite.py --sqlite data_cache/trading.db --dry-run

이관 대상 테이블:
    trades, daily_pnl, equity_snapshots, bot_events, candle_cache

이관 후 확인:
    # PostgreSQL 접속 후
    SELECT count(*) FROM trades;
    SELECT count(*) FROM daily_pnl;
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────
_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(_ROOT))

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# ── 테이블별 이관 SQL ────────────────────────────────────────────

_INSERT_TRADE = """
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
) ON CONFLICT (trade_id) DO NOTHING
"""

_INSERT_DAILY_PNL = """
INSERT INTO daily_pnl (date, pnl_usdt, trade_count, win_count, peak_equity, min_equity, updated_at)
VALUES (:date, :pnl_usdt, :trade_count, :win_count, :peak_equity, :min_equity, :updated_at)
ON CONFLICT (date) DO UPDATE SET
    pnl_usdt    = EXCLUDED.pnl_usdt,
    trade_count = EXCLUDED.trade_count,
    win_count   = EXCLUDED.win_count,
    updated_at  = EXCLUDED.updated_at
"""

_INSERT_EQUITY = """
INSERT INTO equity_snapshots (equity, free, used, created_at)
VALUES (:equity, :free, :used, :created_at)
"""

_INSERT_EVENT = """
INSERT INTO bot_events (event_type, level, message, metadata, created_at)
VALUES (:event_type, :level, :message, :metadata, :created_at)
"""

_INSERT_CANDLE = """
INSERT INTO candle_cache (symbol, timeframe, timestamp, open, high, low, close, volume)
VALUES (:symbol, :timeframe, :timestamp, :open, :high, :low, :close, :volume)
ON CONFLICT ON CONSTRAINT uq_candle DO NOTHING
"""


# ── 이관 함수 ────────────────────────────────────────────────────

async def migrate(sqlite_path: Path, pg_url: str, dry_run: bool) -> None:
    """SQLite에서 PostgreSQL로 전체 테이블을 이관한다."""
    if not sqlite_path.exists():
        logger.error(f"SQLite 파일을 찾을 수 없습니다: {sqlite_path}")
        sys.exit(1)

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    engine = create_async_engine(pg_url, echo=False)

    async with engine.begin() as pg:

        # ── trades ───────────────────────────────────────────────
        rows = cur.execute("SELECT * FROM trades").fetchall()
        logger.info(f"trades: {len(rows)}행 이관 {'(dry-run)' if dry_run else ''}")
        if not dry_run:
            for row in rows:
                d = dict(row)
                d.setdefault("take_profit_2", None)
                await pg.execute(text(_INSERT_TRADE), d)

        # ── daily_pnl ────────────────────────────────────────────
        rows = cur.execute("SELECT * FROM daily_pnl").fetchall()
        logger.info(f"daily_pnl: {len(rows)}행 이관 {'(dry-run)' if dry_run else ''}")
        if not dry_run:
            for row in rows:
                await pg.execute(text(_INSERT_DAILY_PNL), dict(row))

        # ── equity_snapshots ─────────────────────────────────────
        rows = cur.execute("SELECT * FROM equity_snapshots").fetchall()
        logger.info(f"equity_snapshots: {len(rows)}행 이관 {'(dry-run)' if dry_run else ''}")
        if not dry_run:
            for row in rows:
                await pg.execute(text(_INSERT_EQUITY), dict(row))

        # ── bot_events ───────────────────────────────────────────
        rows = cur.execute("SELECT * FROM bot_events").fetchall()
        logger.info(f"bot_events: {len(rows)}행 이관 {'(dry-run)' if dry_run else ''}")
        if not dry_run:
            for row in rows:
                d = dict(row)
                # SQLite 컬럼명 metadata → event_metadata (모델 컬럼명)
                d["metadata"] = d.pop("metadata", None)
                await pg.execute(text(_INSERT_EVENT), d)

        # ── candle_cache (대용량 — 배치 처리) ────────────────────
        rows = cur.execute("SELECT * FROM candle_cache").fetchall()
        logger.info(f"candle_cache: {len(rows)}행 이관 {'(dry-run)' if dry_run else ''}")
        if not dry_run:
            BATCH = 1000
            for i in range(0, len(rows), BATCH):
                batch = [dict(r) for r in rows[i : i + BATCH]]
                for item in batch:
                    await pg.execute(text(_INSERT_CANDLE), item)
                logger.info(f"  candle_cache {i + len(batch)}/{len(rows)} 완료")

    conn.close()
    await engine.dispose()
    logger.info("✅ 이관 완료")


# ── CLI ──────────────────────────────────────────────────────────

def main() -> None:
    import os

    parser = argparse.ArgumentParser(description="SQLite → PostgreSQL 이관")
    parser.add_argument("--sqlite", default="data_cache/trading.db", help="SQLite 파일 경로")
    parser.add_argument("--pg-url", default=None, help="PostgreSQL URL (없으면 환경변수 사용)")
    parser.add_argument("--dry-run", action="store_true", help="실제 이관 없이 개수만 확인")
    args = parser.parse_args()

    pg_url = args.pg_url or os.environ.get("AUTOBOT_API_DATABASE_URL")
    if not pg_url:
        logger.error(
            "PostgreSQL URL이 필요합니다. "
            "--pg-url 옵션 또는 AUTOBOT_API_DATABASE_URL 환경변수를 설정하세요."
        )
        sys.exit(1)

    asyncio.run(migrate(Path(args.sqlite), pg_url, args.dry_run))


if __name__ == "__main__":
    main()
