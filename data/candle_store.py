"""
캔들 데이터 캐시.
메모리에 최근 캔들을 유지하고, SQLite에 영속화한다.
재시작 시 DB에서 복원하여 API 호출을 최소화한다.
"""
from collections import defaultdict
from typing import Optional

import pandas as pd
from loguru import logger

from database.db_manager import DatabaseManager


class CandleStore:
    """타임프레임별 OHLCV 캔들 인메모리 캐시"""

    def __init__(self, db: DatabaseManager, max_candles: int = 500):
        self._db = db
        self._max_candles = max_candles
        # {(symbol, timeframe): DataFrame}
        self._cache: dict[tuple[str, str], pd.DataFrame] = defaultdict(
            lambda: pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        )

    def update(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        """캔들 데이터를 캐시에 병합 (새 데이터 우선)"""
        key = (symbol, timeframe)
        existing = self._cache[key]

        if existing.empty:
            self._cache[key] = df.tail(self._max_candles)
        else:
            merged = pd.concat([existing, df])
            merged = merged[~merged.index.duplicated(keep="last")]
            merged = merged.sort_index().tail(self._max_candles)
            self._cache[key] = merged

    def get(self, symbol: str, timeframe: str, limit: Optional[int] = None) -> pd.DataFrame:
        """캐시에서 캔들 반환. limit이 지정되면 최근 N개만 반환."""
        key = (symbol, timeframe)
        df = self._cache[key]
        if limit:
            return df.tail(limit)
        return df.copy()

    def count(self, symbol: str, timeframe: str) -> int:
        return len(self._cache[(symbol, timeframe)])

    def has_enough(self, symbol: str, timeframe: str, min_candles: int) -> bool:
        return self.count(symbol, timeframe) >= min_candles

    async def persist_to_db(self, symbol: str, timeframe: str) -> None:
        """캐시를 SQLite에 저장 (재시작 시 복원용)"""
        df = self.get(symbol, timeframe)
        if df.empty:
            return
        # batch upsert
        rows = [
            (symbol, timeframe, str(ts), row["open"], row["high"], row["low"], row["close"], row["volume"])
            for ts, row in df.iterrows()
        ]
        sql = """
        INSERT OR REPLACE INTO candle_cache (symbol, timeframe, timestamp, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        await self._db._conn.executemany(sql, rows)
        await self._db._conn.commit()

    async def load_from_db(self, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        """SQLite에서 캔들 복원"""
        async with self._db._conn.execute(
            """SELECT timestamp, open, high, low, close, volume
               FROM candle_cache
               WHERE symbol=? AND timeframe=?
               ORDER BY timestamp DESC LIMIT ?""",
            (symbol, timeframe, limit),
        ) as cur:
            rows = await cur.fetchall()

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(
            [dict(r) for r in rows],
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        return df
