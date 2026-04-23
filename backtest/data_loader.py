"""
백테스팅용 과거 데이터 로더.
OKX REST API에서 과거 캔들을 대량 수집해 CSV/SQLite에 저장한다.
"""
import asyncio
import time
from pathlib import Path

import pandas as pd
from loguru import logger

from data.okx_rest_client import OKXRestClient


class HistoricalDataLoader:
    """과거 OHLCV 데이터 대량 수집"""

    MAX_PER_REQUEST = 300  # OKX API 최대 300개
    RATE_LIMIT_SLEEP = 0.2  # 초당 5회 이하 (OKX 제한 준수)

    def __init__(self, rest: OKXRestClient):
        self._rest = rest

    async def fetch_full_history(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """지정 기간의 전체 캔들 데이터를 수집한다.

        Args:
            start_date: "2024-01-01" 형식
            end_date:   "2024-12-31" 형식

        Returns:
            OHLCV DataFrame (DatetimeIndex)
        """
        start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
        end_ts = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)

        all_dfs = []
        current_ts = start_ts
        total_fetched = 0

        logger.info(f"과거 데이터 수집 시작: {symbol} {timeframe} {start_date}~{end_date}")

        while current_ts < end_ts:
            df = await self._rest.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                limit=self.MAX_PER_REQUEST,
                since=current_ts,
            )

            if df.empty:
                break

            # end_date 이후 제거
            df = df[df.index.tz_localize(None) <= pd.Timestamp(end_date)]
            if df.empty:
                break

            all_dfs.append(df)
            total_fetched += len(df)

            # 다음 배치 시작 시점
            last_ts = df.index[-1]
            next_ts = int(last_ts.timestamp() * 1000) + 1
            if next_ts <= current_ts:
                break
            current_ts = next_ts

            logger.debug(f"수집 중: {total_fetched}개 ({last_ts})")
            await asyncio.sleep(self.RATE_LIMIT_SLEEP)

        if not all_dfs:
            logger.warning("수집된 데이터 없음")
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        result = pd.concat(all_dfs)
        result = result[~result.index.duplicated(keep="last")].sort_index()
        logger.info(f"수집 완료: {len(result)}개 캔들")
        return result

    def save_to_csv(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        logger.info(f"CSV 저장: {path} ({len(df)}개)")

    def load_from_csv(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    async def fetch_or_load_csv(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        cache_dir: Path,
    ) -> pd.DataFrame:
        """CSV 캐시가 있으면 로드, 없으면 API에서 수집 후 저장한다.

        캐시 파일명: {symbol}_{timeframe}.csv (슬래시/콜론 → 언더스코어)
        """
        safe_name = symbol.replace("/", "_").replace(":", "_")
        cache_path = cache_dir / f"{safe_name}_{timeframe}.csv"

        if cache_path.exists():
            logger.info(f"캐시 로드: {cache_path}")
            return self.load_from_csv(cache_path)

        df = await self.fetch_full_history(symbol, timeframe, start_date, end_date)
        if not df.empty:
            self.save_to_csv(df, cache_path)
        return df
