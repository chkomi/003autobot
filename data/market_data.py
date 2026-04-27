"""
통합 시장 데이터 인터페이스.
REST와 캔들 캐시를 조합해 전략 모듈에 단일 진입점을 제공한다.
멀티 심볼을 지원하며, 각 심볼별로 캔들 데이터를 관리한다.
"""
import asyncio
from typing import List, Optional

import pandas as pd
from loguru import logger

from config.settings import TradingConfig
from core.exceptions import InsufficientDataError
from data.candle_store import CandleStore
from data.okx_rest_client import OKXRestClient


class MarketData:
    """전략 레이어에서 사용하는 통합 데이터 접근 객체 (멀티 심볼 지원)"""

    # 각 타임프레임별 지표 계산에 필요한 최소 캔들 수
    MIN_CANDLES = {
        "4h": 220,   # EMA200 + 버퍼
        "1h": 60,    # MACD(26) + 버퍼
        "15m": 30,   # BB(20) + 버퍼
    }

    def __init__(
        self,
        rest: OKXRestClient,
        store: CandleStore,
        config: TradingConfig,
    ):
        self._rest = rest
        self._store = store
        self._config = config
        self._symbols = config.symbol_list

    @property
    def symbols(self) -> List[str]:
        return self._symbols

    async def warm_up(self) -> None:
        """봇 시작 시 모든 심볼 × 타임프레임 캔들을 로딩한다."""
        for symbol in self._symbols:
            for tf in self._config.all_timeframes:
                await self.refresh_candles(tf, symbol=symbol)
                count = self._store.count(symbol, tf)
                logger.info(f"캔들 로딩 완료: {symbol} {tf} — {count}개")

    async def refresh_candles(self, timeframe: str, symbol: Optional[str] = None) -> None:
        """REST API로 최신 캔들을 가져와 캐시를 업데이트한다."""
        if symbol is None:
            # 모든 심볼에 대해 갱신
            for sym in self._symbols:
                await self._refresh_single(sym, timeframe)
        else:
            await self._refresh_single(symbol, timeframe)

    async def _refresh_single(self, symbol: str, timeframe: str) -> None:
        """단일 심볼의 캔들 갱신"""
        df = await self._rest.fetch_ohlcv(
            symbol,
            timeframe,
            limit=self._config.candle_fetch_lookback,
        )
        if not df.empty:
            self._store.update(symbol, timeframe, df)

    def get_candles(self, timeframe: str, limit: Optional[int] = None, symbol: Optional[str] = None) -> pd.DataFrame:
        """캐시에서 캔들을 반환한다. 데이터 부족 시 예외."""
        sym = symbol or self._symbols[0]
        min_required = self.MIN_CANDLES.get(timeframe, 30)
        available = self._store.count(sym, timeframe)

        if available < min_required:
            raise InsufficientDataError(min_required, available, timeframe)

        return self._store.get(sym, timeframe, limit=limit)

    async def get_current_price(self, symbol: Optional[str] = None) -> float:
        """현재 최종 체결가"""
        sym = symbol or self._symbols[0]
        ticker = await self._rest.fetch_ticker(sym)
        return float(ticker["last"])

    async def get_balance(self) -> dict:
        """USDT 잔고 반환"""
        return await self._rest.fetch_balance()

    async def get_funding_rate(self, symbol: Optional[str] = None) -> Optional[float]:
        """현재 펀딩비 반환 (실패 시 None)"""
        sym = symbol or self._symbols[0]
        return await self._rest.fetch_funding_rate(sym)

    async def get_open_interest(self, symbol: Optional[str] = None) -> Optional[float]:
        """현재 미결제약정 반환 (실패 시 None)"""
        sym = symbol or self._symbols[0]
        return await self._rest.fetch_open_interest(sym)

    async def get_long_short_ratio(self, symbol: Optional[str] = None) -> Optional[float]:
        """현재 롱숏비율 반환 (실패 시 None)"""
        sym = symbol or self._symbols[0]
        return await self._rest.fetch_long_short_ratio(sym)

    def is_ready(self, symbol: Optional[str] = None) -> bool:
        """지정 심볼(또는 전체)의 모든 타임프레임 데이터가 충분한지 확인"""
        targets = [symbol] if symbol else self._symbols
        for sym in targets:
            for tf in self._config.all_timeframes:
                min_req = self.MIN_CANDLES.get(tf, 30)
                if not self._store.has_enough(sym, tf, min_req):
                    return False
        return True

    def update_symbols(self, new_symbols: List[str]) -> None:
        """활성 심볼 목록을 교체한다 (주간 로테이션 등에서 호출)."""
        old = set(self._symbols)
        new = set(new_symbols)
        added = new - old
        removed = old - new
        self._symbols = list(new_symbols)
        if added or removed:
            logger.info(
                f"[MarketData] 심볼 업데이트: +"
                f"{', '.join(s.split('/')[0] for s in added) or '-'} / "
                f"-{', '.join(s.split('/')[0] for s in removed) or '-'}"
            )

    async def warm_up_symbols(self, symbols: List[str]) -> None:
        """지정 심볼들의 모든 타임프레임 캔들을 로딩한다 (신규 심볼 진입 시)."""
        for symbol in symbols:
            for tf in self._config.all_timeframes:
                await self.refresh_candles(tf, symbol=symbol)
                count = self._store.count(symbol, tf)
                logger.info(f"[MarketData] 신규 심볼 캔들 로딩: {symbol} {tf} — {count}개")
