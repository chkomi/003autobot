"""
Layer 1: 추세 필터 (4H 타임프레임)
EMA 21/55/200 리본 + Supertrend로 매크로 추세 방향을 결정한다.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger

from config.strategy_params import TrendFilterParams
from strategy.indicators import calc_ema, calc_supertrend


@dataclass
class TrendSignal:
    regime: str               # "BULL" | "BEAR" | "NEUTRAL"
    ema_fast: float
    ema_mid: float
    ema_slow: float
    price: float
    supertrend_bullish: bool
    supertrend_value: float
    ema_aligned: bool         # EMA 정렬 여부 (fast > mid > slow)

    @property
    def is_bullish(self) -> bool:
        return self.regime == "BULL"

    @property
    def is_bearish(self) -> bool:
        return self.regime == "BEAR"


class TrendFilter:
    """4H 추세 필터 — Layer 1"""

    def __init__(self, params: TrendFilterParams):
        self._p = params

    def analyze(self, df: pd.DataFrame, allow_partial: bool = False) -> Optional[TrendSignal]:
        """4H 캔들 DataFrame을 입력받아 TrendSignal을 반환한다.

        Args:
            df: OHLCV DataFrame (index=DatetimeIndex, columns=[open,high,low,close,volume])
            allow_partial: True이면 EMA 완전 정렬 없이도 BULL/BEAR 판정
                          (price > ema_slow + supertrend 방향만으로 판정)

        Returns:
            TrendSignal 또는 None (데이터 부족 시)
        """
        if len(df) < self._p.ema_slow + 10:
            logger.debug(f"TrendFilter: 데이터 부족 ({len(df)}개)")
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]

        ema_fast = calc_ema(close, self._p.ema_fast)
        ema_mid = calc_ema(close, self._p.ema_mid)
        ema_slow = calc_ema(close, self._p.ema_slow)
        st = calc_supertrend(high, low, close, self._p.supertrend_period, self._p.supertrend_multiplier)

        # 최신 값 추출
        f = float(ema_fast.iloc[-1])
        m = float(ema_mid.iloc[-1])
        s = float(ema_slow.iloc[-1])
        price = float(close.iloc[-1])
        st_dir = float(st["supertrend_dir"].iloc[-1])
        st_val = float(st["supertrend"].iloc[-1])

        if any(pd.isna(v) for v in [f, m, s, st_dir]):
            return None

        # 추세 판단
        ema_aligned_bull = f > m > s
        ema_aligned_bear = f < m < s
        supertrend_bullish = st_dir == 1
        above_slow_ema = price > s

        if allow_partial:
            # 완화 모드: price vs ema_slow + supertrend 방향만으로 판정
            if supertrend_bullish and above_slow_ema:
                regime = "BULL"
            elif not supertrend_bullish and not above_slow_ema:
                regime = "BEAR"
            else:
                regime = "NEUTRAL"
        else:
            # 엄격 모드: EMA 완전 정렬 필수
            if ema_aligned_bull and supertrend_bullish and above_slow_ema:
                regime = "BULL"
            elif ema_aligned_bear and not supertrend_bullish and not above_slow_ema:
                regime = "BEAR"
            else:
                regime = "NEUTRAL"

        return TrendSignal(
            regime=regime,
            ema_fast=f,
            ema_mid=m,
            ema_slow=s,
            price=price,
            supertrend_bullish=supertrend_bullish,
            supertrend_value=st_val,
            ema_aligned=ema_aligned_bull or ema_aligned_bear,
        )
