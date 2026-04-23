"""
Layer 2: 모멘텀 트리거 (1H 타임프레임)
MACD 크로스 + RSI 유효 범위로 진입 트리거를 감지한다.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger

from config.strategy_params import MomentumParams
from strategy.indicators import calc_macd, calc_rsi, detect_macd_cross


@dataclass
class MomentumSignal:
    macd_cross: str       # "UP" | "DOWN" | "NONE"
    rsi_value: float
    rsi_valid_long: bool  # RSI가 롱 진입 허용 범위인가
    rsi_valid_short: bool # RSI가 숏 진입 허용 범위인가
    macd_value: float
    macd_signal_value: float
    macd_hist: float

    @property
    def long_trigger(self) -> bool:
        """MACD 상향 크로스 + RSI 롱 범위"""
        return self.macd_cross == "UP" and self.rsi_valid_long

    @property
    def short_trigger(self) -> bool:
        """MACD 하향 크로스 + RSI 숏 범위"""
        return self.macd_cross == "DOWN" and self.rsi_valid_short


class MomentumTrigger:
    """1H 모멘텀 트리거 — Layer 2"""

    def __init__(self, params: MomentumParams):
        self._p = params

    def analyze(
        self,
        df: pd.DataFrame,
        rsi_long_min: float = None,
        rsi_long_max: float = None,
        rsi_short_min: float = None,
        rsi_short_max: float = None,
        macd_cross_lookback: int = None,
    ) -> Optional[MomentumSignal]:
        """1H 캔들 DataFrame을 분석해 MomentumSignal을 반환한다.

        Args:
            rsi_long_min/max: RSI 롱 범위 오버라이드 (None이면 기본값 사용)
            rsi_short_min/max: RSI 숏 범위 오버라이드
            macd_cross_lookback: MACD 크로스 감지 lookback 오버라이드
        """
        min_needed = max(self._p.macd_slow + self._p.macd_signal, self._p.rsi_period) + 5
        if len(df) < min_needed:
            logger.debug(f"MomentumTrigger: 데이터 부족 ({len(df)}개)")
            return None

        close = df["close"]

        macd_df = calc_macd(close, self._p.macd_fast, self._p.macd_slow, self._p.macd_signal)
        rsi = calc_rsi(close, self._p.rsi_period)

        lookback = macd_cross_lookback if macd_cross_lookback is not None else 3
        cross = detect_macd_cross(macd_df, lookback=lookback)
        rsi_val = float(rsi.iloc[-1])
        macd_val = float(macd_df["macd"].iloc[-1])
        sig_val = float(macd_df["macd_signal"].iloc[-1])
        hist_val = float(macd_df["macd_hist"].iloc[-1])

        if pd.isna(rsi_val) or pd.isna(macd_val):
            return None

        r_long_min = rsi_long_min if rsi_long_min is not None else self._p.rsi_long_min
        r_long_max = rsi_long_max if rsi_long_max is not None else self._p.rsi_long_max
        r_short_min = rsi_short_min if rsi_short_min is not None else self._p.rsi_short_min
        r_short_max = rsi_short_max if rsi_short_max is not None else self._p.rsi_short_max

        rsi_valid_long = r_long_min <= rsi_val <= r_long_max
        rsi_valid_short = r_short_min <= rsi_val <= r_short_max

        return MomentumSignal(
            macd_cross=cross,
            rsi_value=rsi_val,
            rsi_valid_long=rsi_valid_long,
            rsi_valid_short=rsi_valid_short,
            macd_value=macd_val,
            macd_signal_value=sig_val,
            macd_hist=hist_val,
        )
