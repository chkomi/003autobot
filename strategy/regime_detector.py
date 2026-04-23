"""
시장 레짐 감지기.
4H 캔들 데이터를 분석해 현재 시장 상태를 판별한다.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger

from strategy.indicators import calc_adx, calc_bollinger, calc_atr, calc_ema


@dataclass
class RegimeResult:
    """시장 레짐 판별 결과"""
    regime: str           # "TRENDING" | "RANGING" | "VOLATILE"
    adx_value: float      # ADX 값
    bb_bandwidth: float   # 볼린저 밴드 폭
    atr_pct: float        # ATR / 현재가 비율

    @property
    def is_trending(self) -> bool:
        return self.regime == "TRENDING"

    @property
    def is_ranging(self) -> bool:
        return self.regime == "RANGING"

    @property
    def is_volatile(self) -> bool:
        return self.regime == "VOLATILE"


class RegimeDetector:
    """4H 시장 레짐 감지기"""

    def __init__(
        self,
        adx_trending_threshold: float = 25.0,
        adx_ranging_threshold: float = 20.0,
        bb_bandwidth_low: float = 0.03,
        bb_bandwidth_high: float = 0.06,
        atr_pct_volatile: float = 0.02,
    ):
        self._adx_trending = adx_trending_threshold
        self._adx_ranging = adx_ranging_threshold
        self._bb_low = bb_bandwidth_low
        self._bb_high = bb_bandwidth_high
        self._atr_volatile = atr_pct_volatile

    def detect(self, df: pd.DataFrame) -> Optional[RegimeResult]:
        """4H OHLCV DataFrame으로 시장 레짐을 판별한다.

        Args:
            df: OHLCV DataFrame (columns: open, high, low, close, volume)

        Returns:
            RegimeResult 또는 None (데이터 부족 시)
        """
        if len(df) < 50:
            logger.debug(f"RegimeDetector: 데이터 부족 ({len(df)}개)")
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # ADX 계산
        adx_df = calc_adx(high, low, close, period=14)
        adx_val = float(adx_df["adx"].iloc[-1])

        # 볼린저 밴드 폭 계산
        bb_df = calc_bollinger(close, period=20, std=2.0)
        bb_bw = float(bb_df["bb_bandwidth"].iloc[-1])

        # ATR 대비 가격 비율
        atr = calc_atr(high, low, close, period=14)
        atr_val = float(atr.iloc[-1])
        current_price = float(close.iloc[-1])
        atr_pct = atr_val / current_price if current_price > 0 else 0

        if any(pd.isna(v) for v in [adx_val, bb_bw, atr_pct]):
            return None

        # EMA50 대비 가격 위치 (추세 일관성 확인)
        ema50 = calc_ema(close, 50)
        ema50_val = float(ema50.iloc[-1])
        price_above_ema = current_price > ema50_val

        # 최근 10봉 동안 EMA50 위/아래 일관성 체크
        recent_close = close.iloc[-10:]
        recent_ema = ema50.iloc[-10:]
        consistency = (recent_close > recent_ema).sum() / len(recent_close)
        # consistency > 0.8 또는 < 0.2면 가격이 한 방향으로 일관되게 이동

        # 레짐 판별
        if bb_bw > self._bb_high or atr_pct > self._atr_volatile:
            regime = "VOLATILE"
        elif adx_val > self._adx_trending and (consistency > 0.7 or consistency < 0.3):
            regime = "TRENDING"
        elif adx_val < self._adx_ranging and bb_bw < self._bb_low:
            regime = "RANGING"
        elif adx_val < self._adx_ranging:
            regime = "RANGING"
        else:
            # ADX 20~25 구간: 약한 추세 → TRENDING으로 분류 (보수적)
            regime = "TRENDING"

        logger.debug(
            f"레짐 감지: {regime} | ADX={adx_val:.1f} BB_BW={bb_bw:.4f} ATR%={atr_pct:.4f} "
            f"EMA일관성={consistency:.1%}"
        )

        return RegimeResult(
            regime=regime,
            adx_value=adx_val,
            bb_bandwidth=bb_bw,
            atr_pct=atr_pct,
        )
