"""
Layer 3: 미시 구조 확인 (15M 타임프레임)
볼린저 밴드 반등 + 거래량 급증으로 정밀 진입 타이밍을 확인한다.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger

from config.strategy_params import MicroParams
from strategy.indicators import calc_bollinger, calc_volume_sma


@dataclass
class MicroSignal:
    bb_bounce: str         # "LONG" | "SHORT" | "NONE"
    volume_confirmed: bool
    price: float
    bb_upper: float
    bb_mid: float
    bb_lower: float
    bb_pct: float          # 현재 가격의 밴드 내 위치 (0~1)
    current_volume: float
    avg_volume: float

    @property
    def long_confirm(self) -> bool:
        return self.bb_bounce == "LONG" and self.volume_confirmed

    @property
    def short_confirm(self) -> bool:
        return self.bb_bounce == "SHORT" and self.volume_confirmed


class MicroConfirmation:
    """15M 미시 구조 확인 — Layer 3"""

    def __init__(self, params: MicroParams):
        self._p = params

    def analyze(
        self,
        df: pd.DataFrame,
        volume_multiplier: float = None,
        bb_long_threshold: float = None,
        bb_short_threshold: float = None,
    ) -> Optional[MicroSignal]:
        """15M 캔들 DataFrame을 분석해 MicroSignal을 반환한다.

        Args:
            volume_multiplier: 거래량 배수 오버라이드 (None이면 기본값 사용)
            bb_long_threshold: BB 롱 진입 임계값 오버라이드 (기본 0.35)
            bb_short_threshold: BB 숏 진입 임계값 오버라이드 (기본 0.65)
        """
        min_needed = self._p.bb_period + 5
        if len(df) < min_needed:
            logger.debug(f"MicroConfirmation: 데이터 부족 ({len(df)}개)")
            return None

        close = df["close"]
        volume = df["volume"]

        bb = calc_bollinger(close, self._p.bb_period, self._p.bb_std)
        vol_sma = calc_volume_sma(volume, self._p.bb_period)

        price = float(close.iloc[-1])
        upper = float(bb["bb_upper"].iloc[-1])
        mid = float(bb["bb_mid"].iloc[-1])
        lower = float(bb["bb_lower"].iloc[-1])
        bb_pct = float(bb["bb_pct"].iloc[-1]) if not pd.isna(bb["bb_pct"].iloc[-1]) else 0.5
        curr_vol = float(volume.iloc[-1])
        avg_vol = float(vol_sma.iloc[-1])

        if pd.isna(upper) or pd.isna(lower) or pd.isna(avg_vol):
            return None

        # 거래량 확인
        vol_mult = volume_multiplier if volume_multiplier is not None else self._p.volume_multiplier
        volume_confirmed = curr_vol >= avg_vol * vol_mult

        # 볼린저 밴드 반등 감지
        long_thresh = bb_long_threshold if bb_long_threshold is not None else 0.35
        short_thresh = bb_short_threshold if bb_short_threshold is not None else 0.65

        if bb_pct <= long_thresh:
            bb_bounce = "LONG"
        elif bb_pct >= short_thresh:
            bb_bounce = "SHORT"
        else:
            bb_bounce = "NONE"

        return MicroSignal(
            bb_bounce=bb_bounce,
            volume_confirmed=volume_confirmed,
            price=price,
            bb_upper=upper,
            bb_mid=mid,
            bb_lower=lower,
            bb_pct=bb_pct,
            current_volume=curr_vol,
            avg_volume=avg_vol,
        )
