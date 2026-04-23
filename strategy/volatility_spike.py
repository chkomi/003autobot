"""
변동성 급증(Volatility Spike) 감지기.

돌발 뉴스(전쟁/규제/해킹/사토시 관련 등) 직후 발생하는 급락·급등을
최단 주기 캔들로 즉시 감지해, 기존 ATR 기반 손절선까지 끌려가기 전에
선제적으로 포지션을 정리할 수 있게 한다.

사용 지표 (외부 의존성 없이 pandas/numpy로 계산):
  1) 실현변동성 비율 — 최근 N분 표준편차 vs 과거 M시간 평균의 비율
  2) 최근 수익률 z-score — 최신 봉 수익률이 과거 분포에서 몇 σ 벗어났는가
  3) 최근 레인지 확대율 — (high-low)/close 의 직전 대비 배수

셋 중 하나라도 임계값을 넘으면 SPIKE 로 판정한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SpikeResult:
    is_spike: bool
    vol_ratio: float          # 최근/과거 실현변동성 비율
    return_zscore: float      # 최근 수익률 z-score (부호 유지)
    range_ratio: float        # (high-low)/close 의 과거 평균 대비 배수
    direction: str            # "UP" | "DOWN" | "NONE"
    reason: str = ""

    def summary(self) -> str:
        if not self.is_spike:
            return f"spike=no vr={self.vol_ratio:.2f} z={self.return_zscore:+.2f}"
        return (
            f"SPIKE[{self.direction}] reason={self.reason} "
            f"vr={self.vol_ratio:.2f} z={self.return_zscore:+.2f} "
            f"range={self.range_ratio:.2f}"
        )


class VolatilitySpikeDetector:
    """짧은 타임프레임(15m/5m/1m) 캔들에서 변동성 급증을 감지한다.

    Args:
        recent_bars: 최근 실현변동성 계산에 쓰는 봉 수 (예: 3~5)
        baseline_bars: 과거 평균 계산에 쓰는 봉 수 (예: 48~96)
        vol_ratio_threshold: recent/baseline 실현변동성 비율 임계 (예: 3.0)
        z_threshold: 최근 수익률 절대 z-score 임계 (예: 3.5)
        range_ratio_threshold: 현재 봉 레인지 비율 임계 (예: 3.0)
    """

    def __init__(
        self,
        recent_bars: int = 3,
        baseline_bars: int = 72,
        vol_ratio_threshold: float = 3.0,
        z_threshold: float = 3.5,
        range_ratio_threshold: float = 3.0,
    ):
        self.recent_bars = max(2, recent_bars)
        self.baseline_bars = max(20, baseline_bars)
        self.vol_ratio_threshold = vol_ratio_threshold
        self.z_threshold = z_threshold
        self.range_ratio_threshold = range_ratio_threshold

    def detect(self, df: pd.DataFrame) -> SpikeResult:
        """OHLC DataFrame을 받아 현재 시점의 스파이크 여부 평가.

        df는 최소 baseline_bars + recent_bars 개 이상의 봉이 필요하다.
        부족하면 감지 불가(is_spike=False)로 반환한다.
        """
        empty = SpikeResult(
            is_spike=False, vol_ratio=0.0, return_zscore=0.0,
            range_ratio=0.0, direction="NONE", reason="데이터 부족",
        )
        if df is None or len(df) < self.recent_bars + self.baseline_bars:
            return empty
        try:
            close = df["close"].astype(float)
            high = df["high"].astype(float)
            low = df["low"].astype(float)
        except (KeyError, ValueError):
            return empty

        returns = close.pct_change().dropna()
        if len(returns) < self.baseline_bars + self.recent_bars:
            return empty

        recent_ret = returns.iloc[-self.recent_bars:]
        baseline_ret = returns.iloc[-(self.baseline_bars + self.recent_bars):-self.recent_bars]

        recent_std = float(recent_ret.std(ddof=0))
        baseline_std = float(baseline_ret.std(ddof=0))
        vol_ratio = (recent_std / baseline_std) if baseline_std > 1e-9 else 0.0

        # 최근 봉 수익률의 z-score (baseline 기준)
        last_ret = float(returns.iloc[-1])
        mean_base = float(baseline_ret.mean())
        z = (last_ret - mean_base) / baseline_std if baseline_std > 1e-9 else 0.0

        # 현재 봉 레인지 확대율
        last_range = float((high.iloc[-1] - low.iloc[-1]) / max(close.iloc[-1], 1e-9))
        base_range = ((high - low) / close.replace(0, np.nan)).iloc[-(self.baseline_bars + 1):-1]
        base_range_mean = float(base_range.mean()) if not base_range.empty else 0.0
        range_ratio = (last_range / base_range_mean) if base_range_mean > 1e-9 else 0.0

        triggered: list[str] = []
        if vol_ratio >= self.vol_ratio_threshold:
            triggered.append(f"vol_ratio={vol_ratio:.2f}")
        if abs(z) >= self.z_threshold:
            triggered.append(f"z={z:+.2f}")
        if range_ratio >= self.range_ratio_threshold:
            triggered.append(f"range={range_ratio:.2f}")

        if not triggered:
            return SpikeResult(
                is_spike=False, vol_ratio=vol_ratio, return_zscore=z,
                range_ratio=range_ratio, direction="NONE",
            )

        direction = "UP" if last_ret > 0 else "DOWN"
        return SpikeResult(
            is_spike=True, vol_ratio=vol_ratio, return_zscore=z,
            range_ratio=range_ratio, direction=direction,
            reason=",".join(triggered),
        )
