"""
종합 스코어링 엔진.
5개 차원(추세/모멘텀/볼륨/변동성/심리)을 0~100점으로 평가하고
가중합으로 최종 스코어를 산출한다.
기존 3-Layer 이진 확인 시스템의 보조 레이어로 동작한다.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from config.strategy_params import MacroParams, ScoringParams, SentimentParams
from strategy.indicators import (
    calc_adx,
    calc_atr,
    calc_bollinger,
    calc_ema,
    calc_macd,
    calc_obv,
    calc_rsi,
    calc_stochastic,
    calc_volume_sma,
)


@dataclass
class ScoreBreakdown:
    """각 차원별 점수와 최종 가중합"""
    trend: float = 0.0          # 0~100
    momentum: float = 0.0       # 0~100
    volume: float = 0.0         # 0~100
    volatility: float = 0.0     # 0~100
    sentiment: float = 0.0      # 0~100
    macro: float = 0.0          # 0~100 (거시경제 매크로)
    total: float = 0.0          # 가중합 0~100

    # 방향 판단
    direction: str = "NEUTRAL"  # "LONG" | "SHORT" | "NEUTRAL"

    # 디버깅용 상세
    details: dict = field(default_factory=dict)

    @property
    def signal_strength(self) -> str:
        if self.total >= 75:
            return "STRONG"
        elif self.total >= 60:
            return "MODERATE"
        elif self.total <= 25:
            return "STRONG_REVERSE"
        elif self.total <= 40:
            return "WEAK_REVERSE"
        return "NEUTRAL"

    def summary(self) -> str:
        return (
            f"Score={self.total:.1f} ({self.signal_strength}) | "
            f"추세={self.trend:.0f} 모멘텀={self.momentum:.0f} "
            f"볼륨={self.volume:.0f} 변동성={self.volatility:.0f} "
            f"심리={self.sentiment:.0f} 매크로={self.macro:.0f} | 방향={self.direction}"
        )


@dataclass
class SentimentData:
    """외부에서 주입하는 심리 데이터"""
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    prev_open_interest: Optional[float] = None
    long_short_ratio: Optional[float] = None


@dataclass
class MacroData:
    """외부에서 주입하는 거시경제 매크로 데이터"""
    fear_greed_index: Optional[int] = None          # 0~100
    fear_greed_label: Optional[str] = None
    upcoming_high_impact_count: int = 0             # 24시간 내 고영향 이벤트 수
    nearest_high_impact_hours: Optional[float] = None  # 가장 가까운 고영향 이벤트까지 시간
    nearest_event_title: Optional[str] = None
    btc_dominance: Optional[float] = None
    total_market_cap_change_24h: Optional[float] = None


class ScoreEngine:
    """멀티팩터 스코어링 엔진"""

    def __init__(self, scoring: ScoringParams, sentiment_params: SentimentParams,
                 macro_params: Optional[MacroParams] = None):
        self._s = scoring
        self._sp = sentiment_params
        self._mp = macro_params or MacroParams()

    def evaluate(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        sentiment: Optional[SentimentData] = None,
        macro: Optional[MacroData] = None,
    ) -> ScoreBreakdown:
        """모든 타임프레임의 OHLCV 데이터와 심리 데이터를 종합 평가한다."""
        result = ScoreBreakdown()
        details = {}

        # ── 1. 추세 점수 (4H 기반, 0~100) ─────────────────────────
        trend_score, trend_dir, trend_details = self._score_trend(df_4h)
        result.trend = trend_score
        details["trend"] = trend_details

        # ── 2. 모멘텀 점수 (1H 기반, 0~100) ────────────────────────
        mom_score, mom_details = self._score_momentum(df_1h, trend_dir)
        result.momentum = mom_score
        details["momentum"] = mom_details

        # ── 3. 볼륨 점수 (1H 기반, 0~100) ──────────────────────────
        vol_score, vol_details = self._score_volume(df_1h, trend_dir)
        result.volume = vol_score
        details["volume"] = vol_details

        # ── 4. 변동성 점수 (15M 기반, 0~100) ────────────────────────
        vola_score, vola_details = self._score_volatility(df_15m)
        result.volatility = vola_score
        details["volatility"] = vola_details

        # ── 5. 심리 점수 (외부 데이터, 0~100) ───────────────────────
        sent_score, sent_details = self._score_sentiment(sentiment, trend_dir)
        result.sentiment = sent_score
        details["sentiment"] = sent_details

        # ── 6. 매크로 점수 (거시경제, 0~100) ────────────────────────
        macro_score, macro_details = self._score_macro(macro, trend_dir)
        result.macro = macro_score
        details["macro"] = macro_details

        # ── 가중합 계산 ──────────────────────────────────────────────
        w = self._s
        result.total = (
            result.trend * w.weight_trend
            + result.momentum * w.weight_momentum
            + result.volume * w.weight_volume
            + result.volatility * w.weight_volatility
            + result.sentiment * w.weight_sentiment
            + result.macro * w.weight_macro
        )

        result.direction = trend_dir
        result.details = details

        logger.debug(f"[스코어] {result.summary()}")
        return result

    # ── 추세 스코어링 ────────────────────────────────────────────────

    def _score_trend(self, df: pd.DataFrame) -> tuple[float, str, dict]:
        """4H 추세 점수 산출. (점수, 방향, 상세)"""
        if len(df) < 210:
            return 50.0, "NEUTRAL", {"error": "데이터 부족"}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        ema50 = calc_ema(close, 50)
        ema200 = calc_ema(close, 200)
        adx_df = calc_adx(high, low, close, 14)

        price = float(close.iloc[-1])
        ema50_val = float(ema50.iloc[-1])
        ema200_val = float(ema200.iloc[-1])
        adx_val = float(adx_df["adx"].iloc[-1])
        plus_di = float(adx_df["plus_di"].iloc[-1])
        minus_di = float(adx_df["minus_di"].iloc[-1])

        if any(np.isnan(v) for v in [ema50_val, ema200_val, adx_val]):
            return 50.0, "NEUTRAL", {"error": "NaN 지표"}

        score = 0.0
        detail = {}

        # EMA 정배열/역배열 (최대 40점)
        if ema50_val > ema200_val:
            score += 20
            detail["ema_cross"] = "골든크로스"
            if price > ema50_val:
                score += 20
                detail["price_position"] = "EMA50 위"
            else:
                score += 10
                detail["price_position"] = "EMA50 아래, EMA200 위"
            direction = "LONG"
        elif ema50_val < ema200_val:
            score += 20
            detail["ema_cross"] = "데드크로스"
            if price < ema50_val:
                score += 20
                detail["price_position"] = "EMA50 아래"
            else:
                score += 10
                detail["price_position"] = "EMA50 위, EMA200 아래"
            direction = "SHORT"
        else:
            score += 0
            direction = "NEUTRAL"
            detail["ema_cross"] = "평행"

        # ADX 추세 강도 (최대 35점)
        if adx_val >= self._s.adx_very_strong:
            score += 35
            detail["adx"] = f"매우 강한 추세 ({adx_val:.1f})"
        elif adx_val >= self._s.adx_strong:
            score += 25
            detail["adx"] = f"강한 추세 ({adx_val:.1f})"
        elif adx_val >= 15:
            score += 10
            detail["adx"] = f"약한 추세 ({adx_val:.1f})"
        else:
            score += 0
            detail["adx"] = f"추세 없음 ({adx_val:.1f})"

        # DI 방향 일치 보너스 (최대 25점)
        if direction == "LONG" and plus_di > minus_di:
            score += 25
            detail["di"] = f"+DI 우위 ({plus_di:.1f} > {minus_di:.1f})"
        elif direction == "SHORT" and minus_di > plus_di:
            score += 25
            detail["di"] = f"-DI 우위 ({minus_di:.1f} > {plus_di:.1f})"
        else:
            score += 5
            detail["di"] = f"DI 불일치 (+{plus_di:.1f} / -{minus_di:.1f})"

        return min(score, 100.0), direction, detail

    # ── 모멘텀 스코어링 ──────────────────────────────────────────────

    def _score_momentum(self, df: pd.DataFrame, trend_dir: str) -> tuple[float, dict]:
        """1H 모멘텀 점수 산출."""
        if len(df) < 40:
            return 50.0, {"error": "데이터 부족"}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        rsi = calc_rsi(close, 14)
        macd_df = calc_macd(close)
        stoch = calc_stochastic(high, low, close)

        rsi_val = float(rsi.iloc[-1])
        macd_hist = float(macd_df["macd_hist"].iloc[-1])
        macd_hist_prev = float(macd_df["macd_hist"].iloc[-2])
        stoch_k = float(stoch["stoch_k"].iloc[-1])

        if any(np.isnan(v) for v in [rsi_val, macd_hist, stoch_k]):
            return 50.0, {"error": "NaN 지표"}

        score = 0.0
        detail = {}

        # RSI 구간 점수 (최대 40점)
        if trend_dir == "LONG":
            if 40 <= rsi_val <= 60:
                score += 40  # 건강한 상승 모멘텀
                detail["rsi"] = f"건강한 구간 ({rsi_val:.1f})"
            elif 60 < rsi_val <= 70:
                score += 30  # 약간 과매수
                detail["rsi"] = f"약간 과열 ({rsi_val:.1f})"
            elif rsi_val > 70:
                score += 10  # 과매수 주의
                detail["rsi"] = f"과매수 ({rsi_val:.1f})"
            elif 30 <= rsi_val < 40:
                score += 20  # 반등 가능
                detail["rsi"] = f"반등 구간 ({rsi_val:.1f})"
            else:
                score += 5
                detail["rsi"] = f"과매도 ({rsi_val:.1f})"
        elif trend_dir == "SHORT":
            if 40 <= rsi_val <= 60:
                score += 40
                detail["rsi"] = f"건강한 하락 모멘텀 ({rsi_val:.1f})"
            elif 30 <= rsi_val < 40:
                score += 30
                detail["rsi"] = f"약간 과매도 ({rsi_val:.1f})"
            elif rsi_val < 30:
                score += 10
                detail["rsi"] = f"과매도 ({rsi_val:.1f})"
            elif 60 < rsi_val <= 70:
                score += 20
                detail["rsi"] = f"반락 구간 ({rsi_val:.1f})"
            else:
                score += 5
                detail["rsi"] = f"과매수 ({rsi_val:.1f})"
        else:
            score += 25  # 중립
            detail["rsi"] = f"중립 ({rsi_val:.1f})"

        # MACD 히스토그램 방향 (최대 35점)
        hist_increasing = macd_hist > macd_hist_prev
        if trend_dir == "LONG" and macd_hist > 0 and hist_increasing:
            score += 35
            detail["macd"] = "상승 모멘텀 가속"
        elif trend_dir == "LONG" and macd_hist > 0:
            score += 20
            detail["macd"] = "상승 모멘텀 (감속)"
        elif trend_dir == "SHORT" and macd_hist < 0 and not hist_increasing:
            score += 35
            detail["macd"] = "하락 모멘텀 가속"
        elif trend_dir == "SHORT" and macd_hist < 0:
            score += 20
            detail["macd"] = "하락 모멘텀 (감속)"
        else:
            score += 5
            detail["macd"] = "모멘텀 불일치"

        # 스토캐스틱 (최대 25점)
        if trend_dir == "LONG" and stoch_k < self._s.stoch_overbought:
            score += 25 if stoch_k <= self._s.stoch_oversold else 15
            detail["stoch"] = f"롱 유리 ({stoch_k:.1f})"
        elif trend_dir == "SHORT" and stoch_k > self._s.stoch_oversold:
            score += 25 if stoch_k >= self._s.stoch_overbought else 15
            detail["stoch"] = f"숏 유리 ({stoch_k:.1f})"
        else:
            score += 5
            detail["stoch"] = f"중립 ({stoch_k:.1f})"

        return min(score, 100.0), detail

    # ── 볼륨 스코어링 ────────────────────────────────────────────────

    def _score_volume(self, df: pd.DataFrame, trend_dir: str) -> tuple[float, dict]:
        """1H 볼륨 점수 산출."""
        if len(df) < 25:
            return 50.0, {"error": "데이터 부족"}

        close = df["close"]
        volume = df["volume"]

        vol_sma = calc_volume_sma(volume, 20)
        obv = calc_obv(close, volume)

        curr_vol = float(volume.iloc[-1])
        avg_vol = float(vol_sma.iloc[-1])
        obv_now = float(obv.iloc[-1])
        obv_prev = float(obv.iloc[-self._s.obv_lookback])

        if any(np.isnan(v) for v in [curr_vol, avg_vol, obv_now]):
            return 50.0, {"error": "NaN 지표"}

        score = 0.0
        detail = {}

        # 거래량 대비 이동평균 (최대 50점)
        vol_ratio = curr_vol / avg_vol if avg_vol > 0 else 1.0
        if vol_ratio >= 2.0:
            score += 50
            detail["vol_ratio"] = f"거래량 폭증 ({vol_ratio:.1f}x)"
        elif vol_ratio >= 1.5:
            score += 40
            detail["vol_ratio"] = f"거래량 증가 ({vol_ratio:.1f}x)"
        elif vol_ratio >= 1.0:
            score += 25
            detail["vol_ratio"] = f"평균 이상 ({vol_ratio:.1f}x)"
        else:
            score += 10
            detail["vol_ratio"] = f"거래량 감소 ({vol_ratio:.1f}x)"

        # OBV 방향 (최대 50점)
        obv_rising = obv_now > obv_prev
        if trend_dir == "LONG" and obv_rising:
            score += 50
            detail["obv"] = "OBV 상승 (추세 확인)"
        elif trend_dir == "SHORT" and not obv_rising:
            score += 50
            detail["obv"] = "OBV 하락 (추세 확인)"
        elif trend_dir == "LONG" and not obv_rising:
            score += 10
            detail["obv"] = "OBV 하락 (다이버전스 주의)"
        elif trend_dir == "SHORT" and obv_rising:
            score += 10
            detail["obv"] = "OBV 상승 (다이버전스 주의)"
        else:
            score += 25
            detail["obv"] = "OBV 중립"

        return min(score, 100.0), detail

    # ── 변동성 스코어링 ──────────────────────────────────────────────

    def _score_volatility(self, df: pd.DataFrame) -> tuple[float, dict]:
        """15M 변동성 점수 산출. 적절한 변동성 = 높은 점수."""
        if len(df) < 25:
            return 50.0, {"error": "데이터 부족"}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        bb = calc_bollinger(close, 20, 2.0)
        atr = calc_atr(high, low, close, 14)

        bandwidth = float(bb["bb_bandwidth"].iloc[-1])
        atr_val = float(atr.iloc[-1])
        price = float(close.iloc[-1])
        atr_pct = atr_val / price if price > 0 else 0

        if np.isnan(bandwidth):
            return 50.0, {"error": "NaN 지표"}

        score = 0.0
        detail = {}

        # 볼린저 밴드 폭 (최대 60점)
        # 적절한 변동성(중간)이 가장 유리 — 너무 낮으면 기회 없음, 너무 높으면 위험
        if self._s.bb_bandwidth_low <= bandwidth <= self._s.bb_bandwidth_high:
            score += 60
            detail["bandwidth"] = f"적절한 변동성 ({bandwidth:.4f})"
        elif bandwidth < self._s.bb_bandwidth_low:
            score += 40  # 스퀴즈 — 곧 큰 움직임 가능
            detail["bandwidth"] = f"스퀴즈 (돌파 대기) ({bandwidth:.4f})"
        else:
            score += 20  # 과변동 — 휩소 위험
            detail["bandwidth"] = f"과변동 주의 ({bandwidth:.4f})"

        # ATR 대비 가격 비율 (최대 40점)
        if 0.005 <= atr_pct <= 0.03:
            score += 40
            detail["atr_pct"] = f"적절한 ATR ({atr_pct:.4f})"
        elif atr_pct < 0.005:
            score += 25
            detail["atr_pct"] = f"낮은 ATR ({atr_pct:.4f})"
        else:
            score += 15
            detail["atr_pct"] = f"높은 ATR ({atr_pct:.4f})"

        return min(score, 100.0), detail

    # ── 심리 스코어링 ────────────────────────────────────────────────

    def _score_sentiment(
        self, data: Optional[SentimentData], trend_dir: str
    ) -> tuple[float, dict]:
        """심리 지표 점수 산출."""
        if data is None:
            return 50.0, {"note": "심리 데이터 없음 (중립 처리)"}

        score = 0.0
        detail = {}
        components = 0

        # 펀딩비 (최대 33점)
        if data.funding_rate is not None:
            components += 1
            fr = data.funding_rate
            neutral = self._sp.funding_neutral_range
            if abs(fr) <= neutral:
                score += 25
                detail["funding"] = f"중립 ({fr:.4%})"
            elif trend_dir == "LONG" and fr < -neutral:
                # 음의 펀딩 = 숏 과열 → 롱에 유리
                score += 33
                detail["funding"] = f"숏 과열, 롱 유리 ({fr:.4%})"
            elif trend_dir == "SHORT" and fr > neutral:
                score += 33
                detail["funding"] = f"롱 과열, 숏 유리 ({fr:.4%})"
            elif trend_dir == "LONG" and fr > neutral:
                score += 10
                detail["funding"] = f"롱 과열 주의 ({fr:.4%})"
            elif trend_dir == "SHORT" and fr < -neutral:
                score += 10
                detail["funding"] = f"숏 과열 주의 ({fr:.4%})"
            else:
                score += 15
                detail["funding"] = f"({fr:.4%})"

        # 미결제약정 변화 (최대 34점)
        if data.open_interest is not None and data.prev_open_interest is not None:
            components += 1
            if data.prev_open_interest > 0:
                oi_change = (data.open_interest - data.prev_open_interest) / data.prev_open_interest
            else:
                oi_change = 0

            if abs(oi_change) >= self._sp.oi_surge_threshold:
                if oi_change > 0:
                    score += 34  # OI 급증 = 새 자금 유입
                    detail["oi"] = f"OI 급증 ({oi_change:+.2%})"
                else:
                    score += 15  # OI 급감 = 포지션 정리
                    detail["oi"] = f"OI 급감 ({oi_change:+.2%})"
            else:
                score += 20
                detail["oi"] = f"OI 안정 ({oi_change:+.2%})"

        # 롱숏비율 (최대 33점)
        if data.long_short_ratio is not None:
            components += 1
            ls = data.long_short_ratio
            if trend_dir == "LONG" and ls < self._sp.long_short_extreme_long:
                score += 33  # 롱 과열 아님 → 진입 여유
                detail["ls_ratio"] = f"롱 여유 ({ls:.2f})"
            elif trend_dir == "SHORT" and ls > self._sp.long_short_extreme_short:
                score += 33  # 숏 과열 아님
                detail["ls_ratio"] = f"숏 여유 ({ls:.2f})"
            elif trend_dir == "LONG" and ls >= self._sp.long_short_extreme_long:
                score += 10  # 롱 과열
                detail["ls_ratio"] = f"롱 과열 ({ls:.2f})"
            elif trend_dir == "SHORT" and ls <= self._sp.long_short_extreme_short:
                score += 10
                detail["ls_ratio"] = f"숏 과열 ({ls:.2f})"
            else:
                score += 20
                detail["ls_ratio"] = f"({ls:.2f})"

        # 데이터가 전혀 없으면 중립
        if components == 0:
            return 50.0, {"note": "심리 데이터 없음"}

        # 사용 가능한 컴포넌트 수로 정규화 (100점 만점)
        max_possible = components * 33.5
        normalized = (score / max_possible) * 100 if max_possible > 0 else 50.0

        return min(normalized, 100.0), detail

    # ── 매크로 스코어링 ──────────────────────────────────────────────

    def _score_macro(
        self, data: Optional[MacroData], trend_dir: str
    ) -> tuple[float, dict]:
        """거시경제 매크로 점수 산출.

        Fear & Greed Index, 고영향 이벤트 근접도, 시장 전체 시총 변화율,
        BTC 도미넌스를 종합해 매크로 환경이 거래에 유리한지 판단한다.
        """
        if data is None:
            return 50.0, {"note": "매크로 데이터 없음 (중립 처리)"}

        score = 0.0
        detail = {}
        components = 0
        mp = self._mp

        # 1. Fear & Greed Index (최대 35점)
        if data.fear_greed_index is not None:
            components += 1
            fg = data.fear_greed_index
            if trend_dir == "LONG":
                # 롱: 극단 공포(20 이하)는 역발상 매수 기회 → 높은 점수
                if fg <= mp.fg_extreme_fear:
                    score += 35
                    detail["fear_greed"] = f"극단 공포 = 역발상 매수 기회 ({fg})"
                elif fg <= mp.fg_fear:
                    score += 28
                    detail["fear_greed"] = f"공포 = 매수 기회 ({fg})"
                elif fg <= mp.fg_greed:
                    score += 20  # 중립~약간 탐욕
                    detail["fear_greed"] = f"중립 구간 ({fg})"
                elif fg <= mp.fg_extreme_greed:
                    score += 10  # 탐욕 → 롱 위험
                    detail["fear_greed"] = f"탐욕 주의 ({fg})"
                else:
                    score += 5   # 극단 탐욕 → 롱 매우 위험
                    detail["fear_greed"] = f"극단 탐욕 = 조정 임박 ({fg})"
            elif trend_dir == "SHORT":
                # 숏: 극단 탐욕이 역발상 매도 기회
                if fg >= mp.fg_extreme_greed:
                    score += 35
                    detail["fear_greed"] = f"극단 탐욕 = 역발상 매도 기회 ({fg})"
                elif fg >= mp.fg_greed:
                    score += 28
                    detail["fear_greed"] = f"탐욕 = 매도 기회 ({fg})"
                elif fg >= mp.fg_fear:
                    score += 20
                    detail["fear_greed"] = f"중립 구간 ({fg})"
                elif fg >= mp.fg_extreme_fear:
                    score += 10
                    detail["fear_greed"] = f"공포 = 숏 위험 ({fg})"
                else:
                    score += 5
                    detail["fear_greed"] = f"극단 공포 = 반등 임박 ({fg})"
            else:
                score += 17
                detail["fear_greed"] = f"중립 ({fg})"

        # 2. 고영향 이벤트 근접도 (최대 30점)
        #    이벤트가 가까울수록 점수 낮음 (변동성 리스크)
        components += 1
        if data.upcoming_high_impact_count == 0:
            score += 30
            detail["events"] = "24시간 내 고영향 이벤트 없음"
        elif data.nearest_high_impact_hours is not None:
            hrs = data.nearest_high_impact_hours
            if hrs <= mp.event_danger_hours:
                score += 5  # 매우 가까움 — 거래 위험
                detail["events"] = f"{data.nearest_event_title} {hrs:.1f}시간 후 (위험)"
            elif hrs <= mp.event_caution_hours:
                score += 15  # 주의
                detail["events"] = f"{data.nearest_event_title} {hrs:.1f}시간 후 (주의)"
            else:
                score += 25  # 충분한 거리
                detail["events"] = f"{data.nearest_event_title} {hrs:.1f}시간 후"
        else:
            score += 15
            detail["events"] = f"고영향 이벤트 {data.upcoming_high_impact_count}건"

        # 3. 시장 전체 시총 변화율 (최대 20점)
        if data.total_market_cap_change_24h is not None:
            components += 1
            mc = data.total_market_cap_change_24h
            if trend_dir == "LONG":
                if mc > 2.0:
                    score += 20
                    detail["market_cap"] = f"시장 강세 ({mc:+.1f}%)"
                elif mc > 0:
                    score += 15
                    detail["market_cap"] = f"시장 약강세 ({mc:+.1f}%)"
                elif mc > -2.0:
                    score += 10
                    detail["market_cap"] = f"시장 약보합 ({mc:+.1f}%)"
                else:
                    score += 5
                    detail["market_cap"] = f"시장 약세 ({mc:+.1f}%)"
            elif trend_dir == "SHORT":
                if mc < -2.0:
                    score += 20
                    detail["market_cap"] = f"시장 약세 = 숏 유리 ({mc:+.1f}%)"
                elif mc < 0:
                    score += 15
                    detail["market_cap"] = f"시장 약보합 ({mc:+.1f}%)"
                elif mc < 2.0:
                    score += 10
                    detail["market_cap"] = f"시장 약강세 ({mc:+.1f}%)"
                else:
                    score += 5
                    detail["market_cap"] = f"시장 강세 = 숏 불리 ({mc:+.1f}%)"
            else:
                score += 10
                detail["market_cap"] = f"({mc:+.1f}%)"

        # 4. BTC 도미넌스 (최대 15점, 알트코인 거래 시 참고)
        if data.btc_dominance is not None:
            components += 1
            bd = data.btc_dominance
            if bd >= mp.btc_dom_high:
                # BTC 도미넌스 높음 → 알트 약세 환경
                if trend_dir == "LONG":
                    score += 8   # 알트 롱에 불리
                    detail["btc_dom"] = f"BTC 도미넌스 높음 = 알트 약세 ({bd:.1f}%)"
                else:
                    score += 12
                    detail["btc_dom"] = f"BTC 도미넌스 높음 ({bd:.1f}%)"
            elif bd <= mp.btc_dom_low:
                # BTC 도미넌스 낮음 → 알트 시즌
                if trend_dir == "LONG":
                    score += 15
                    detail["btc_dom"] = f"BTC 도미넌스 낮음 = 알트 시즌 ({bd:.1f}%)"
                else:
                    score += 8
                    detail["btc_dom"] = f"BTC 도미넌스 낮음 ({bd:.1f}%)"
            else:
                score += 10
                detail["btc_dom"] = f"BTC 도미넌스 중립 ({bd:.1f}%)"

        if components == 0:
            return 50.0, {"note": "매크로 데이터 없음"}

        # 정규화: 100점 만점 환산
        max_possible = 35 + 30 + (20 if data.total_market_cap_change_24h is not None else 0) + (15 if data.btc_dominance is not None else 0)
        normalized = (score / max_possible) * 100 if max_possible > 0 else 50.0

        return min(normalized, 100.0), detail
