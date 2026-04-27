"""
심볼 로테이터 — 주간 모멘텀 기반 심볼 선택기.

매주 일요일 자정(KST)에 호출되어 후보 심볼들을 다음 4가지 지표로 평가하고
상위 N개를 반환한다:

  점수 = 0.4 × 7일 수익률(정규화)
         + 0.3 × 트렌드 강도(EMA 정렬 + ADX)
         + 0.2 × 7일 평균 거래량(BTC 기준 상대)
         + 0.1 × 변동성 점수(ATR% 역수 — 너무 과도한 변동성 페널티)

사용법:
    ranker = SymbolRanker(rest_client)
    top_symbols = await ranker.rank(candidate_symbols, top_n=3)
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from loguru import logger

_KST = timezone(timedelta(hours=9))

# 각 지표 가중치
_W_RETURN = 0.40
_W_TREND = 0.30
_W_VOLUME = 0.20
_W_VOLATILITY = 0.10


@dataclass
class SymbolScore:
    symbol: str
    rank: int
    total_score: float        # 0~100
    return_7d: float          # 7일 수익률 (소수, 예: 0.12 = 12%)
    trend_score: float        # 0~1
    volume_score: float       # 0~1
    volatility_score: float   # 0~1 (높을수록 변동성 적절)
    atr_pct: float            # ATR / 현재가 비율
    adx: float                # ADX 값


class SymbolRanker:
    """주간 모멘텀 기반 심볼 랭킹"""

    # 기준 캔들 수
    _LOOKBACK = 200     # 지표 계산용 (1H 캔들)
    _RETURN_BARS = 168  # 7일 = 168개 1H 봉

    def __init__(self, rest_client):
        self._rest = rest_client

    async def rank(
        self,
        candidate_symbols: list[str],
        top_n: int = 3,
        timeframe: str = "1h",
    ) -> list[SymbolScore]:
        """후보 심볼들을 평가하고 상위 top_n개를 반환한다.

        Args:
            candidate_symbols: 평가할 심볼 목록
            top_n: 반환할 상위 심볼 수
            timeframe: 평가 타임프레임 (기본 1h)

        Returns:
            상위 SymbolScore 목록 (rank 순 정렬)
        """
        scores: list[SymbolScore] = []

        for symbol in candidate_symbols:
            try:
                score = await self._evaluate_symbol(symbol, timeframe)
                if score is not None:
                    scores.append(score)
            except Exception as e:
                logger.warning(f"[SymbolRanker] {symbol} 평가 실패: {e}")

        if not scores:
            logger.warning("[SymbolRanker] 평가된 심볼 없음")
            return []

        # 점수 정규화 후 순위 결정
        scores = _normalize_scores(scores)
        scores.sort(key=lambda s: s.total_score, reverse=True)

        # 순위 부여
        for i, s in enumerate(scores):
            s.rank = i + 1

        top = scores[:top_n]

        logger.info(
            f"[SymbolRanker] 상위 {top_n}개 심볼 선택:\n"
            + "\n".join(
                f"  #{s.rank} {s.symbol} | 점수={s.total_score:.1f} | "
                f"7d수익={s.return_7d:+.2%} | ADX={s.adx:.1f}"
                for s in top
            )
        )

        return top

    async def rank_symbols_list(
        self,
        candidate_symbols: list[str],
        top_n: int = 3,
    ) -> list[str]:
        """rank()의 간이 버전 — 심볼 문자열 목록만 반환."""
        scored = await self.rank(candidate_symbols, top_n=top_n)
        return [s.symbol for s in scored]

    # ── 내부 평가 ──────────────────────────────────────────

    async def _evaluate_symbol(
        self, symbol: str, timeframe: str = "1h"
    ) -> Optional[SymbolScore]:
        """단일 심볼을 평가해 SymbolScore를 반환한다."""
        df = await self._rest.fetch_ohlcv(
            symbol, timeframe, limit=self._LOOKBACK
        )
        if df is None or len(df) < self._RETURN_BARS + 10:
            logger.debug(f"[SymbolRanker] {symbol} 데이터 부족: {len(df) if df is not None else 0}개")
            return None

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # 1. 7일 수익률
        ret_7d = float(close.iloc[-1] / close.iloc[-self._RETURN_BARS] - 1)

        # 2. 트렌드 강도 (EMA 정렬 + ADX)
        ema_21 = close.ewm(span=21, adjust=False).mean()
        ema_55 = close.ewm(span=55, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean()
        current_close = float(close.iloc[-1])
        ema21 = float(ema_21.iloc[-1])
        ema55 = float(ema_55.iloc[-1])
        ema200 = float(ema_200.iloc[-1])

        # EMA 정렬 점수 (0~3)
        ema_align = 0
        if current_close > ema21:
            ema_align += 1
        if ema21 > ema55:
            ema_align += 1
        if ema55 > ema200:
            ema_align += 1
        ema_trend_score = ema_align / 3.0  # 0~1

        # ADX
        adx_val = _calc_adx(high, low, close, period=14)
        adx_score = min(adx_val / 50.0, 1.0)  # ADX 50 이상 = 최고점

        trend_score = 0.6 * ema_trend_score + 0.4 * adx_score

        # 3. 거래량 점수 (최근 7일 vs 30일 평균)
        vol_7d = float(volume.iloc[-self._RETURN_BARS:].mean())
        vol_30d = float(volume.iloc[-720:].mean()) if len(volume) >= 720 else float(volume.mean())
        volume_score = min(vol_7d / vol_30d, 2.0) / 2.0 if vol_30d > 0 else 0.0

        # 4. 변동성 점수 (ATR% 기반 — 0.02~0.06 범위가 적절, 너무 작거나 크면 페널티)
        atr = _calc_atr_simple(high, low, close, period=14)
        atr_pct = atr / current_close if current_close > 0 else 0.0
        # 최적 범위: 2~6% → 이 범위에서 점수 최고
        if 0.02 <= atr_pct <= 0.06:
            vol_score = 1.0
        elif atr_pct < 0.02:
            vol_score = atr_pct / 0.02  # 낮은 변동성 페널티
        else:
            vol_score = max(0, 1.0 - (atr_pct - 0.06) / 0.10)  # 과도한 변동성 페널티

        return SymbolScore(
            symbol=symbol,
            rank=0,  # 정렬 후 부여
            total_score=0.0,  # 정규화 후 계산
            return_7d=ret_7d,
            trend_score=trend_score,
            volume_score=volume_score,
            volatility_score=vol_score,
            atr_pct=atr_pct,
            adx=adx_val,
        )


# ── 유틸 함수 ──────────────────────────────────────────────

def _calc_atr_simple(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> float:
    """ATR 계산 (단순 버전)."""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return float(tr.ewm(span=period, adjust=False).mean().iloc[-1])


def _calc_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> float:
    """ADX 계산 (간이)."""
    try:
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)

        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, float("nan"))
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / atr.replace(0, float("nan"))

        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan")))
        adx = dx.ewm(span=period, adjust=False).mean()
        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 20.0
    except Exception:
        return 20.0


def _normalize_scores(scores: list[SymbolScore]) -> list[SymbolScore]:
    """점수를 0~100으로 정규화하고 가중합 계산."""
    if len(scores) <= 1:
        if scores:
            scores[0].total_score = 50.0
        return scores

    # 각 지표의 최소/최대 범위 계산
    returns = [s.return_7d for s in scores]
    ret_min, ret_max = min(returns), max(returns)
    ret_range = ret_max - ret_min

    for s in scores:
        # 수익률 정규화 (0~1)
        ret_norm = (s.return_7d - ret_min) / ret_range if ret_range > 0 else 0.5

        # 가중합
        raw = (
            _W_RETURN * ret_norm
            + _W_TREND * s.trend_score
            + _W_VOLUME * s.volume_score
            + _W_VOLATILITY * s.volatility_score
        )
        s.total_score = round(raw * 100, 1)

    return scores
