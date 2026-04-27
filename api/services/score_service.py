"""
점수 계산 서비스 (plan v2 §19 – 3단계 업데이트).

OKX REST API에서 OHLCV를 가져와 ScoreEngine으로 평가하고,
결과를 CacheBackend(Redis or MemoryCache)에 보관한다.

캐시 TTL:
  - 개별 심볼 점수  : SYMBOL_CACHE_TTL (기본 120s)
  - 리더보드        : LEADERBOARD_CACHE_TTL (기본 60s)

Phase 3부터 Redis 캐시가 주(primary), MemoryCache가 폴백으로 동작한다.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from api.cache.base import CacheBackend
from api.cache.memory_cache import MemoryCache
from config.settings import OKXConfig, TradingConfig
from config.strategy_params import MacroParams, ScoringParams, SentimentParams
from data.okx_rest_client import OKXRestClient
from strategy.score_engine import ScoreEngine

# ── 캐시 TTL 상수 ────────────────────────────────────────────────
SYMBOL_CACHE_TTL: float = 120.0       # 개별 심볼 점수 캐시 (초)
LEADERBOARD_CACHE_TTL: float = 60.0   # 리더보드 캐시 (초)

# OKX API rate-limit 보호: 동시 최대 심볼 수
_CONCURRENCY_LIMIT = 5


# ── 메인 서비스 ──────────────────────────────────────────────────

class ScoreService:
    """종목별 롱/숏 점수 계산 및 리더보드 집계를 담당한다.

    앱 lifespan에서 단일 인스턴스로 생성해 `app.state.score_svc`에 보관한다.
    ScoreEngine과 OKXRestClient는 최초 호출 시 지연(lazy) 초기화된다.

    캐시 백엔드는 외부에서 주입할 수 있다.
    주입하지 않으면 MemoryCache를 사용한다 (plan v2 §14 로컬 개발).
    """

    def __init__(self, cache: Optional[CacheBackend] = None) -> None:
        self._okx_cfg = OKXConfig()
        self._trading_cfg = TradingConfig()
        self._client: Optional[OKXRestClient] = None
        self._engine: Optional[ScoreEngine] = None
        self._cache: CacheBackend = cache or MemoryCache()
        self._sem = asyncio.Semaphore(_CONCURRENCY_LIMIT)

    # ── 프로퍼티 ─────────────────────────────────────────────────

    @property
    def symbols(self) -> list[str]:
        """거래 설정에서 읽은 모니터링 심볼 목록."""
        return self._trading_cfg.symbol_list

    # ── Lazy 초기화 ──────────────────────────────────────────────

    def _client_(self) -> OKXRestClient:
        """OKX REST 클라이언트를 최초 호출 시 초기화한다.

        Fix #12: 초기화 실패 시 None을 반환하지 않고 RuntimeError를 즉시 raise.
        이후 호출에서 AttributeError가 발생하는 것보다 명확한 오류를 제공한다.
        """
        if self._client is None:
            try:
                self._client = OKXRestClient(self._okx_cfg)
            except Exception as exc:
                raise RuntimeError(
                    f"OKX 클라이언트 초기화 실패: {exc}. "
                    "OKX_API_KEY / OKX_API_SECRET 설정을 확인하세요."
                ) from exc
        return self._client

    def _engine_(self) -> ScoreEngine:
        """ScoreEngine을 최초 호출 시 초기화한다.

        strategy_params.yaml 은 lru_cache 덕분에 최초 1회만 파일 I/O를 수행한다.
        """
        if self._engine is None:
            try:
                self._engine = ScoreEngine(
                    scoring=ScoringParams.from_yaml(),
                    sentiment_params=SentimentParams.from_yaml(),
                    macro_params=MacroParams.from_yaml(),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"ScoreEngine 초기화 실패: {exc}"
                ) from exc
        return self._engine

    # ── 점수 계산 ─────────────────────────────────────────────────

    async def score_symbol(self, symbol: str) -> dict:
        """단일 심볼의 롱/숏 점수를 계산해 dict로 반환한다.

        결과는 SYMBOL_CACHE_TTL 동안 캐시된다.
        캐시 히트 여부는 반환 dict의 ``cache_hit`` 필드로 확인할 수 있다.
        """
        cache_key = f"score:{symbol}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return {**cached, "cache_hit": True}

        async with self._sem:
            # 세마포어 대기 중 다른 코루틴이 캐시를 채웠을 수 있음
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return {**cached, "cache_hit": True}

            result = await self._compute_score(symbol)

        await self._cache.set(cache_key, result, SYMBOL_CACHE_TTL)
        return {**result, "cache_hit": False}

    async def _compute_score(self, symbol: str) -> dict:
        """실제 OHLCV 조회 + ScoreEngine 평가. 캐시 없이 항상 신규 계산."""
        client = self._client_()
        engine = self._engine_()

        # 세 타임프레임 병렬 조회
        df_4h, df_1h, df_15m = await asyncio.gather(
            client.fetch_ohlcv(symbol, "4h", limit=300),
            client.fetch_ohlcv(symbol, "1h", limit=300),
            client.fetch_ohlcv(symbol, "15m", limit=300),
        )

        breakdown = engine.evaluate(df_4h, df_1h, df_15m)

        # ── 롱/숏 점수 도출 ──────────────────────────────────────
        # ScoreEngine.total 은 감지된 방향의 강도 (0~100).
        # LONG  : long_score = total, short_score = 100 - total
        # SHORT : short_score = total, long_score = 100 - total
        # NEUTRAL: 양쪽 모두 50
        direction = breakdown.direction
        total = round(breakdown.total, 1)

        if direction == "LONG":
            long_score = total
            short_score = round(100.0 - total, 1)
        elif direction == "SHORT":
            short_score = total
            long_score = round(100.0 - total, 1)
        else:  # NEUTRAL
            long_score = short_score = 50.0

        # neutral: 50에 가까울수록 중립도 높음 (0~100)
        neutral = round(max(0.0, 100.0 - 2.0 * abs(long_score - 50.0)), 1)

        # 현재가 / 24h 변동률 (ticker)
        last_price: Optional[float] = None
        change_24h: Optional[float] = None
        try:
            ticker = await client.fetch_ticker(symbol)
            last_price = ticker.get("last")
            change_24h = ticker.get("percentage")
        except Exception as exc:
            logger.warning(f"[score_service] {symbol} ticker 조회 실패: {exc}")

        # 점수 근거 텍스트 (rationale)
        rationale = _build_rationale(breakdown)

        return {
            "symbol": symbol,
            "long_score": long_score,
            "short_score": short_score,
            "neutral": neutral,
            "direction": direction,
            "signal_strength": breakdown.signal_strength,
            "breakdown": {
                "trend": round(breakdown.trend, 1),
                "momentum": round(breakdown.momentum, 1),
                "volume": round(breakdown.volume, 1),
                "volatility": round(breakdown.volatility, 1),
                "sentiment": round(breakdown.sentiment, 1),
                "macro": round(breakdown.macro, 1),
            },
            "last_price": last_price,
            "change_24h": change_24h,
            "calculated_at": datetime.now(timezone.utc).isoformat(),
            "rationale": rationale,
        }

    # ── 리더보드 ──────────────────────────────────────────────────

    async def leaderboard(self, direction: str = "LONG", limit: int = 50) -> list[dict]:
        """모니터링 심볼 전체의 점수를 계산해 방향별로 정렬한 리더보드를 반환한다."""
        cache_key = f"leaderboard:{direction}:{limit}"
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached

        # 심볼별 점수 병렬 계산 (세마포어는 score_symbol 내부에서 처리)
        tasks = [self.score_symbol(sym) for sym in self.symbols]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        entries: list[dict] = []
        for sym, res in zip(self.symbols, raw_results):
            if isinstance(res, Exception):
                logger.warning(f"[leaderboard] {sym} 스코어 계산 오류, 건너뜀: {res}")
                continue
            entries.append(res)

        # 방향 기준 정렬
        sort_key = "long_score" if direction == "LONG" else "short_score"
        entries.sort(key=lambda x: x.get(sort_key, 0.0), reverse=True)

        # 순위 부여
        result = []
        for rank, entry in enumerate(entries[:limit], start=1):
            result.append({**entry, "rank": rank})

        await self._cache.set(cache_key, result, LEADERBOARD_CACHE_TTL)
        return result

    # ── 정리 ─────────────────────────────────────────────────────

    async def close(self) -> None:
        """앱 종료 시 ccxt 세션 및 캐시 연결을 닫는다."""
        if self._client is not None:
            try:
                await self._client._exchange.close()
            except Exception:
                pass
        try:
            await self._cache.close()
        except Exception:
            pass


# ── 점수 근거 텍스트 빌더 ────────────────────────────────────────

def _build_rationale(breakdown) -> list[str]:
    """ScoreBreakdown에서 사람이 읽을 수 있는 근거 텍스트를 생성한다."""
    lines: list[str] = []
    details: dict = breakdown.details or {}

    # 추세
    trend_d = details.get("trend", {})
    ema = trend_d.get("ema_cross", "")
    if ema:
        lines.append(f"추세: {ema} — 추세 점수 {breakdown.trend:.0f}점")

    # 모멘텀
    mom_d = details.get("momentum", {})
    macd = mom_d.get("macd", "")
    if macd:
        lines.append(f"모멘텀(MACD): {macd} — 모멘텀 점수 {breakdown.momentum:.0f}점")

    # 볼륨
    vol_d = details.get("volume", {})
    obv = vol_d.get("obv_trend", "")
    if obv:
        lines.append(f"볼륨(OBV): {obv} — 볼륨 점수 {breakdown.volume:.0f}점")

    # 변동성
    lines.append(f"변동성 점수 {breakdown.volatility:.0f}점")

    # 심리
    sent_d = details.get("sentiment", {})
    fr = sent_d.get("funding_rate", "")
    if fr:
        lines.append(f"심리(펀딩레이트): {fr} — 심리 점수 {breakdown.sentiment:.0f}점")

    # 매크로
    macro_d = details.get("macro", {})
    fg = macro_d.get("fear_greed", "")
    if fg:
        lines.append(f"매크로(F&G): {fg} — 매크로 점수 {breakdown.macro:.0f}점")

    # 최종 요약
    lines.append(
        f"종합 점수 {breakdown.total:.1f}점 ({breakdown.signal_strength}) — 방향: {breakdown.direction}"
    )
    return lines
