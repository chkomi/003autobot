"""시장/점수 관련 엔드포인트 (plan v2 §3.2 — 2단계 구현).

연결된 서비스:
  - ScoreService → strategy/score_engine.py + data/okx_rest_client.py
  - 인메모리 TTL 캐시 (leaderboard 60s / symbol 120s)

Phase 3 이후 Redis 캐시로 교체 예정.
"""
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Path, Query
from loguru import logger

from api.deps import ScoreSvcDep
from api.schemas.common import Envelope
from api.schemas.market import LeaderboardEntry, ScoreBreakdown, SymbolDetail, SymbolScore

router = APIRouter(prefix="/market", tags=["market"])


# ── 리더보드 ─────────────────────────────────────────────────────

@router.get(
    "/leaderboard",
    response_model=Envelope[list[LeaderboardEntry]],
    summary="전 종목 롱/숏 점수 랭킹",
    description=(
        "모니터링 심볼 전체를 대상으로 점수를 계산해 지정 방향(LONG/SHORT) 기준 정렬한다. "
        "결과는 60초 캐시된다 (plan v2 §5.1)."
    ),
)
async def get_leaderboard(
    svc: ScoreSvcDep,
    direction: str = Query("LONG", pattern="^(LONG|SHORT)$", description="정렬 기준 방향"),
    limit: int = Query(50, ge=1, le=500, description="반환할 최대 항목 수"),
):
    try:
        entries = await svc.leaderboard(direction=direction, limit=limit)
    except Exception as exc:
        logger.exception(f"[leaderboard] 오류: {exc}")
        raise HTTPException(status_code=502, detail=f"점수 계산 중 오류: {exc}") from exc

    data = [
        LeaderboardEntry(
            rank=e["rank"],
            symbol=e["symbol"],
            long_score=e["long_score"],
            short_score=e["short_score"],
            direction=e["direction"],
            last_price=e.get("last_price"),
            change_24h=e.get("change_24h"),
        )
        for e in entries
    ]
    return Envelope(
        data=data,
        generated_at=datetime.now(timezone.utc),
        cache_hit=bool(entries and entries[0].get("cache_hit")),
    )


# ── 심볼 점수 ────────────────────────────────────────────────────

@router.get(
    "/symbols/{symbol}/score",
    response_model=Envelope[SymbolScore],
    summary="개별 심볼 롱/숏 점수",
    description=(
        "지정 심볼의 6개 차원 점수(추세/모멘텀/볼륨/변동성/심리/매크로)와 "
        "최종 롱·숏 점수를 반환한다. 결과는 120초 캐시된다."
    ),
)
async def get_symbol_score(
    svc: ScoreSvcDep,
    symbol: str = Path(..., min_length=1, description="ccxt 형식 심볼 (예: BTC/USDT:USDT)"),
):
    try:
        result = await svc.score_symbol(symbol)
    except Exception as exc:
        logger.exception(f"[symbol_score] {symbol} 오류: {exc}")
        raise HTTPException(status_code=502, detail=f"점수 계산 중 오류: {exc}") from exc

    data = SymbolScore(
        symbol=result["symbol"],
        long_score=result["long_score"],
        short_score=result["short_score"],
        neutral=result["neutral"],
        direction=result["direction"],
        signal_strength=result["signal_strength"],
        breakdown=ScoreBreakdown(**result["breakdown"]),
        calculated_at=datetime.fromisoformat(result["calculated_at"]),
    )
    return Envelope(
        data=data,
        generated_at=datetime.now(timezone.utc),
        cache_hit=result.get("cache_hit", False),
    )


# ── 심볼 상세 ────────────────────────────────────────────────────

@router.get(
    "/symbols/{symbol}",
    response_model=Envelope[SymbolDetail],
    summary="개별 심볼 상세 분석",
    description="점수 정보 + 점수 근거 텍스트(rationale)를 함께 반환한다.",
)
async def get_symbol_detail(
    svc: ScoreSvcDep,
    symbol: str = Path(..., min_length=1, description="ccxt 형식 심볼 (예: BTC/USDT:USDT)"),
):
    try:
        result = await svc.score_symbol(symbol)
    except Exception as exc:
        logger.exception(f"[symbol_detail] {symbol} 오류: {exc}")
        raise HTTPException(status_code=502, detail=f"점수 계산 중 오류: {exc}") from exc

    score = SymbolScore(
        symbol=result["symbol"],
        long_score=result["long_score"],
        short_score=result["short_score"],
        neutral=result["neutral"],
        direction=result["direction"],
        signal_strength=result["signal_strength"],
        breakdown=ScoreBreakdown(**result["breakdown"]),
        calculated_at=datetime.fromisoformat(result["calculated_at"]),
    )
    data = SymbolDetail(
        symbol=result["symbol"],
        score=score,
        rationale=result.get("rationale", []),
    )
    return Envelope(
        data=data,
        generated_at=datetime.now(timezone.utc),
        cache_hit=result.get("cache_hit", False),
    )
