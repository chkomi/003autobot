"""Autobot REST API 진입점.

실행 예:
    uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

환경변수:
    AUTOBOT_API_DATABASE_URL  PostgreSQL URL (없으면 SQLite 폴백)
    AUTOBOT_API_REDIS_URL     Redis URL    (없으면 MemoryCache 폴백)
"""
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.cache.base import CacheBackend
from api.cache.memory_cache import MemoryCache
from api.config import get_settings
from api.db.base import create_all_tables, get_engine
from api.deps import verify_api_key
from api.routers import backtests, market, ops, reports
from api.services.backtest_service import BacktestService
from api.services.score_service import ScoreService


def _build_cache(redis_url: Optional[str]) -> CacheBackend:
    """REDIS_URL 이 있으면 RedisCache, 없으면 MemoryCache를 반환한다."""
    if redis_url:
        try:
            from api.cache.redis_cache import RedisCache
            cache = RedisCache(redis_url=redis_url)
            logger.info(f"캐시 백엔드: Redis ({redis_url})")
            return cache
        except Exception as exc:
            logger.warning(f"Redis 연결 실패 — MemoryCache로 폴백: {exc}")
    logger.info("캐시 백엔드: MemoryCache (in-process)")
    return MemoryCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info(
        f"[{settings.app_name}] startup env={settings.env} "
        f"host={settings.host}:{settings.port}"
    )

    # ── DB 초기화 (phase3) ───────────────────────────────────────
    if settings.database_url:
        logger.info(f"DB: PostgreSQL ({settings.database_url.split('@')[-1]})")
    else:
        logger.info("DB: SQLite 폴백 (AUTOBOT_API_DATABASE_URL 미설정)")
        # dev 환경에서는 테이블 자동 생성
        if settings.env == "dev":
            try:
                await create_all_tables()
                logger.info("DB 테이블 자동 생성 완료 (dev 모드)")
            except Exception as exc:
                logger.warning(f"DB 테이블 생성 실패 (무시): {exc}")

    # ── Redis / MemoryCache 초기화 (phase3) ──────────────────────
    cache = _build_cache(settings.redis_url)
    app.state.cache = cache

    # ── ScoreService 초기화 (phase2 + cache 주입) ────────────────
    svc = ScoreService(cache=cache)
    app.state.score_svc = svc
    logger.info(f"[{settings.app_name}] ScoreService 초기화 완료 — 심볼: {svc.symbols}")

    # ── BacktestService 초기화 (phase4) ──────────────────────────
    svc_bt = BacktestService(redis_url=settings.redis_url)
    app.state.backtest_svc = svc_bt
    logger.info(f"[{settings.app_name}] BacktestService 초기화 완료")

    yield

    # ── 정리 ────────────────────────────────────────────────────
    await svc.close()           # ccxt + 캐시 연결 닫기
    engine = get_engine()
    await engine.dispose()      # DB 커넥션 풀 해제
    logger.info(f"[{settings.app_name}] shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Autobot API",
        version="0.1.0",
        description="OKX 기반 트레이딩 분석 플랫폼 REST API (plan v2)",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    api_v1 = "/api/v1"
    # ── Fix #6: 모든 API 라우터에 X-API-Key 인증 적용 ─────────────
    # AUTOBOT_API_API_KEY 미설정 시 인증 비활성화(dev 모드).
    # ops(health) 는 모니터링 시스템 접근을 위해 인증 제외.
    _auth = [Depends(verify_api_key)]
    app.include_router(market.router, prefix=api_v1, dependencies=_auth)
    app.include_router(backtests.router, prefix=api_v1, dependencies=_auth)
    app.include_router(reports.router, prefix=api_v1, dependencies=_auth)
    app.include_router(ops.router, prefix=api_v1)   # health: 인증 제외

    return app


app = create_app()
