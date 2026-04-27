"""SQLAlchemy 비동기 엔진 및 세션 팩토리 (plan v2 §3.4).

환경변수 AUTOBOT_API_DATABASE_URL 이 있으면 PostgreSQL 을 사용하고,
없으면 기존 SQLite(aiosqlite)로 폴백한다. 이 덕분에 Phase 3 이전에도
기존 봇과 동일한 DB 파일을 읽을 수 있다.

PostgreSQL URL 형식: postgresql+asyncpg://user:pass@host:5432/dbname
SQLite URL 형식   : sqlite+aiosqlite:///data_cache/trading.db  (기본값)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from api.config import get_settings

# ── ORM 베이스 ────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """모든 ORM 모델이 상속하는 선언적 베이스."""
    pass


# ── 엔진 팩토리 ──────────────────────────────────────────────────

def _make_engine(db_url: str | None) -> AsyncEngine:
    """DB URL에 맞는 비동기 엔진을 생성한다."""
    if not db_url:
        # 기본값: 기존 SQLite 파일 (봇과 공유)
        sqlite_path = Path(__file__).parents[2] / "data_cache" / "trading.db"
        db_url = f"sqlite+aiosqlite:///{sqlite_path}"

    is_sqlite = db_url.startswith("sqlite")
    kwargs: dict = {}

    if not is_sqlite:
        # PostgreSQL — 커넥션 풀 설정
        kwargs = {
            "pool_size": 5,
            "max_overflow": 10,
            "pool_pre_ping": True,
            "pool_recycle": 1800,
        }

    return create_async_engine(db_url, echo=False, **kwargs)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """앱 전역 싱글턴 엔진을 반환한다."""
    settings = get_settings()
    return _make_engine(settings.database_url)


@lru_cache(maxsize=1)
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """앱 전역 세션 팩토리를 반환한다."""
    engine = get_engine()
    return async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


async def create_all_tables() -> None:
    """개발/테스트 환경에서 테이블을 직접 생성한다.

    프로덕션에서는 ``alembic upgrade head`` 를 사용할 것.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
