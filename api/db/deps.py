"""DB 세션 FastAPI 의존성 (plan v2 §3.2).

사용법:
    from api.db.deps import DbSession

    @router.get("/something")
    async def handler(db: DbSession):
        result = await db.execute(select(Trade))
        ...
"""
from typing import Annotated, AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.base import get_session_factory


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """요청마다 새 DB 세션을 열고, 완료/예외 시 자동 닫는다."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# 타입 별칭 — 라우터에서 간결하게 사용
DbSession = Annotated[AsyncSession, Depends(get_db)]
