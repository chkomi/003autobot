"""FastAPI 의존성 주입 헬퍼 (plan v2 §3.2).

사용법:
    from api.deps import get_score_svc, verify_api_key

    @router.get("/leaderboard")
    async def leaderboard(svc: ScoreService = Depends(get_score_svc)):
        ...
"""
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security.api_key import APIKeyHeader

from api.config import get_settings
from api.services.score_service import ScoreService

# ── API Key 인증 ─────────────────────────────────────────────────

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: Optional[str] = Security(_API_KEY_HEADER)) -> None:
    """X-API-Key 헤더를 검증하는 FastAPI 의존성.

    AUTOBOT_API_API_KEY 환경변수가 설정되어 있을 때만 검증을 수행한다.
    미설정 시(로컬 개발 환경)에는 인증을 건너뛴다.

    사용 예 (router include 시):
        app.include_router(router, dependencies=[Depends(verify_api_key)])
    """
    settings = get_settings()
    if not settings.api_key:
        # 환경변수 미설정 = 인증 비활성화 (dev 모드)
        return
    if not key or key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key 헤더가 없거나 올바르지 않습니다.",
            headers={"WWW-Authenticate": "ApiKey"},
        )


# ── 서비스 의존성 ────────────────────────────────────────────────

def get_score_svc(request: Request) -> ScoreService:
    """app.state에서 ScoreService 싱글턴을 꺼낸다.

    lifespan에서 ``app.state.score_svc``로 등록되어 있어야 한다.
    """
    return request.app.state.score_svc


# 타입 별칭 — 라우터에서 간결하게 사용
ScoreSvcDep = Annotated[ScoreService, Depends(get_score_svc)]
