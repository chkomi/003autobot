"""운영 상태 점검 엔드포인트 (plan v2 §5.4)."""
from datetime import datetime

from fastapi import APIRouter

from api.schemas.common import Envelope
from api.schemas.reports import OpsHealth

router = APIRouter(prefix="/ops", tags=["ops"])


@router.get("/health", response_model=Envelope[OpsHealth])
async def health():
    """Liveness/Readiness 공통 엔드포인트.
    심층 의존성 검사는 Phase 4에서 DB/Redis/WS 체크로 확장한다.
    """
    return Envelope(
        data=OpsHealth(
            status="ok",
            notes=["skeleton stage — deep checks pending phase4"],
        ),
        generated_at=datetime.utcnow(),
    )
