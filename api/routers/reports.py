"""리포트 관련 엔드포인트 (plan v2 §5.3)."""
from fastapi import APIRouter, HTTPException, Query

from api.schemas.common import Envelope
from api.schemas.reports import ReportOverview

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/overview", response_model=Envelope[ReportOverview])
async def get_overview(
    period: str = Query("daily", pattern="^(daily|weekly|monthly)$"),
):
    raise HTTPException(status_code=501, detail="not_implemented: phase4")
