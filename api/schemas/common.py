"""공통 응답 스키마."""
from datetime import datetime
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class Envelope(BaseModel, Generic[T]):
    """표준 응답 래퍼. 모든 엔드포인트는 이 구조를 반환하는 것을 권장한다."""
    data: T
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    cache_hit: Optional[bool] = None
