"""캐시 추상화 프로토콜 (plan v2 §3.4).

ScoreService 등 캐시 사용자는 이 인터페이스에만 의존한다.
구현체:
  - MemoryCache  : 인메모리 TTL dict (개발/폴백용)
  - RedisCache   : Redis TTL (프로덕션용)
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class CacheBackend(Protocol):
    """캐시 백엔드 인터페이스."""

    async def get(self, key: str) -> Optional[Any]:
        """키에 해당하는 값을 반환한다. 없거나 만료되면 None."""
        ...

    async def set(self, key: str, value: Any, ttl: float) -> None:
        """키-값을 TTL 초 동안 저장한다."""
        ...

    async def delete(self, key: str) -> None:
        """키를 삭제한다. 없어도 예외를 발생시키지 않는다."""
        ...

    async def close(self) -> None:
        """연결/리소스를 해제한다."""
        ...
