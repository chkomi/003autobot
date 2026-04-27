"""인메모리 TTL 캐시 — Redis 없는 환경의 폴백 구현 (plan v2 §3.4).

스레드 안전: asyncio 단일 스레드 내에서만 사용. 멀티프로세스 환경에서는
각 프로세스가 독립적인 캐시를 가지므로 Redis로 교체해야 한다.
"""
from __future__ import annotations

import time
from typing import Any, Optional


class _Entry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, ttl: float) -> None:
        self.value = value
        self.expires_at = time.monotonic() + ttl


class MemoryCache:
    """프로세스 내 인메모리 TTL 캐시."""

    def __init__(self) -> None:
        self._store: dict[str, _Entry] = {}

    async def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.monotonic() >= entry.expires_at:
            del self._store[key]
            return None
        return entry.value

    async def set(self, key: str, value: Any, ttl: float) -> None:
        self._store[key] = _Entry(value, ttl)

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def close(self) -> None:
        self._store.clear()
