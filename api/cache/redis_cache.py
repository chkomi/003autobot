"""Redis 기반 TTL 캐시 구현 (plan v2 §3.4).

직렬화: JSON (pickle 대신 사용).
  - pickle은 외부에서 Redis에 악의적 데이터 주입 시 RCE 취약점 존재
  - ScoreService가 저장하는 값은 모두 JSON 직렬화 가능한 dict/list/float/str
  - JSON 직렬화 불가 타입(datetime 등)은 default=str 으로 안전하게 처리

키 네임스페이스 (plan v2 §7 리스크 7):
  - 모든 키에 prefix를 붙여 다른 앱과 충돌 방지
  - 기본 prefix: "autobot:"

사용 예:
    cache = RedisCache(redis_url="redis://localhost:6379/0")
    await cache.set("score:BTC/USDT:USDT", {...}, ttl=120.0)
    data = await cache.get("score:BTC/USDT:USDT")
    await cache.close()
"""
from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger

try:
    import redis.asyncio as aioredis  # redis-py >= 4.2
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False


class RedisCache:
    """redis-py async 기반 TTL 캐시.

    redis-py 가 설치되어 있지 않으면 ImportError 를 즉시 발생시킨다.
    런타임에 연결 실패가 발생하면 예외를 전파한다 (폴백은 상위에서 처리).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        prefix: str = "autobot:",
    ) -> None:
        if not _REDIS_AVAILABLE:
            raise ImportError(
                "redis-py 가 설치되어 있지 않습니다. `pip install redis` 로 설치하세요."
            )
        self._prefix = prefix
        self._client: aioredis.Redis = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,  # JSON 문자열 사용
        )

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        raw = await self._client.get(self._key(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception as exc:
            logger.warning(f"[RedisCache] 역직렬화 실패 key={key}: {exc}")
            await self._client.delete(self._key(key))  # 손상된 캐시 즉시 제거
            return None

    async def set(self, key: str, value: Any, ttl: float) -> None:
        # default=str: datetime 등 JSON 미지원 타입을 문자열로 안전하게 변환
        raw = json.dumps(value, ensure_ascii=False, default=str)
        await self._client.setex(self._key(key), int(ttl), raw)

    async def delete(self, key: str) -> None:
        await self._client.delete(self._key(key))

    async def close(self) -> None:
        await self._client.aclose()
        logger.debug("[RedisCache] 연결 닫힘")
