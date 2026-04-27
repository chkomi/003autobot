"""API 런타임 설정.

.env / 환경변수에서 값을 읽는다. 민감정보(OKX 키 등)는 여기서 읽지 않고
Worker 런타임에만 주입한다 (plan v2 §11).
"""
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="AUTOBOT_API_",
        extra="ignore",
    )

    app_name: str = "autobot-api"
    env: str = Field(default="dev", description="dev | staging | prod")
    host: str = "0.0.0.0"
    port: int = 8080

    # Vercel 프론트엔드 origin 허용 (쉼표 구분)
    cors_allow_origins: str = "http://localhost:3000"

    database_url: Optional[str] = None  # postgresql+psycopg://...
    redis_url: Optional[str] = None     # redis://...

    # API Key 인증 (미설정 시 인증 비활성화 — 로컬 개발용)
    # 프로덕션에서는 반드시 AUTOBOT_API_API_KEY 환경변수로 설정할 것
    api_key: Optional[str] = Field(
        default=None,
        description="X-API-Key 인증 키. 미설정 시 모든 요청 허용 (dev 전용).",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allow_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()
