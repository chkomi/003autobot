"""Alembic 환경 설정 — 비동기(asyncpg) 지원 (plan v2 §3).

실행 예:
    # 마이그레이션 생성
    alembic revision --autogenerate -m "initial schema"

    # 마이그레이션 적용
    alembic upgrade head

    # SQL 미리보기 (dry-run)
    alembic upgrade head --sql

DB URL 우선순위:
    1. 환경변수 AUTOBOT_API_DATABASE_URL
    2. alembic.ini 의 sqlalchemy.url
"""
from __future__ import annotations

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────
_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── 모델 임포트 (autogenerate가 메타데이터를 인식하도록) ──────────
from api.db.base import Base   # noqa: E402
from api.db import models      # noqa: F401, E402 — 모델 등록 트리거

# ── Alembic Config 객체 ──────────────────────────────────────────
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


# ── DB URL 해석 ──────────────────────────────────────────────────

def _get_db_url() -> str:
    """환경변수 → alembic.ini 순서로 DB URL을 결정한다."""
    env_url = os.environ.get("AUTOBOT_API_DATABASE_URL")
    if env_url:
        return env_url
    return config.get_main_option("sqlalchemy.url")


# ── 오프라인 마이그레이션 (SQL 출력 모드) ────────────────────────

def run_migrations_offline() -> None:
    """DB 연결 없이 SQL 문을 stdout으로 출력한다."""
    url = _get_db_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


# ── 온라인 마이그레이션 (실제 DB 적용) ──────────────────────────

def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
        # SQLite는 ALTER COLUMN을 지원하지 않으므로 batch 모드 활성화
        # PostgreSQL에서는 무시됨
        render_as_batch=True,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """비동기 엔진으로 마이그레이션을 실행한다."""
    url = _get_db_url()
    connectable = create_async_engine(url, echo=False)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


# ── 진입점 ───────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
