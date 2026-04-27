"""backtest_job start_date/end_date 컬럼 타입 VARCHAR → DATE 변환.

Revision ID: c7d8e9f0a1b2
Revises: 808af8583dee
Create Date: 2026-04-24

변경 내용:
  backtest_jobs.start_date  VARCHAR(10) → DATE
  backtest_jobs.end_date    VARCHAR(10) → DATE

SQLite: batch_alter_table 으로 테이블 재생성 (ALTER COLUMN 미지원)
PostgreSQL: USING start_date::DATE 캐스트로 기존 데이터 변환
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "c7d8e9f0a1b2"
down_revision = "808af8583dee"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "sqlite":
        # SQLite: batch mode로 테이블을 재생성해 컬럼 타입 변경
        # 기존 "YYYY-MM-DD" 문자열 값은 SQLite의 DATE 친화적 TEXT 처리 덕분에
        # date 조회 시 자동으로 Python date 객체로 변환됨
        with op.batch_alter_table("backtest_jobs") as batch_op:
            batch_op.alter_column(
                "start_date",
                existing_type=sa.String(10),
                type_=sa.Date(),
                existing_nullable=False,
            )
            batch_op.alter_column(
                "end_date",
                existing_type=sa.String(10),
                type_=sa.Date(),
                existing_nullable=False,
            )
    else:
        # PostgreSQL: USING 절로 기존 VARCHAR 값을 DATE로 캐스트
        op.execute(
            "ALTER TABLE backtest_jobs "
            "ALTER COLUMN start_date TYPE DATE USING start_date::DATE"
        )
        op.execute(
            "ALTER TABLE backtest_jobs "
            "ALTER COLUMN end_date TYPE DATE USING end_date::DATE"
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "sqlite":
        with op.batch_alter_table("backtest_jobs") as batch_op:
            batch_op.alter_column(
                "start_date",
                existing_type=sa.Date(),
                type_=sa.String(10),
                existing_nullable=False,
            )
            batch_op.alter_column(
                "end_date",
                existing_type=sa.Date(),
                type_=sa.String(10),
                existing_nullable=False,
            )
    else:
        op.execute(
            "ALTER TABLE backtest_jobs "
            "ALTER COLUMN start_date TYPE VARCHAR(10) USING start_date::TEXT"
        )
        op.execute(
            "ALTER TABLE backtest_jobs "
            "ALTER COLUMN end_date TYPE VARCHAR(10) USING end_date::TEXT"
        )
