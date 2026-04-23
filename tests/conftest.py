"""
테스트 공통 fixture.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from config.settings import Settings, OKXConfig, TradingConfig, RiskConfig
from database.db_manager import DatabaseManager


@pytest.fixture
def mock_settings():
    """기본 Settings mock"""
    s = MagicMock(spec=Settings)
    s.okx = OKXConfig()
    s.trading = TradingConfig()
    s.risk = RiskConfig()
    s.telegram_bot_token = "test_token"
    s.telegram_chat_id = "test_chat"
    s.discord_webhook_url = "https://discord.com/api/webhooks/test"
    s.telegram_configured = True
    s.discord_configured = True
    s.log_level = "DEBUG"
    s.db_path = ":memory:"
    return s


@pytest_asyncio.fixture
async def in_memory_db(tmp_path):
    """인메모리 SQLite DB"""
    db = DatabaseManager(tmp_path / "test.db")
    await db.initialize()
    yield db
    await db.close()
