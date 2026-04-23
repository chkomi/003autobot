"""
알림 전송기 단위 테스트.
aiohttp 세션을 mock하여 네트워크 호출 없이 테스트한다.
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from notification.telegram_notifier import TelegramNotifier
from notification.discord_notifier import DiscordNotifier
from database.models import TradeRecord


class TestTelegramNotifier:

    @pytest.fixture
    def notifier(self):
        return TelegramNotifier(bot_token="test_token", chat_id="test_chat")

    @pytest.fixture
    def disabled_notifier(self):
        return TelegramNotifier(bot_token="", chat_id="")

    @pytest.mark.asyncio
    async def test_disabled_returns_false(self, disabled_notifier):
        result = await disabled_notifier.send("hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_success(self, notifier):
        mock_resp = MagicMock()
        mock_resp.status = 200

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_cm.__aexit__.return_value = False

        mock_session = MagicMock()
        mock_session.post.return_value = mock_cm
        mock_session.closed = False

        notifier._session = mock_session
        notifier._get_session = lambda: mock_session

        result = await notifier.send("test message")
        assert result is True
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_failure_status(self, notifier):
        mock_resp = MagicMock()
        mock_resp.status = 403
        mock_resp.text = AsyncMock(return_value="Forbidden")

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_cm.__aexit__.return_value = False

        mock_session = MagicMock()
        mock_session.post.return_value = mock_cm
        mock_session.closed = False

        notifier._session = mock_session
        notifier._get_session = lambda: mock_session

        result = await notifier.send("test")
        assert result is False

    @pytest.mark.asyncio
    async def test_close_session(self, notifier):
        mock_session = AsyncMock()
        mock_session.closed = False
        notifier._session = mock_session

        await notifier.close()
        mock_session.close.assert_called_once()
        assert notifier._session is None


class TestDiscordNotifier:

    @pytest.fixture
    def notifier(self):
        return DiscordNotifier(webhook_url="https://discord.com/api/webhooks/test")

    @pytest.fixture
    def disabled_notifier(self):
        return DiscordNotifier(webhook_url="")

    @pytest.mark.asyncio
    async def test_disabled_returns_false(self, disabled_notifier):
        result = await disabled_notifier.send("hello")
        assert result is False

    @pytest.mark.asyncio
    async def test_send_success(self, notifier):
        mock_resp = MagicMock()
        mock_resp.status = 204

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_cm.__aexit__.return_value = False

        mock_session = MagicMock()
        mock_session.post.return_value = mock_cm
        mock_session.closed = False

        notifier._session = mock_session
        notifier._get_session = lambda: mock_session

        result = await notifier.send("test")
        assert result is True

    @pytest.mark.asyncio
    async def test_send_embed_success(self, notifier):
        mock_resp = MagicMock()
        mock_resp.status = 200

        mock_cm = AsyncMock()
        mock_cm.__aenter__.return_value = mock_resp
        mock_cm.__aexit__.return_value = False

        mock_session = MagicMock()
        mock_session.post.return_value = mock_cm
        mock_session.closed = False

        notifier._session = mock_session
        notifier._get_session = lambda: mock_session

        result = await notifier.send_embed("Title", "Description", 0xFF0000)
        assert result is True

    @pytest.mark.asyncio
    async def test_close_session(self, notifier):
        mock_session = AsyncMock()
        mock_session.closed = False
        notifier._session = mock_session

        await notifier.close()
        mock_session.close.assert_called_once()
