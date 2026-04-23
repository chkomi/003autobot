"""
알림 라우팅 관리자.
텔레그램 + Discord 알림 동시 전송 지원.
"""
import asyncio

from loguru import logger

from config.settings import Settings
from database.models import TradeRecord
from notification.discord_notifier import DiscordNotifier
from notification.telegram_notifier import TelegramNotifier


class NotificationManager:
    def __init__(self, settings: Settings):
        self._telegram = TelegramNotifier(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )
        self._discord = DiscordNotifier(
            webhook_url=settings.discord_webhook_url,
        )

    async def close(self) -> None:
        """알림 전송기 세션 정리"""
        await asyncio.gather(
            self._telegram.close(),
            self._discord.close(),
            return_exceptions=True,
        )

    @property
    def telegram(self) -> TelegramNotifier:
        return self._telegram

    @property
    def discord(self) -> DiscordNotifier:
        return self._discord

    async def on_trade_opened(self, trade: TradeRecord, current_price: float) -> None:
        await asyncio.gather(
            self._telegram.send_trade_opened(trade, current_price),
            self._discord.send_trade_opened(trade, current_price),
            return_exceptions=True,
        )
        logger.info(f"[알림] 포지션 오픈 — {trade.trade_id} {trade.direction}")

    async def on_trade_closed(self, trade: TradeRecord) -> None:
        await asyncio.gather(
            self._telegram.send_trade_closed(trade),
            self._discord.send_trade_closed(trade),
            return_exceptions=True,
        )
        logger.info(
            f"[알림] 포지션 청산 — {trade.trade_id} "
            f"P&L: {trade.pnl_usdt:.2f} USDT ({trade.exit_reason})"
        )

    async def on_daily_summary(
        self,
        date: str,
        pnl_usdt: float,
        trade_count: int,
        win_rate: float,
        equity: float,
    ) -> None:
        await asyncio.gather(
            self._telegram.send_daily_summary(date, pnl_usdt, trade_count, win_rate, equity),
            self._discord.send_daily_summary(date, pnl_usdt, trade_count, win_rate, equity),
            return_exceptions=True,
        )

    async def on_alert(self, level: str, message: str) -> None:
        logger.log(level, f"[알림] {message}")
        await asyncio.gather(
            self._telegram.send_alert(level, message),
            self._discord.send_alert(level, message),
            return_exceptions=True,
        )

    async def on_halt(self, reason: str) -> None:
        logger.critical(f"[봇 정지] {reason}")
        await asyncio.gather(
            self._telegram.send_halt_notification(reason),
            self._discord.send_halt_notification(reason),
            return_exceptions=True,
        )

    async def on_startup(self, symbol: str, leverage: int, is_demo: bool) -> None:
        await asyncio.gather(
            self._telegram.send_startup(symbol, leverage, is_demo),
            self._discord.send_startup(symbol, leverage, is_demo),
            return_exceptions=True,
        )
