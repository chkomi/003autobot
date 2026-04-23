"""
Discord Webhook 알림 전송.
거래 진입/청산, 일간 요약, 경고, 봇 정지 등을 Discord 채널로 전송한다.
aiohttp 세션을 재사용하여 connection pool을 활용한다.
"""
import asyncio
from typing import Optional

import aiohttp
from loguru import logger

from database.models import TradeRecord


class DiscordNotifier:
    """Discord Webhook 알림 전송기"""

    def __init__(self, webhook_url: str):
        self._webhook_url = webhook_url
        self._enabled = bool(webhook_url)
        self._session: Optional[aiohttp.ClientSession] = None
        if not self._enabled:
            logger.warning("Discord Webhook URL이 설정되지 않음 — Discord 알림 비활성화")

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def send(self, content: str) -> bool:
        """Discord Webhook으로 메시지 전송."""
        if not self._enabled:
            return False
        payload = {"content": content}
        try:
            session = self._get_session()
            async with session.post(self._webhook_url, json=payload) as resp:
                if resp.status not in (200, 204):
                    body = await resp.text()
                    logger.warning(f"Discord 전송 실패 {resp.status}: {body[:200]}")
                    return False
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Discord 전송 예외: {e}")
            return False

    async def send_embed(self, title: str, description: str, color: int = 0x3498DB) -> bool:
        """Discord Embed 메시지 전송."""
        if not self._enabled:
            return False
        payload = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color,
            }]
        }
        try:
            session = self._get_session()
            async with session.post(self._webhook_url, json=payload) as resp:
                if resp.status not in (200, 204):
                    body = await resp.text()
                    logger.warning(f"Discord Embed 전송 실패 {resp.status}: {body[:200]}")
                    return False
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"Discord Embed 전송 예외: {e}")
            return False

    # ── 포맷된 메시지 전송 메서드 ────────────────────────────────

    async def send_trade_opened(self, trade: TradeRecord, current_price: float) -> None:
        direction_emoji = "🟢" if trade.direction == "LONG" else "🔴"
        pnl_pct_sl = abs((trade.stop_loss - trade.entry_price) / trade.entry_price * 100) if trade.stop_loss and trade.entry_price else 0
        pnl_pct_tp = abs((trade.take_profit_1 - trade.entry_price) / trade.entry_price * 100) if trade.take_profit_1 and trade.entry_price else 0

        color = 0x00FF00 if trade.direction == "LONG" else 0xFF0000
        desc = (
            f"💰 **진입가:** ${trade.entry_price:,.2f}\n"
            f"📦 **수량:** {trade.quantity:.4f} BTC\n"
            f"⛔ **SL:** ${trade.stop_loss:,.2f} (-{pnl_pct_sl:.2f}%)\n"
            f"🎯 **TP1:** ${trade.take_profit_1:,.2f} (+{pnl_pct_tp:.2f}%)\n"
            f"⚡ **레버리지:** {trade.leverage}x\n"
            f"📊 **신뢰도:** {(trade.signal_confidence or 0) * 100:.0f}%\n"
            f"🆔 `{trade.trade_id}`"
        )
        await self.send_embed(
            f"{direction_emoji} {trade.direction} {trade.symbol} 진입",
            desc,
            color,
        )

    async def send_trade_closed(self, trade: TradeRecord) -> None:
        if trade.pnl_usdt is None:
            return
        profit = trade.pnl_usdt >= 0
        emoji = "✅" if profit else "❌"
        pnl_sign = "+" if profit else ""
        color = 0x00FF00 if profit else 0xFF0000

        desc = (
            f"💹 **P&L:** {pnl_sign}{trade.pnl_usdt:.2f} USDT"
            f" ({pnl_sign}{(trade.pnl_pct or 0) * 100:.2f}%)\n"
            f"📍 **청산가:** ${trade.exit_price:,.2f}\n"
            f"📝 **사유:** {trade.exit_reason}\n"
            f"🆔 `{trade.trade_id}`"
        )
        await self.send_embed(
            f"{emoji} {trade.direction} 포지션 청산",
            desc,
            color,
        )

    async def send_daily_summary(
        self,
        date: str,
        pnl_usdt: float,
        trade_count: int,
        win_rate: float,
        equity: float,
    ) -> None:
        profit = pnl_usdt >= 0
        sign = "+" if profit else ""
        color = 0x00FF00 if profit else 0xFF0000

        desc = (
            f"💰 **일일 P&L:** {sign}{pnl_usdt:.2f} USDT\n"
            f"🔢 **거래 횟수:** {trade_count}회\n"
            f"🏆 **승률:** {win_rate * 100:.1f}%\n"
            f"💼 **현재 자산:** ${equity:,.2f} USDT"
        )
        await self.send_embed(f"📊 일간 요약 — {date}", desc, color)

    async def send_alert(self, level: str, message: str) -> None:
        color_map = {
            "INFO": 0x3498DB,
            "WARNING": 0xF39C12,
            "ERROR": 0xE74C3C,
            "CRITICAL": 0x8B0000,
        }
        await self.send_embed(f"[{level}] 알림", message, color_map.get(level, 0x3498DB))

    async def send_halt_notification(self, reason: str) -> None:
        desc = f"**사유:** {reason}\n수동으로 재시작이 필요합니다."
        await self.send_embed("🆘 봇 긴급 정지", desc, 0x8B0000)

    async def send_startup(self, symbol: str, leverage: int, is_demo: bool) -> None:
        mode = "📋 페이퍼 트레이딩" if is_demo else "💸 실거래"
        desc = (
            f"📊 **페어:** {symbol}\n"
            f"⚡ **레버리지:** {leverage}x\n"
            f"🔧 **모드:** {mode}"
        )
        await self.send_embed("🤖 자동매매봇 시작", desc, 0x3498DB)

    async def send_test_ping(self) -> bool:
        """연결 테스트용"""
        return await self.send("✅ 자동매매봇 Discord 연결 성공")
