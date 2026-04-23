"""
텔레그램 봇 알림 전송.
Telegram Bot API REST 직접 호출 (aiohttp 세션 재사용).
"""
import asyncio
from typing import Optional

import aiohttp
from loguru import logger

from core.exceptions import NotificationError
from database.models import TradeRecord


class TelegramNotifier:
    """텔레그램 봇 알림 전송기"""

    BASE_URL = "https://api.telegram.org/bot{token}/sendMessage"

    def __init__(self, bot_token: str, chat_id: str):
        self._token = bot_token
        self._chat_id = chat_id
        self._enabled = bool(bot_token and chat_id)
        self._session: Optional[aiohttp.ClientSession] = None
        if not self._enabled:
            logger.warning("텔레그램 봇 토큰 또는 채팅 ID가 설정되지 않음 — 알림 비활성화")

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10))
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """메시지 전송. 실패해도 봇 운영에 영향 없도록 예외를 흡수한다."""
        if not self._enabled:
            return False
        url = self.BASE_URL.format(token=self._token)
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": parse_mode,
        }
        try:
            session = self._get_session()
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(f"텔레그램 전송 실패 {resp.status}: {body[:200]}")
                    return False
                return True
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"텔레그램 전송 예외: {e}")
            return False

    # ── 포맷된 메시지 전송 메서드 ────────────────────────────────

    async def send_trade_opened(self, trade: TradeRecord, current_price: float) -> None:
        direction_emoji = "🟢" if trade.direction == "LONG" else "🔴"
        pnl_pct_sl = abs((trade.stop_loss - trade.entry_price) / trade.entry_price * 100) if trade.stop_loss and trade.entry_price else 0
        pnl_pct_tp = abs((trade.take_profit_1 - trade.entry_price) / trade.entry_price * 100) if trade.take_profit_1 and trade.entry_price else 0

        msg = (
            f"{direction_emoji} <b>{trade.direction} {trade.symbol} 진입</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 진입가: ${trade.entry_price:,.2f}\n"
            f"📦 수량: {trade.quantity:.4f} BTC\n"
            f"⛔ SL: ${trade.stop_loss:,.2f} (-{pnl_pct_sl:.2f}%)\n"
            f"🎯 TP1: ${trade.take_profit_1:,.2f} (+{pnl_pct_tp:.2f}%)\n"
            f"⚡ 레버리지: {trade.leverage}x\n"
            f"📊 신호 신뢰도: {(trade.signal_confidence or 0) * 100:.0f}%\n"
            f"🆔 {trade.trade_id}"
        )
        await self.send(msg)

    async def send_trade_closed(self, trade: TradeRecord) -> None:
        if trade.pnl_usdt is None:
            return
        profit = trade.pnl_usdt >= 0
        emoji = "✅" if profit else "❌"
        pnl_sign = "+" if profit else ""

        msg = (
            f"{emoji} <b>{trade.direction} 포지션 청산</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💹 P&L: <b>{pnl_sign}{trade.pnl_usdt:.2f} USDT</b>"
            f" ({pnl_sign}{(trade.pnl_pct or 0) * 100:.2f}%)\n"
            f"📍 청산가: ${trade.exit_price:,.2f}\n"
            f"📝 청산 사유: {trade.exit_reason}\n"
            f"🆔 {trade.trade_id}"
        )
        await self.send(msg)

    async def send_daily_summary(
        self,
        date: str,
        pnl_usdt: float,
        trade_count: int,
        win_rate: float,
        equity: float,
    ) -> None:
        profit = pnl_usdt >= 0
        emoji = "📈" if profit else "📉"
        sign = "+" if profit else ""
        msg = (
            f"{emoji} <b>일간 요약 — {date}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"💰 일일 P&L: <b>{sign}{pnl_usdt:.2f} USDT</b>\n"
            f"🔢 거래 횟수: {trade_count}회\n"
            f"🏆 승률: {win_rate * 100:.1f}%\n"
            f"💼 현재 자산: ${equity:,.2f} USDT"
        )
        await self.send(msg)

    async def send_alert(self, level: str, message: str) -> None:
        level_emoji = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "🚨",
            "CRITICAL": "🆘",
        }.get(level, "📢")
        msg = f"{level_emoji} <b>[{level}]</b> {message}"
        await self.send(msg)

    async def send_halt_notification(self, reason: str) -> None:
        msg = (
            f"🆘 <b>봇 긴급 정지</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"사유: {reason}\n"
            f"수동으로 재시작이 필요합니다."
        )
        await self.send(msg)

    async def send_startup(self, symbol: str, leverage: int, is_demo: bool) -> None:
        mode = "📋 페이퍼 트레이딩" if is_demo else "💸 실거래"
        msg = (
            f"🤖 <b>자동매매봇 시작</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📊 페어: {symbol}\n"
            f"⚡ 레버리지: {leverage}x\n"
            f"🔧 모드: {mode}"
        )
        await self.send(msg)

    async def send_test_ping(self) -> bool:
        """연결 테스트용"""
        return await self.send("✅ 자동매매봇 텔레그램 연결 성공")
