"""
OKX WebSocket 클라이언트.
캔들, 티커, 주문, 포지션 실시간 데이터를 구독하고
콜백 함수로 전달한다. 연결이 끊어지면 자동 재연결한다.
"""
import asyncio
import hashlib
import hmac
import json
import time
from typing import Callable, Awaitable, Optional

import websockets
from loguru import logger

from core.exceptions import WebSocketError


# 콜백 타입 힌트
MessageCallback = Callable[[dict], Awaitable[None]]


class OKXWebSocketClient:
    """OKX WebSocket Public + Private 구독 관리"""

    WS_PUBLIC_LIVE = "wss://ws.okx.com:8443/ws/v5/public"
    WS_PUBLIC_DEMO = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
    WS_PRIVATE_LIVE = "wss://ws.okx.com:8443/ws/v5/private"
    WS_PRIVATE_DEMO = "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999"

    PING_INTERVAL = 25  # OKX 서버는 30초 무응답 시 연결 끊음
    RECONNECT_DELAY = 5  # 재연결 대기 시간(초)
    MAX_RECONNECT_DELAY = 60

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        passphrase: str = "",
        is_demo: bool = True,
    ):
        self._api_key = api_key
        self._secret_key = secret_key
        self._passphrase = passphrase
        self._is_demo = is_demo

        self._public_url = self.WS_PUBLIC_DEMO if is_demo else self.WS_PUBLIC_LIVE
        self._private_url = self.WS_PRIVATE_DEMO if is_demo else self.WS_PRIVATE_LIVE

        # channel_key → callback 매핑
        self._callbacks: dict[str, list[MessageCallback]] = {}
        # 구독할 채널 목록 (재연결 시 재구독용)
        self._public_subscriptions: list[dict] = []
        self._private_subscriptions: list[dict] = []

        self._public_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._private_ws: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._tasks: list[asyncio.Task] = []

    # ── 구독 등록 API ────────────────────────────────────────────

    def subscribe_candles(self, inst_id: str, channel: str, callback: MessageCallback) -> None:
        """캔들 구독. channel 예: 'candle15m', 'candle1H', 'candle4H'"""
        arg = {"channel": channel, "instId": inst_id}
        self._add_public_subscription(arg, callback)

    def subscribe_ticker(self, inst_id: str, callback: MessageCallback) -> None:
        """실시간 티커 구독"""
        arg = {"channel": "tickers", "instId": inst_id}
        self._add_public_subscription(arg, callback)

    def subscribe_orders(self, inst_id: str, callback: MessageCallback) -> None:
        """주문 체결 실시간 구독 (Private)"""
        arg = {"channel": "orders", "instType": "SWAP", "instId": inst_id}
        self._add_private_subscription(arg, callback)

    def subscribe_positions(self, inst_id: str, callback: MessageCallback) -> None:
        """포지션 변경 실시간 구독 (Private)"""
        arg = {"channel": "positions", "instType": "SWAP", "instId": inst_id}
        self._add_private_subscription(arg, callback)

    def subscribe_account(self, callback: MessageCallback) -> None:
        """계좌 잔고 변경 실시간 구독 (Private)"""
        arg = {"channel": "account"}
        self._add_private_subscription(arg, callback)

    # ── 시작 / 종료 ──────────────────────────────────────────────

    async def start(self) -> None:
        """WebSocket 연결 루프를 백그라운드 태스크로 시작"""
        self._running = True
        if self._public_subscriptions:
            self._tasks.append(asyncio.create_task(self._public_loop()))
        if self._private_subscriptions:
            self._tasks.append(asyncio.create_task(self._private_loop()))
        logger.info(
            f"WebSocket 시작 — public 구독: {len(self._public_subscriptions)}, "
            f"private 구독: {len(self._private_subscriptions)}"
        )

    async def stop(self) -> None:
        """WebSocket 연결 종료"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._public_ws:
            await self._public_ws.close()
        if self._private_ws:
            await self._private_ws.close()
        logger.info("WebSocket 종료")

    # ── 내부 연결 루프 ───────────────────────────────────────────

    async def _public_loop(self) -> None:
        delay = self.RECONNECT_DELAY
        while self._running:
            try:
                async with websockets.connect(
                    self._public_url,
                    ping_interval=self.PING_INTERVAL,
                    ping_timeout=10,
                ) as ws:
                    self._public_ws = ws
                    delay = self.RECONNECT_DELAY  # 성공 시 delay 리셋
                    logger.info("WebSocket Public 연결 성공")
                    await self._send_subscribe(ws, self._public_subscriptions)
                    await self._message_loop(ws)
            except (websockets.ConnectionClosed, OSError) as e:
                logger.warning(f"WebSocket Public 연결 끊김: {e}. {delay}초 후 재연결")
            except (websockets.WebSocketException, asyncio.TimeoutError, ValueError) as e:
                logger.error(f"WebSocket Public 예외: {e}. {delay}초 후 재연결")

            if not self._running:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.MAX_RECONNECT_DELAY)

    async def _private_loop(self) -> None:
        if not (self._api_key and self._secret_key):
            logger.warning("Private WebSocket: API 키가 설정되지 않아 연결하지 않음")
            return

        delay = self.RECONNECT_DELAY
        while self._running:
            try:
                async with websockets.connect(
                    self._private_url,
                    ping_interval=self.PING_INTERVAL,
                    ping_timeout=10,
                ) as ws:
                    self._private_ws = ws
                    delay = self.RECONNECT_DELAY
                    logger.info("WebSocket Private 연결 성공")
                    await self._login(ws)
                    await asyncio.sleep(1)  # 로그인 응답 대기
                    await self._send_subscribe(ws, self._private_subscriptions)
                    await self._message_loop(ws)
            except (websockets.ConnectionClosed, OSError) as e:
                logger.warning(f"WebSocket Private 연결 끊김: {e}. {delay}초 후 재연결")
            except (websockets.WebSocketException, asyncio.TimeoutError, ValueError) as e:
                logger.error(f"WebSocket Private 예외: {e}. {delay}초 후 재연결")

            if not self._running:
                break
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.MAX_RECONNECT_DELAY)

    async def _message_loop(self, ws) -> None:
        """수신 메시지를 파싱해 콜백에 전달"""
        async for raw in ws:
            if not self._running:
                break
            if raw == "pong":
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"WebSocket 메시지 JSON 파싱 실패: {e}")
                continue

            # 에러 응답 처리
            if data.get("event") == "error":
                logger.error(f"WebSocket 에러 응답: {data}")
                continue

            # 구독 확인 응답
            if data.get("event") in ("subscribe", "login"):
                logger.debug(f"WebSocket 이벤트: {data}")
                continue

            # 데이터 메시지 → 콜백
            arg = data.get("arg", {})
            channel_key = self._make_key(arg)
            callbacks = self._callbacks.get(channel_key, [])
            for cb in callbacks:
                try:
                    await cb(data)
                except (KeyError, ValueError, TypeError) as e:
                    logger.error(f"WebSocket 콜백 예외 ({channel_key}): {e}")

    # ── 내부 유틸 ────────────────────────────────────────────────

    async def _login(self, ws) -> None:
        """Private WebSocket 로그인"""
        ts = str(int(time.time()))
        sign = self._sign(ts)
        msg = {
            "op": "login",
            "args": [
                {
                    "apiKey": self._api_key,
                    "passphrase": self._passphrase,
                    "timestamp": ts,
                    "sign": sign,
                }
            ],
        }
        await ws.send(json.dumps(msg))

    def _sign(self, timestamp: str) -> str:
        """OKX WebSocket 로그인 서명 생성"""
        message = timestamp + "GET" + "/users/self/verify"
        return hmac.new(
            self._secret_key.encode(), message.encode(), hashlib.sha256
        ).hexdigest()

    async def _send_subscribe(self, ws, args: list[dict]) -> None:
        """채널 구독 메시지 전송"""
        if not args:
            return
        msg = {"op": "subscribe", "args": args}
        await ws.send(json.dumps(msg))

    def _add_public_subscription(self, arg: dict, callback: MessageCallback) -> None:
        if arg not in self._public_subscriptions:
            self._public_subscriptions.append(arg)
        key = self._make_key(arg)
        self._callbacks.setdefault(key, []).append(callback)

    def _add_private_subscription(self, arg: dict, callback: MessageCallback) -> None:
        if arg not in self._private_subscriptions:
            self._private_subscriptions.append(arg)
        key = self._make_key(arg)
        self._callbacks.setdefault(key, []).append(callback)

    @staticmethod
    def _make_key(arg: dict) -> str:
        """채널 식별 키 생성"""
        channel = arg.get("channel", "")
        inst_id = arg.get("instId", "")
        return f"{channel}:{inst_id}"
