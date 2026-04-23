"""
OKX REST API ccxt 래퍼.
모든 API 호출은 이 클래스를 통해서만 수행한다.

[개선] 네트워크 재연결 및 일시적 오류 자동 재시도 로직 포함.
"""
import asyncio
import functools
from typing import Optional

import ccxt.async_support as ccxt
import pandas as pd
from loguru import logger

from config.settings import OKXConfig
from core.exceptions import APIError, AuthenticationError, RateLimitError

# 재시도 설정
_MAX_RETRY = 3           # 최대 재시도 횟수
_RETRY_DELAY_SEC = 2.0  # 첫 재시도 대기 (지수 백오프 적용)


def _is_retryable(exc: Exception) -> bool:
    """재시도 가능한 일시적 오류인지 판별한다."""
    if isinstance(exc, (RateLimitError,)):
        return True
    if isinstance(exc, APIError):
        msg = str(exc).lower()
        # 네트워크/서버 일시 장애 패턴
        return any(k in msg for k in (
            "timeout", "connection", "network", "reset", "eof",
            "service unavailable", "503", "502", "504", "rate limit",
            "too many requests", "50011",  # OKX rate limit code
        ))
    return False


class OKXRestClient:
    """ccxt OKX async 래퍼 (자동 재연결 + 재시도 포함)"""

    def __init__(self, config: OKXConfig):
        self._config = config
        self._exchange = self._build_exchange()

    def _build_exchange(self) -> ccxt.okx:
        """ccxt 교환소 인스턴스를 생성한다."""
        options: dict = {"defaultType": "swap"}
        if self._config.is_demo:
            options["headers"] = {"x-simulated-trading": "1"}
            logger.info("OKX REST 클라이언트: 페이퍼 트레이딩 모드")
        else:
            logger.warning("OKX REST 클라이언트: 실거래 모드 — 실제 자금 사용")

        return ccxt.okx(
            {
                "apiKey": self._config.api_key,
                "secret": self._config.secret_key,
                "password": self._config.passphrase,
                "options": options,
                "enableRateLimit": True,  # ccxt 내장 레이트 리밋 준수
            }
        )

    async def _reconnect(self) -> None:
        """ccxt 세션이 끊긴 경우 교환소 인스턴스를 재생성한다."""
        logger.warning("OKX REST 클라이언트 재연결 시도...")
        try:
            await self._exchange.close()
        except Exception:
            pass
        self._exchange = self._build_exchange()
        logger.info("OKX REST 클라이언트 재연결 완료")

    async def _call_with_retry(self, coro_fn, *args, **kwargs):
        """API 호출을 재시도 래퍼로 감싼다.

        일시적 오류(_is_retryable) 발생 시 지수 백오프로 최대 _MAX_RETRY회 재시도.
        재시도 전 ccxt 세션을 재생성한다.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRY + 1):
            try:
                return await coro_fn(*args, **kwargs)
            except (AuthenticationError,) as e:
                # 인증 오류는 재시도해도 의미 없음
                raise
            except (APIError, RateLimitError) as e:
                last_exc = e
                if not _is_retryable(e) or attempt == _MAX_RETRY:
                    raise
                wait = _RETRY_DELAY_SEC * (2 ** (attempt - 1))
                logger.warning(
                    f"OKX API 일시 오류 (시도 {attempt}/{_MAX_RETRY}), "
                    f"{wait:.1f}초 후 재시도: {e}"
                )
                await asyncio.sleep(wait)
                await self._reconnect()
            except Exception as e:
                # 예상치 못한 예외는 바로 raise
                raise
        raise last_exc  # type: ignore[misc]

    # ── 시장 데이터 ─────────────────────────────────────────────

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 300,
        since: Optional[int] = None,
    ) -> pd.DataFrame:
        """OHLCV 캔들 데이터를 DataFrame으로 반환 (자동 재시도 포함).

        Returns:
            columns: [timestamp, open, high, low, close, volume]
            index: DatetimeIndex (UTC)
        """
        async def _call():
            try:
                return await self._exchange.fetch_ohlcv(
                    symbol, timeframe=timeframe, limit=limit, since=since
                )
            except ccxt.AuthenticationError as e:
                raise AuthenticationError(str(e)) from e
            except ccxt.RateLimitExceeded as e:
                raise RateLimitError(str(e)) from e
            except ccxt.BaseError as e:
                raise APIError(str(e)) from e

        raw = await self._call_with_retry(_call)

        if not raw:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        return df

    async def fetch_ticker(self, symbol: str) -> dict:
        """현재가, 24h 거래량, bid/ask 반환"""
        async def _call():
            try:
                return await self._exchange.fetch_ticker(symbol)
            except ccxt.BaseError as e:
                raise APIError(str(e)) from e
        return await self._call_with_retry(_call)

    async def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """현재 펀딩비율 반환 (예: 0.0001 = 0.01%). 실패 시 None."""
        try:
            data = await self._exchange.fetch_funding_rate(symbol)
            rate = data.get("fundingRate") if isinstance(data, dict) else None
            return float(rate) if rate is not None else None
        except (ccxt.BaseError, KeyError, ValueError) as e:
            logger.debug(f"펀딩비 조회 실패: {e}")
            return None

    async def fetch_open_interest(self, symbol: str) -> Optional[float]:
        """미결제약정(OI) 반환 (계약 수). 실패 시 None."""
        try:
            data = await self._exchange.fetch_open_interest(symbol)
            oi = data.get("openInterestAmount") if isinstance(data, dict) else None
            return float(oi) if oi is not None else None
        except (ccxt.BaseError, KeyError, ValueError) as e:
            logger.debug(f"OI 조회 실패: {e}")
            return None

    async def fetch_long_short_ratio(self, symbol: str) -> Optional[float]:
        """롱숏비율 반환 (롱/숏). 실패 시 None.

        OKX 공개 API /api/v5/rubik/stat/contracts/long-short-account-ratio
        ccxt에 직접 지원이 없으므로 publicGetRubikStatContractsLongShortAccountRatio 사용.
        파라미터는 instId가 아닌 ccy(BTC, ETH 등 베이스 통화)를 요구함.
        """
        try:
            base = symbol.split("/")[0].split("-")[0].upper()
            if not base:
                return None

            resp = await self._exchange.publicGetRubikStatContractsLongShortAccountRatio({
                "ccy": base,
                "period": "1H",
            })
            data_list = resp.get("data", []) if isinstance(resp, dict) else []
            if data_list:
                latest = data_list[0]
                if isinstance(latest, (list, tuple)) and len(latest) >= 2:
                    ratio = float(latest[1])
                    if ratio > 0:
                        return ratio
            return None
        except (ccxt.BaseError, KeyError, ValueError) as e:
            logger.debug(f"롱숏비율 조회 실패: {e}")
            return None

    # ── 계좌 ────────────────────────────────────────────────────

    async def fetch_balance(self) -> dict:
        """USDT 잔고 반환 (free, used, total) — 자동 재시도 포함."""
        async def _call():
            try:
                balance = await self._exchange.fetch_balance({"type": "swap"})
                usdt = balance.get("USDT", {})
                return {
                    "free": float(usdt.get("free", 0)),
                    "used": float(usdt.get("used", 0)),
                    "total": float(usdt.get("total", 0)),
                }
            except ccxt.AuthenticationError as e:
                raise AuthenticationError(str(e)) from e
            except ccxt.BaseError as e:
                raise APIError(str(e)) from e
        return await self._call_with_retry(_call)

    async def fetch_positions(self, symbol: str) -> list[dict]:
        """열려있는 포지션 목록 반환"""
        async def _call():
            try:
                positions = await self._exchange.fetch_positions([symbol])
                return [p for p in positions if float(p.get("contracts", 0)) != 0]
            except ccxt.BaseError as e:
                raise APIError(str(e)) from e
        return await self._call_with_retry(_call)

    # ── 레버리지 설정 ───────────────────────────────────────────

    async def set_leverage(self, symbol: str, leverage: int, pos_side: str = "") -> dict:
        """레버리지를 설정한다. 크로스 마진, 헤지모드 기준."""
        try:
            params = {"mgnMode": "cross"}
            if pos_side:
                params["posSide"] = pos_side
            return await self._exchange.set_leverage(leverage, symbol, params=params)
        except ccxt.BaseError as e:
            raise APIError(str(e)) from e

    # ── 주문 ────────────────────────────────────────────────────

    async def create_market_order(
        self, symbol: str, side: str, amount: float, params: Optional[dict] = None
    ) -> dict:
        """시장가 주문 생성 — 네트워크/레이트리밋 오류 시 자동 재시도.

        Args:
            side: "buy" (롱 진입 or 숏 청산) / "sell" (숏 진입 or 롱 청산)
            amount: BTC 수량

        Note:
            잔고 부족(InsufficientFunds) 등 비일시적 오류는 재시도 없이 즉시 raise.
        """
        async def _call():
            try:
                return await self._exchange.create_order(
                    symbol, "market", side, amount, params=params or {}
                )
            except ccxt.InsufficientFunds as e:
                raise APIError(f"잔고 부족: {e}") from e
            except ccxt.BaseError as e:
                raise APIError(str(e)) from e
        return await self._call_with_retry(_call)

    async def create_limit_order(
        self, symbol: str, side: str, amount: float, price: float, params: Optional[dict] = None
    ) -> dict:
        """지정가 주문 생성."""
        try:
            return await self._exchange.create_order(
                symbol, "limit", side, amount, price, params=params or {}
            )
        except ccxt.BaseError as e:
            raise APIError(str(e)) from e

    async def create_sl_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        sl_trigger_price: float,
        pos_side: str = "",
    ) -> dict:
        """SL(손절) 알고 주문 생성."""
        params = {
            "ordType": "conditional",
            "slTriggerPx": str(sl_trigger_price),
            "slOrdPx": "-1",
            "slTriggerPxType": "last",
            "sz": str(amount),
        }
        if pos_side:
            params["posSide"] = pos_side
        try:
            return await self._exchange.create_order(
                symbol, "market", side, amount, params=params
            )
        except ccxt.BaseError as e:
            raise APIError(str(e)) from e

    async def create_algo_order(
        self,
        symbol: str,
        side: str,
        amount: float,
        trigger_price: float,
        order_type: str = "conditional",
        pos_side: str = "",
    ) -> dict:
        """일반 조건부 주문 (TP 등)"""
        params = {
            "ordType": order_type,
            "tpTriggerPx": str(trigger_price),
            "tpOrdPx": "-1",
            "tpTriggerPxType": "last",
        }
        if pos_side:
            params["posSide"] = pos_side
        try:
            return await self._exchange.create_order(
                symbol, "market", side, amount, params=params
            )
        except ccxt.BaseError as e:
            raise APIError(str(e)) from e

    async def cancel_order(self, order_id: str, symbol: str) -> dict:
        """일반 주문 취소"""
        try:
            return await self._exchange.cancel_order(order_id, symbol)
        except ccxt.BaseError as e:
            raise APIError(str(e)) from e

    async def cancel_algo_order(self, algo_id: str, symbol: str) -> dict:
        """알고 주문 취소"""
        try:
            return await self._exchange.cancel_order(
                algo_id, symbol, params={"algoId": algo_id}
            )
        except ccxt.BaseError as e:
            raise APIError(str(e)) from e

    async def fetch_open_orders(self, symbol: str) -> list[dict]:
        """미체결 주문 목록"""
        try:
            return await self._exchange.fetch_open_orders(symbol)
        except ccxt.BaseError as e:
            raise APIError(str(e)) from e

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        """특정 주문 상태 조회"""
        try:
            return await self._exchange.fetch_order(order_id, symbol)
        except ccxt.BaseError as e:
            raise APIError(str(e)) from e

    # ── 정리 ────────────────────────────────────────────────────

    async def close(self) -> None:
        """ccxt 연결 종료. 봇 종료 시 반드시 호출."""
        await self._exchange.close()
        logger.info("OKX REST 클라이언트 종료")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
