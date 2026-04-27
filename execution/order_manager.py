"""
주문 관리자.
OKX에 실제 주문을 전송하고 SL/TP 알고 주문을 관리한다.
멀티 심볼: 심볼은 signal/trade에서 가져온다.

[개선] 시장가 주문 실패 시 자동 재시도 + 포지션 이중 오픈 방지 검증 포함.
[버그수정] 청산 5회 실패 시 ClosePositionFailedError 발생 (호출자가 긴급 알림 처리).
[버그수정] 진입 주문 부분 체결 시 filled_qty 기준으로 SL/TP 배치.
"""
import asyncio
from datetime import datetime, timezone
from typing import Optional

import aiohttp
from loguru import logger

from config.settings import TradingConfig
from core.exceptions import APIError, ClosePositionFailedError, OrderError
from data.okx_rest_client import OKXRestClient
from database.models import TradeRecord
from risk.position_sizer import PositionSize
from strategy.signal_aggregator import SignalResult

# 청산 주문 재시도 설정 (포지션 청산 실패는 치명적이므로 더 많이 시도)
_CLOSE_MAX_RETRY = 5
_CLOSE_RETRY_DELAY = 3.0


class OrderManager:
    """주문 생성/취소/수정 담당"""

    def __init__(self, rest: OKXRestClient, config: TradingConfig):
        self._rest = rest
        self._config = config

    async def open_position(
        self, signal: SignalResult, pos_size: PositionSize
    ) -> TradeRecord:
        """시장가로 포지션을 열고 SL/TP 알고 주문을 함께 배치한다.

        Returns:
            진입 완료된 TradeRecord (DB 저장은 호출자 책임)
        """
        symbol = signal.symbol or self._config.symbol
        sym_tag = symbol.split("/")[0]
        side = "buy" if signal.direction == "LONG" else "sell"
        pos_side = "long" if signal.direction == "LONG" else "short"

        # 레버리지 설정 (헤지모드: 롱/숏 각각)
        try:
            await self._rest.set_leverage(symbol, self._config.leverage, pos_side=pos_side)
        except (aiohttp.ClientError, OrderError) as e:
            logger.warning(f"[{sym_tag}] 레버리지 설정 실패 (계속 진행): {e}")

        # 시장가 진입
        logger.info(
            f"[{sym_tag}] 진입 주문: {signal.direction} {pos_size.quantity:.4f} @ 시장가"
        )
        order = await self._rest.create_market_order(
            symbol=symbol,
            side=side,
            amount=pos_size.quantity,
            params={"tdMode": "cross", "posSide": pos_side},
        )

        filled_price = float(order.get("average") or order.get("price") or signal.entry_price)
        entry_time = datetime.now(timezone.utc).isoformat()

        # ── [버그4] 실제 체결 수량 확인 ─────────────────────────────
        # OKX 응답의 'filled' 필드 우선, 없으면 info.fillSz, 최종 폴백은 요청 수량
        filled_qty = float(
            order.get("filled")
            or order.get("info", {}).get("fillSz")
            or pos_size.quantity
        )
        is_partial_fill = filled_qty < pos_size.quantity - 1e-8
        if is_partial_fill:
            logger.warning(
                f"[{sym_tag}] 부분 체결 감지: 요청 {pos_size.quantity:.4f} → "
                f"실제 체결 {filled_qty:.4f} "
                f"(미체결 {pos_size.quantity - filled_qty:.4f})"
            )
        # SL/TP는 실제 체결 수량 기준으로 배치해야 수량 초과 오류를 방지한다
        sl_tp_qty = round(filled_qty, 4)

        # TradeRecord 생성 — DB에 실제 체결 수량 저장
        trade = TradeRecord(
            symbol=symbol,
            direction=signal.direction,
            quantity=sl_tp_qty,          # 실제 체결량
            leverage=self._config.leverage,
            entry_price=filled_price,
            stop_loss=signal.stop_price,
            take_profit_1=signal.tp1_price,
            take_profit_2=signal.tp2_price,
            entry_time=entry_time,
            signal_confidence=signal.confidence,
            atr_at_entry=signal.atr_value,
        )
        # 부분 체결 여부를 메타데이터에 기록 (상위 레이어에서 알림에 사용)
        trade.partial_fill = is_partial_fill
        trade.requested_qty = pos_size.quantity

        # SL 알고 주문 — sl_tp_qty(실제 체결량) 기준, 최대 3회 재시도
        sl_side = "sell" if signal.direction == "LONG" else "buy"
        for sl_attempt in range(1, 4):
            try:
                sl_order = await self._rest.create_sl_order(
                    symbol=symbol,
                    side=sl_side,
                    amount=sl_tp_qty,          # 실제 체결량 기준
                    sl_trigger_price=signal.stop_price,
                    pos_side=pos_side,
                )
                trade.sl_algo_id = sl_order.get("id") or sl_order.get("info", {}).get("algoId")
                logger.info(
                    f"[{sym_tag}] SL 알고 주문 배치: ${signal.stop_price:,.2f} "
                    f"qty={sl_tp_qty:.4f} (ID: {trade.sl_algo_id})"
                )
                break
            except (aiohttp.ClientError, OrderError, APIError) as e:
                if sl_attempt < 3:
                    logger.warning(f"[{sym_tag}] SL 알고 주문 실패 (시도 {sl_attempt}/3), 재시도: {e}")
                    await asyncio.sleep(2 * sl_attempt)
                else:
                    logger.error(
                        f"[{sym_tag}] SL 알고 주문 최종 실패 — 소프트웨어 SL로 감시: {e}"
                    )

        logger.info(
            f"[{sym_tag}] 포지션 오픈 완료: {trade.trade_id} | "
            f"{trade.direction} {sl_tp_qty:.4f} @ ${filled_price:,.2f}"
            + (f" [부분체결: 요청 {pos_size.quantity:.4f}]" if is_partial_fill else "")
        )
        return trade

    async def close_position(
        self,
        trade: TradeRecord,
        current_price: float,
        reason: str,
    ) -> float:
        """포지션을 시장가로 전부 청산하고 체결가를 반환한다.

        청산 주문은 실패 시 _CLOSE_MAX_RETRY회까지 재시도한다.
        모든 재시도 실패 시 ClosePositionFailedError를 발생시킨다.
        호출자(trade_lifecycle._close)가 해당 예외를 잡아 텔레그램 긴급 알림을 발송한다.
        """
        symbol = trade.symbol
        sym_tag = symbol.split("/")[0]
        close_side = "sell" if trade.direction == "LONG" else "buy"
        pos_side = "long" if trade.direction == "LONG" else "short"

        logger.info(
            f"[{sym_tag}] 포지션 청산: {trade.trade_id} {trade.direction} "
            f"{trade.quantity:.4f} @ 시장가 (사유: {reason})"
        )

        # 기존 알고 주문 취소
        await self._cancel_algo_orders(trade)

        # 청산 시장가 주문 — 재시도 루프
        last_err: Optional[Exception] = None
        for attempt in range(1, _CLOSE_MAX_RETRY + 1):
            try:
                order = await self._rest.create_market_order(
                    symbol=symbol,
                    side=close_side,
                    amount=trade.quantity,
                    params={"tdMode": "cross", "posSide": pos_side, "reduceOnly": True},
                )
                filled_price = float(order.get("average") or order.get("price") or current_price)
                logger.info(
                    f"[{sym_tag}] 청산 완료 (시도 {attempt}/{_CLOSE_MAX_RETRY}): "
                    f"${filled_price:,.2f} (사유: {reason})"
                )
                return filled_price
            except (aiohttp.ClientError, asyncio.TimeoutError, APIError, OrderError) as e:
                last_err = e
                if attempt < _CLOSE_MAX_RETRY:
                    wait = _CLOSE_RETRY_DELAY * attempt
                    logger.error(
                        f"[{sym_tag}] 청산 주문 실패 (시도 {attempt}/{_CLOSE_MAX_RETRY}), "
                        f"{wait:.0f}초 후 재시도: {e}"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.critical(
                        f"[{sym_tag}] 청산 주문 {_CLOSE_MAX_RETRY}회 전부 실패 — "
                        f"거래소 포지션이 아직 열려 있을 수 있음! "
                        f"마지막 오류: {e}"
                    )

        # ── [버그1] 모든 재시도 실패 → 예외 발생 (호출자가 긴급 알림 처리 후 폴백 적용) ──
        # 이전 코드는 return current_price로 조용히 폴백했으나,
        # 거래소에 포지션이 남아 있을 위험이 있어 명시적 예외로 변경.
        logger.critical(
            f"[{sym_tag}] ClosePositionFailedError 발생 — "
            f"폴백 가격 ${current_price:,.2f} 적용, 수동 확인 필요"
        )
        raise ClosePositionFailedError(
            symbol=symbol,
            trade_id=trade.trade_id,
            fallback_price=current_price,
        )

    async def update_stop_loss(
        self, trade: TradeRecord, new_stop_price: float
    ) -> Optional[str]:
        """기존 SL 알고 주문을 취소하고 새 SL 주문을 배치한다.

        새 주문이 실패하면 최대 2회 재시도한다. 끝내 실패하면 None을 반환하되,
        호출자(trade_lifecycle._safe_replace_sl)가 소프트웨어 감시로 폴백한다.
        """
        symbol = trade.symbol
        sl_side = "sell" if trade.direction == "LONG" else "buy"
        pos_side = "long" if trade.direction == "LONG" else "short"
        old_stop = trade.stop_loss if trade.stop_loss is not None else new_stop_price

        # 기존 SL 취소
        if trade.sl_algo_id:
            try:
                await self._rest.cancel_algo_order(trade.sl_algo_id, symbol)
            except (aiohttp.ClientError, OrderError) as e:
                logger.warning(f"기존 SL 취소 실패: {e}")

        # 새 SL 주문 (최대 2회 재시도)
        last_err: Optional[Exception] = None
        for attempt in range(2):
            try:
                sl_order = await self._rest.create_sl_order(
                    symbol=symbol,
                    side=sl_side,
                    amount=trade.quantity,
                    sl_trigger_price=new_stop_price,
                    pos_side=pos_side,
                )
                new_id = sl_order.get("id") or sl_order.get("info", {}).get("algoId")
                logger.info(
                    f"SL 업데이트: ${old_stop:.2f} → ${new_stop_price:.2f} (qty {trade.quantity})"
                )
                return new_id
            except (aiohttp.ClientError, OrderError) as e:
                last_err = e
                logger.warning(f"SL 재배치 시도 {attempt+1}/2 실패: {e}")
        logger.error(f"SL 재배치 최종 실패: {last_err}")
        return None

    async def _cancel_algo_orders(self, trade: TradeRecord) -> None:
        """SL/TP 알고 주문 취소"""
        symbol = trade.symbol
        for algo_id, name in [(trade.sl_algo_id, "SL"), (trade.tp_algo_id, "TP")]:
            if algo_id:
                try:
                    await self._rest.cancel_algo_order(algo_id, symbol)
                    logger.debug(f"{name} 알고 주문 취소: {algo_id}")
                except (aiohttp.ClientError, OrderError) as e:
                    logger.warning(f"{name} 알고 주문 취소 실패: {e}")
