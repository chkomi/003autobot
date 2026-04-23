"""
거래 라이프사이클 관리자.
진입 → TP1 부분 청산(50%) → TP2 부분 청산(30%) → 트레일링 스탑(잔여 20%) → 최종 청산.
"""
from datetime import datetime, timezone
from typing import Optional

import aiohttp
from loguru import logger

from config.settings import TradingConfig
from config.strategy_params import RiskParams
from database.db_manager import DatabaseManager
from database.models import BotEvent, TradeRecord
from execution.order_manager import OrderManager
from notification.notification_manager import NotificationManager
from risk.stop_manager import StopManager


class TradeLifecycle:
    """단일 포지션의 전체 생명 주기 관리"""

    def __init__(
        self,
        order_mgr: OrderManager,
        stop_mgr: StopManager,
        db: DatabaseManager,
        notifier: NotificationManager,
        config: TradingConfig,
        risk_params: Optional[RiskParams] = None,
    ):
        self._orders = order_mgr
        self._stops = stop_mgr
        self._db = db
        self._notifier = notifier
        self._config = config
        self._risk = risk_params or RiskParams()
        # 단계별 도달 추적 (trade_id → bool)
        self._tp1_hit: dict[str, bool] = {}
        self._tp2_hit: dict[str, bool] = {}
        # 각 trade의 초기 수량 저장 (부분청산 비율 계산용)
        self._initial_qty: dict[str, float] = {}

    async def monitor(
        self,
        trade: TradeRecord,
        current_price: float,
        atr_value: float,
    ) -> Optional[str]:
        """포지션 상태를 점검하고 필요한 액션을 취한다.

        순서: SL 체크 → TP1/TP2 단계 처리(동일 틱 동시 도달 허용) → 본절/트레일링 이동 →
        타임 리밋(수익 보호 포함) → 적응형 SL(TP1 전).
        """
        trade_id = trade.trade_id

        # 최초 감시 시 초기 수량 기록
        if trade_id not in self._initial_qty:
            self._initial_qty[trade_id] = trade.quantity

        # 1. 손절 체크
        if self._stops.is_stop_hit(trade, current_price):
            return await self._close(trade, current_price, "SL")

        # 2. TP1 도달 → 50% 부분 청산 + 본절 이동
        tp1_just_hit = False
        if not self._tp1_hit.get(trade_id, False):
            if self._stops.is_tp1_hit(trade, current_price):
                self._tp1_hit[trade_id] = True
                tp1_just_hit = True
                await self._partial_close(
                    trade, current_price,
                    ratio=self._risk.tp1_exit_pct,
                    label="TP1",
                )
                await self._move_stop_to_breakeven(trade)

        # 3. TP2 도달 → 추가 청산 (TP1과 같은 틱에 동시 도달 가능)
        if self._tp1_hit.get(trade_id, False) and not self._tp2_hit.get(trade_id, False):
            if self._stops.is_tp2_hit(trade, current_price):
                self._tp2_hit[trade_id] = True
                target_qty = self._initial_qty[trade_id] * self._risk.tp2_exit_pct
                await self._partial_close(
                    trade, current_price,
                    qty=target_qty,
                    label="TP2",
                )

        # 4. 트레일링 스탑 (TP1 이후 상시, 단 TP1 직후에는 본절 이동이 우선이므로 스킵)
        if self._tp1_hit.get(trade_id, False) and not tp1_just_hit:
            stop_update = self._stops.check_trailing_stop(trade, current_price, atr_value)
            if stop_update.should_update:
                await self._safe_replace_sl(trade, stop_update.new_stop_price)

        # 5. 적응형 SL (TP1 도달 전에만 — ATR 변화에 따라 SL 거리 조정)
        elif (
            not self._tp1_hit.get(trade_id, False)
            and getattr(self._risk, "adaptive_sl_enabled", False)
        ):
            adjust = self._stops.propose_adaptive_sl(trade, atr_value)
            if adjust.should_update:
                await self._safe_replace_sl(trade, adjust.new_stop_price)

        # 6. 타임 리밋 체크 (수익률이 임계 이상이면 청산 대신 유지 → 트레일링/적응형이 보호)
        if self._stops.is_time_limit_hit(trade, self._config.max_trade_duration_hours):
            if not self._is_profit_locked(trade, current_price):
                return await self._close(trade, current_price, "TIME")
            logger.info(
                f"[{trade.symbol.split('/')[0]}] 타임 리밋 도달했지만 이익 중 — 청산 대신 트레일링 유지 ({trade.trade_id})"
            )

        return None

    def _is_profit_locked(self, trade: TradeRecord, current_price: float) -> bool:
        """타임 리밋 도달 시 청산 대신 유지할 정도의 수익 상태인지 판정."""
        threshold = getattr(self._risk, "time_exit_profit_lock_pct", 0.003)
        if trade.entry_price is None or trade.entry_price <= 0:
            return False
        if trade.direction == "LONG":
            profit_pct = (current_price - trade.entry_price) / trade.entry_price
        else:
            profit_pct = (trade.entry_price - current_price) / trade.entry_price
        return profit_pct >= threshold

    async def _safe_replace_sl(self, trade: TradeRecord, new_stop_price: float) -> None:
        """SL 알고 주문을 안전하게 재배치한다. 실패 시 메모리상 SL은 새 값으로 유지하되
        DB/알고 ID는 이전 상태를 보존한다 (소프트웨어 모니터가 계속 감시)."""
        prev_algo_id = trade.sl_algo_id
        prev_stop = trade.stop_loss
        try:
            new_id = await self._orders.update_stop_loss(trade, new_stop_price)
        except (aiohttp.ClientError, KeyError, ValueError) as e:
            logger.warning(f"SL 재배치 실패({trade.trade_id}): {e} — 이전 SL 유지")
            return
        trade.stop_loss = new_stop_price
        if new_id:
            trade.sl_algo_id = new_id
            try:
                await self._db.update_trade(trade)
            except Exception as e:
                logger.warning(f"SL 재배치 DB 동기화 실패({trade.trade_id}): {e}")
                trade.sl_algo_id = prev_algo_id
        else:
            # 알고 주문 ID가 없으면 거래소 측 SL은 이미 취소된 상태 — 소프트웨어 SL로만 감시
            logger.warning(
                f"SL 알고 ID 미반환({trade.trade_id}): prev=${prev_stop} → ${new_stop_price} "
                f"— 소프트웨어 감시로 동작"
            )

    async def _close(self, trade: TradeRecord, current_price: float, reason: str) -> str:
        """전량 청산 후 DB 업데이트 및 알림"""
        exit_price = await self._orders.close_position(trade, current_price, reason)
        pnl = self._calc_pnl(trade, exit_price)

        trade.status = "CLOSED"
        trade.exit_price = exit_price
        trade.exit_time = datetime.now(timezone.utc).isoformat()
        trade.exit_reason = reason
        trade.pnl_usdt = pnl
        trade.pnl_pct = (exit_price - (trade.entry_price or exit_price)) / (trade.entry_price or 1)
        if trade.direction == "SHORT":
            trade.pnl_pct = -trade.pnl_pct

        await self._db.update_trade(trade)
        await self._db.update_daily_pnl(
            pnl_delta=pnl,
            is_win=pnl > 0,
        )
        await self._notifier.on_trade_closed(trade)

        # 추적 상태 정리
        self._tp1_hit.pop(trade.trade_id, None)
        self._tp2_hit.pop(trade.trade_id, None)
        self._initial_qty.pop(trade.trade_id, None)

        logger.info(
            f"거래 완료: {trade.trade_id} | {reason} | "
            f"P&L: {'+' if pnl >= 0 else ''}{pnl:.2f} USDT"
        )
        return reason

    async def _partial_close(
        self,
        trade: TradeRecord,
        current_price: float,
        ratio: Optional[float] = None,
        qty: Optional[float] = None,
        label: str = "TP",
    ) -> None:
        """부분 청산 — ratio(현재 수량의 비율) 또는 qty(고정 수량) 중 하나 지정.

        부분 청산 후 남은 수량 기준으로 거래소 SL 알고 주문도 재배치해, 원래 수량으로
        걸려 있던 SL이 트리거 시 실패·체결 불일치를 일으키지 않도록 한다.
        """
        if qty is None and ratio is not None:
            qty = round(trade.quantity * ratio, 4)
        if qty is None:
            return
        partial_qty = min(round(qty, 4), trade.quantity)
        if partial_qty < 0.001:
            return

        close_side = "sell" if trade.direction == "LONG" else "buy"
        pos_side = "long" if trade.direction == "LONG" else "short"
        try:
            order = await self._orders._rest.create_market_order(
                symbol=trade.symbol,
                side=close_side,
                amount=partial_qty,
                params={"tdMode": "cross", "posSide": pos_side, "reduceOnly": True},
            )
            partial_price = float(order.get("average") or current_price)
            partial_pnl = self._calc_pnl_partial(trade, partial_price, partial_qty)

            trade.quantity = round(trade.quantity - partial_qty, 4)
            await self._db.update_trade(trade)

            logger.info(
                f"{label} 부분 청산: {trade.trade_id} | "
                f"{partial_qty:.4f} @ ${partial_price:,.2f} | "
                f"P&L: {partial_pnl:+.2f} USDT"
            )

            # 잔여 수량이 남아있다면 거래소 SL 알고 주문 수량도 동기화
            if trade.quantity >= 0.001 and trade.stop_loss is not None:
                try:
                    new_id = await self._orders.update_stop_loss(trade, trade.stop_loss)
                    if new_id:
                        trade.sl_algo_id = new_id
                        await self._db.update_trade(trade)
                except (aiohttp.ClientError, KeyError, ValueError) as e:
                    logger.warning(
                        f"{label} 이후 SL 수량 동기화 실패({trade.trade_id}): {e}"
                    )
        except (aiohttp.ClientError, KeyError, ValueError) as e:
            logger.error(f"{label} 부분 청산 실패: {e}")

    async def _move_stop_to_breakeven(self, trade: TradeRecord) -> None:
        """TP1 도달 후 SL을 진입가+수수료 버퍼로 이동(리스크 프리 전환)."""
        if trade.entry_price is None:
            return
        buffer_pct = getattr(self._risk, "breakeven_buffer_pct", 0.0006)
        if trade.direction == "LONG":
            new_stop = trade.entry_price * (1.0 + buffer_pct)
        else:
            new_stop = trade.entry_price * (1.0 - buffer_pct)
        try:
            new_id = await self._orders.update_stop_loss(trade, new_stop)
            trade.stop_loss = new_stop
            if new_id:
                trade.sl_algo_id = new_id
            await self._db.update_trade(trade)
            logger.info(
                f"SL → 본절+버퍼 이동: {trade.trade_id} @ ${new_stop:,.2f} "
                f"(entry ${trade.entry_price:,.2f}, buffer {buffer_pct*100:.3f}%)"
            )
        except (aiohttp.ClientError, KeyError, ValueError) as e:
            logger.warning(f"본절 이동 실패: {e}")

    def _calc_pnl(self, trade: TradeRecord, exit_price: float) -> float:
        """전체 P&L 계산"""
        if trade.entry_price is None:
            return 0.0
        if trade.direction == "LONG":
            return (exit_price - trade.entry_price) * trade.quantity * trade.leverage
        return (trade.entry_price - exit_price) * trade.quantity * trade.leverage

    def _calc_pnl_partial(self, trade: TradeRecord, exit_price: float, qty: float) -> float:
        if trade.entry_price is None:
            return 0.0
        if trade.direction == "LONG":
            return (exit_price - trade.entry_price) * qty * trade.leverage
        return (trade.entry_price - exit_price) * qty * trade.leverage

    # ── 외부용: 이벤트/뉴스/스파이크 대응 비상 핸들러 ──────────────

    async def emergency_close(
        self,
        trade: TradeRecord,
        current_price: float,
        reason: str,
    ) -> str:
        """외부 모듈(이벤트/뉴스/스파이크 감지)에서 단일 포지션 전량 청산 요청."""
        return await self._close(trade, current_price, reason)

    async def reduce_position(
        self,
        trade: TradeRecord,
        current_price: float,
        ratio: float,
        reason: str = "REDUCE",
    ) -> None:
        """외부용 부분 감축. ratio(0~1)만큼 수량을 줄인다."""
        ratio = max(0.0, min(1.0, float(ratio)))
        if ratio <= 0.0 or trade.quantity <= 0:
            return
        await self._partial_close(trade, current_price, ratio=ratio, label=reason)
