"""
이벤트 드리븐 백테스터.
과거 OHLCV 데이터로 Triple Confirmation 전략을 시뮬레이션한다.
"""
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from loguru import logger

from backtest.performance import PerformanceAnalyzer, PerformanceReport
from config.strategy_params import AllStrategyParams, BacktestParams
from strategy.indicators import calc_atr
from strategy.micro_confirmation import MicroConfirmation
from strategy.momentum_trigger import MomentumTrigger
from strategy.regime_detector import RegimeDetector
from strategy.trend_filter import TrendFilter


@dataclass
class BacktestTrade:
    direction: str
    entry_price: float
    stop_price: float
    tp1_price: float
    quantity: float
    atr_value: float
    entry_idx: int

    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    exit_idx: Optional[int] = None
    pnl_usdt: Optional[float] = None

    entry_time: Optional[str] = None
    exit_time: Optional[str] = None
    regime: Optional[str] = None  # "TRENDING" | "RANGING" | "VOLATILE"

    @property
    def status(self) -> str:
        return "CLOSED" if self.exit_price is not None else "OPEN"


class Backtester:
    """Triple Confirmation 전략 백테스터"""

    def __init__(
        self,
        params: AllStrategyParams,
        initial_balance: float = 1000.0,
        leverage: int = 5,
        max_position_pct: float = 0.10,
        commission_pct: float = 0.0005,  # OKX taker 수수료 0.05%
        slippage_pct: float = 0.0002,    # 슬리피지 0.02%
        max_trade_hours: int = 48,
        backtest_params: BacktestParams = None,
    ):
        self._params = params
        self._balance = initial_balance
        self._initial = initial_balance
        self._leverage = leverage
        self._max_pos_pct = max_position_pct
        self._commission = commission_pct
        self._slippage = slippage_pct
        self._max_hours = max_trade_hours
        self._bt_params = backtest_params or getattr(params, 'backtest', None) or BacktestParams()

        self._trend = TrendFilter(params.trend)
        self._momentum = MomentumTrigger(params.momentum)
        self._micro = MicroConfirmation(params.micro)

        self._regime_detector = RegimeDetector(
            adx_trending_threshold=params.regime.adx_trending_threshold,
            adx_ranging_threshold=params.regime.adx_ranging_threshold,
            bb_bandwidth_low=params.regime.bb_bandwidth_low,
            bb_bandwidth_high=params.regime.bb_bandwidth_high,
            atr_pct_volatile=params.regime.atr_pct_volatile,
        )
        # Debug counters
        self._debug_counts = {"layer1_neutral": 0, "layer1_pass": 0, "layer2_fail": 0, "layer3_fail": 0, "regime_block": 0, "signal_pass": 0}

        self._trades: list[BacktestTrade] = []
        self._open_trade: Optional[BacktestTrade] = None

    def run(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
    ) -> PerformanceReport:
        """백테스트를 실행한다.

        모든 DataFrame은 DatetimeIndex(UTC)이어야 한다.
        15M 기준으로 이벤트를 처리한다.
        """
        logger.info(f"백테스트 시작: {len(df_15m)}개 15M 캔들")

        for i in range(len(df_15m)):
            current_time = df_15m.index[i]
            current_price = float(df_15m["close"].iloc[i])

            # 1. 오픈 포지션 관리
            if self._open_trade is not None:
                exited = self._check_exit(self._open_trade, df_15m, i)
                if exited:
                    self._open_trade = None
                    continue

            # 2. 새 시그널 탐색 (포지션 없을 때만)
            if self._open_trade is None:
                signal = self._evaluate_signal(df_4h, df_1h, df_15m, i, current_time)
                if signal:
                    self._enter(signal, df_15m, i)

        # 백테스트 종료 시 미청산 포지션 강제 청산
        if self._open_trade and len(df_15m) > 0:
            last_price = float(df_15m["close"].iloc[-1])
            self._force_close(self._open_trade, last_price, len(df_15m) - 1, df_15m)

        logger.info(f"[백테스트 디버그] 레이어 통과율: {self._debug_counts}")

        closed = []
        for t in self._trades:
            if t.exit_price is not None:
                d = t.__dict__.copy()
                d["status"] = "CLOSED"
                d["regime"] = t.regime
                closed.append(d)
        report = PerformanceAnalyzer(closed, self._initial).calculate()

        logger.info(f"백테스트 완료: {report.summary()}")
        return report

    def get_trades(self) -> list[dict]:
        result = []
        for t in self._trades:
            d = t.__dict__.copy()
            d["status"] = "CLOSED" if t.exit_price is not None else "OPEN"
            result.append(d)
        return result

    def get_debug_counts(self) -> dict:
        """레이어별 통과/차단 카운트 반환 (게이트 진단용)"""
        return dict(self._debug_counts)

    def _evaluate_signal(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        idx_15m: int,
        current_time: pd.Timestamp,
    ) -> Optional[dict]:
        """현재 시점의 신호를 평가한다."""
        # 타임프레임별 슬라이스 (look-ahead bias 방지: 현재 시간 이전 데이터만 사용)
        slice_4h = df_4h[df_4h.index <= current_time]
        slice_1h = df_1h[df_1h.index <= current_time]
        slice_15m = df_15m.iloc[:idx_15m + 1]

        if len(slice_4h) < 220 or len(slice_1h) < 60 or len(slice_15m) < 30:
            return None

        bp = self._bt_params

        # Layer 1: 추세 필터 (allow_partial 지원)
        trend = self._trend.analyze(slice_4h, allow_partial=bp.allow_partial_ema)
        if trend is None or trend.regime == "NEUTRAL":
            self._debug_counts["layer1_neutral"] += 1
            return None

        self._debug_counts["layer1_pass"] += 1

        # 레짐 감지
        regime_result = self._regime_detector.detect(slice_4h)
        regime_tag = regime_result.regime if regime_result else "UNKNOWN"

        # RANGING 시장에서 진입 차단 (라이브 동일: block_ranging 설정 존중)
        if regime_result and regime_result.is_ranging and self._params.regime.block_ranging:
            self._debug_counts["regime_block"] += 1
            return None

        # Layer 2: 모멘텀 트리거 (완화된 RSI/MACD lookback 지원)
        mom = self._momentum.analyze(
            slice_1h,
            rsi_long_min=bp.rsi_long_min,
            rsi_long_max=bp.rsi_long_max,
            rsi_short_min=bp.rsi_short_min,
            rsi_short_max=bp.rsi_short_max,
            macd_cross_lookback=bp.macd_cross_lookback,
        )

        # Layer 3: 미시 구조 확인 (완화된 BB/볼륨 임계값 지원)
        micro = self._micro.analyze(
            slice_15m,
            volume_multiplier=bp.volume_multiplier,
            bb_long_threshold=bp.bb_long_threshold,
            bb_short_threshold=bp.bb_short_threshold,
        )

        # 시그널 모드에 따른 Layer2/Layer3 조합 판정
        layer2_pass = False
        layer3_pass = False

        if mom is not None:
            if trend.is_bullish and mom.long_trigger:
                layer2_pass = True
            elif trend.is_bearish and mom.short_trigger:
                layer2_pass = True

        if micro is not None:
            if trend.is_bullish and micro.long_confirm:
                layer3_pass = True
            elif trend.is_bearish and micro.short_confirm:
                layer3_pass = True

        signal_mode = bp.signal_mode
        if signal_mode == "AND":
            if not (layer2_pass and layer3_pass):
                return None
        elif signal_mode == "OR":
            if not (layer2_pass or layer3_pass):
                return None
        elif signal_mode == "SCORE_ONLY":
            pass  # Layer2/Layer3 무시, 스코어만으로 판단
        else:
            # 기본값: AND
            if not (layer2_pass and layer3_pass):
                return None

        direction = "LONG" if trend.is_bullish else "SHORT"

        # ATR 기반 SL/TP
        atr_s = calc_atr(slice_1h["high"], slice_1h["low"], slice_1h["close"], self._params.risk.atr_period)
        atr_val = float(atr_s.iloc[-1])
        if pd.isna(atr_val):
            return None

        entry = float(slice_15m["close"].iloc[-1]) if len(slice_15m) > 0 else float(slice_1h["close"].iloc[-1])
        if direction == "LONG":
            stop = entry - atr_val * self._params.risk.atr_sl_multiplier
            tp1 = entry + atr_val * self._params.risk.atr_tp1_multiplier
        else:
            stop = entry + atr_val * self._params.risk.atr_sl_multiplier
            tp1 = entry - atr_val * self._params.risk.atr_tp1_multiplier

        self._debug_counts["signal_pass"] += 1
        return {
            "direction": direction,
            "entry_price": entry,
            "stop_price": stop,
            "tp1_price": tp1,
            "atr_value": atr_val,
            "regime": regime_tag,
        }

    def _enter(self, signal: dict, df_15m: pd.DataFrame, idx: int) -> None:
        # 수수료 + 슬리피지 반영
        entry_price = signal["entry_price"] * (1 + self._slippage)

        # 포지션 사이즈: 잔고의 max_position_pct
        notional = self._balance * self._max_pos_pct * self._leverage
        qty = round(notional / entry_price, 4)
        qty = max(qty, 0.001)

        trade = BacktestTrade(
            direction=signal["direction"],
            entry_price=entry_price,
            stop_price=signal["stop_price"],
            tp1_price=signal["tp1_price"],
            quantity=qty,
            atr_value=signal["atr_value"],
            entry_idx=idx,
            entry_time=str(df_15m.index[idx]),
        )
        trade.regime = signal.get("regime")
        self._open_trade = trade
        self._trades.append(trade)

    def _check_exit(self, trade: BacktestTrade, df_15m: pd.DataFrame, idx: int) -> bool:
        row = df_15m.iloc[idx]
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        current_time = df_15m.index[idx]

        # SL 체크
        if trade.direction == "LONG" and low <= trade.stop_price:
            self._force_close(trade, trade.stop_price, idx, df_15m, "SL")
            return True
        if trade.direction == "SHORT" and high >= trade.stop_price:
            self._force_close(trade, trade.stop_price, idx, df_15m, "SL")
            return True

        # TP1 체크
        if trade.direction == "LONG" and high >= trade.tp1_price:
            self._force_close(trade, trade.tp1_price, idx, df_15m, "TP1")
            return True
        if trade.direction == "SHORT" and low <= trade.tp1_price:
            self._force_close(trade, trade.tp1_price, idx, df_15m, "TP1")
            return True

        # 타임 리밋
        if trade.entry_time:
            elapsed = (current_time - pd.Timestamp(trade.entry_time)).total_seconds() / 3600
            if elapsed >= self._max_hours:
                self._force_close(trade, close, idx, df_15m, "TIME")
                return True

        return False

    def _force_close(
        self,
        trade: BacktestTrade,
        price: float,
        idx: int,
        df_15m: pd.DataFrame,
        reason: str = "FORCE",
    ) -> None:
        exit_price = price * (1 - self._slippage)

        if trade.direction == "LONG":
            pnl = (exit_price - trade.entry_price) * trade.quantity * self._leverage
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity * self._leverage

        # 수수료 차감
        commission = trade.entry_price * trade.quantity * self._commission * 2  # 진입+청산
        pnl -= commission

        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.exit_idx = idx
        trade.exit_time = str(df_15m.index[idx])
        trade.pnl_usdt = pnl

        self._balance += pnl
