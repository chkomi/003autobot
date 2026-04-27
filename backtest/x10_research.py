"""
10x 목표 타당성 연구 백테스터.

목표: $1,096 → $10,960 (12개월, 월 21.44% 복리)
방법: 3개 시나리오 × 고베타 알트코인 심볼 조합으로 백테스팅 후 타당성 평가

사용법:
    python -m backtest.x10_research
    python -m backtest.x10_research --from 2024-01-01 --to 2024-12-31 --scenario balanced
"""
import argparse
import asyncio
import csv
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

# ── 10x 목표 상수 ──────────────────────────────────────────
INITIAL_CAPITAL = 1096.0
TARGET_CAPITAL = 10960.0
MONTHLY_TARGET_PCT = 0.2144  # 10^(1/12) - 1

# 월별 목표 자산 스케줄 (복리)
MONTHLY_SCHEDULE = [INITIAL_CAPITAL * (1 + MONTHLY_TARGET_PCT) ** m for m in range(1, 13)]

# 시나리오 정의
SCENARIOS = {
    "conservative": {
        "leverage": 3,
        "max_position_pct": 0.10,
        "atr_sl_multiplier": 1.5,
        "atr_tp1_multiplier": 2.0,
        "atr_tp2_multiplier": 3.5,
        "signal_mode": "OR",
        "max_trade_hours": 48,
    },
    "balanced": {
        "leverage": 5,
        "max_position_pct": 0.12,
        "atr_sl_multiplier": 1.5,
        "atr_tp1_multiplier": 2.0,
        "atr_tp2_multiplier": 4.5,
        "signal_mode": "OR",
        "max_trade_hours": 72,
    },
    "aggressive": {
        "leverage": 5,
        "max_position_pct": 0.15,
        "atr_sl_multiplier": 1.5,
        "atr_tp1_multiplier": 2.0,
        "atr_tp2_multiplier": 5.0,
        "signal_mode": "AND",
        "max_trade_hours": 96,
    },
}

# 후보 심볼 (고베타 알트코인)
CANDIDATE_SYMBOLS = [
    "AVAX/USDT:USDT",
    "SOL/USDT:USDT",
    "SUI/USDT:USDT",
    "LINK/USDT:USDT",
    "ARB/USDT:USDT",
    "INJ/USDT:USDT",
]


# ── 결과 데이터클래스 ──────────────────────────────────────

@dataclass
class MonthlyResult:
    month: int
    start_equity: float
    end_equity: float
    pnl_usdt: float
    pnl_pct: float
    trade_count: int
    win_count: int
    target_equity: float

    @property
    def win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count > 0 else 0.0

    @property
    def target_achieved(self) -> bool:
        return self.pnl_pct >= MONTHLY_TARGET_PCT

    @property
    def vs_schedule(self) -> float:
        """스케줄 대비 달성률 (1.0 = 목표 정확히 달성)"""
        if self.target_equity <= 0:
            return 0.0
        return self.end_equity / self.target_equity


@dataclass
class ScenarioResult:
    scenario: str
    symbol: str
    from_date: str
    to_date: str

    initial_equity: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float

    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown_pct: float
    avg_trade_duration_hours: float

    monthly_results: list[MonthlyResult] = field(default_factory=list)

    # 10x 관련 지표
    months_on_schedule: int = 0        # 스케줄 달성 월 수
    months_above_schedule: int = 0     # 스케줄 초과 월 수
    projected_months_to_10x: float = 0.0  # 현재 페이스로 10x까지 소요 월

    @property
    def feasibility_score(self) -> float:
        """10x 타당성 점수 (0~100). 높을수록 달성 가능성 높음."""
        score = 0.0
        # 연수익률 기여 (40점)
        if self.annualized_return_pct >= 9.0:   # 900% = 10x
            score += 40.0
        elif self.annualized_return_pct >= 4.0:  # 400%
            score += 20.0
        elif self.annualized_return_pct >= 1.0:  # 100%
            score += 10.0

        # 승률 기여 (20점)
        score += min(self.win_rate * 30, 20.0)

        # 드로다운 기여 (20점) — 낮을수록 좋음
        dd_score = max(0, 20.0 - self.max_drawdown_pct * 100)
        score += dd_score

        # 스케줄 달성 월 수 기여 (20점)
        score += min(self.months_on_schedule * 2.0, 20.0)

        return round(score, 1)

    @property
    def verdict(self) -> str:
        s = self.feasibility_score
        if s >= 70:
            return "✅ FEASIBLE"
        elif s >= 45:
            return "⚠️ POSSIBLE (조건부)"
        else:
            return "❌ UNLIKELY"


# ── 심플 백테스터 ──────────────────────────────────────────

class X10Backtester:
    """10x 연구용 단순화 백테스터 (전략 레이어 재사용)"""

    OKX_FEE = 0.0005  # taker fee per side

    def __init__(
        self,
        scenario_name: str,
        params,
        initial_balance: float = INITIAL_CAPITAL,
    ):
        from strategy.trend_filter import TrendFilter
        from strategy.momentum_trigger import MomentumTrigger
        from strategy.micro_confirmation import MicroConfirmation

        cfg = SCENARIOS[scenario_name]
        self._scenario = scenario_name
        self._leverage = cfg["leverage"]
        self._max_pos_pct = cfg["max_position_pct"]
        self._atr_sl = cfg["atr_sl_multiplier"]
        self._atr_tp1 = cfg["atr_tp1_multiplier"]
        self._atr_tp2 = cfg["atr_tp2_multiplier"]
        self._signal_mode = cfg["signal_mode"]
        self._max_hours = cfg["max_trade_hours"]

        self._balance = initial_balance
        self._initial = initial_balance
        self._params = params

        self._trend = TrendFilter(params.trend)
        self._momentum = MomentumTrigger(params.momentum)
        self._micro = MicroConfirmation(params.micro)

        self._trades = []
        self._equity_curve = [initial_balance]

    def run(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        symbol: str = "AVAX/USDT:USDT",
    ) -> dict:
        """백테스트 실행 후 결과 dict 반환"""
        from strategy.indicators import calc_atr

        if df_4h is None or df_1h is None or df_15m is None:
            logger.warning(f"데이터 없음: {symbol}")
            return {}
        if len(df_4h) < 50 or len(df_1h) < 50:
            logger.warning(f"데이터 부족: {symbol} 4H={len(df_4h)} 1H={len(df_1h)}")
            return {}

        open_trade = None
        trades = []
        equity = self._initial
        equity_curve = [equity]
        peak_equity = equity

        for i in range(50, len(df_1h)):
            ts_1h = df_1h.index[i]

            # 1H 기준으로 4H/15M 슬라이스
            df4_slice = df_4h[df_4h.index <= ts_1h].tail(260)
            df15_slice = df_15m[df_15m.index <= ts_1h].tail(260)

            if len(df4_slice) < 210:
                continue

            # ATR (1H)
            df1_slice = df_1h.iloc[:i+1]
            atr_series = calc_atr(df1_slice["high"], df1_slice["low"], df1_slice["close"], period=14)
            if atr_series.empty or pd.isna(atr_series.iloc[-1]):
                continue
            atr = float(atr_series.iloc[-1])
            close = float(df_1h["close"].iloc[i])

            # ── 오픈 포지션 관리 ──
            if open_trade is not None:
                t = open_trade
                duration_hours = (ts_1h - pd.Timestamp(t["entry_time"])).total_seconds() / 3600

                # TP2 체크
                if t["direction"] == "LONG":
                    if close >= t["tp2_price"]:
                        pnl = self._calc_pnl(t, close, t["quantity"])
                        equity += pnl
                        equity_curve.append(equity)
                        peak_equity = max(peak_equity, equity)
                        trades.append({**t, "exit_price": close, "exit_reason": "TP2",
                                       "pnl_usdt": pnl, "duration_hours": duration_hours,
                                       "exit_time": str(ts_1h)})
                        open_trade = None
                        continue
                    elif close <= t["stop_price"]:
                        pnl = self._calc_pnl(t, close, t["quantity"])
                        equity += pnl
                        equity_curve.append(equity)
                        trades.append({**t, "exit_price": close, "exit_reason": "SL",
                                       "pnl_usdt": pnl, "duration_hours": duration_hours,
                                       "exit_time": str(ts_1h)})
                        open_trade = None
                        continue
                else:  # SHORT
                    if close <= t["tp2_price"]:
                        pnl = self._calc_pnl(t, close, t["quantity"])
                        equity += pnl
                        equity_curve.append(equity)
                        peak_equity = max(peak_equity, equity)
                        trades.append({**t, "exit_price": close, "exit_reason": "TP2",
                                       "pnl_usdt": pnl, "duration_hours": duration_hours,
                                       "exit_time": str(ts_1h)})
                        open_trade = None
                        continue
                    elif close >= t["stop_price"]:
                        pnl = self._calc_pnl(t, close, t["quantity"])
                        equity += pnl
                        equity_curve.append(equity)
                        trades.append({**t, "exit_price": close, "exit_reason": "SL",
                                       "pnl_usdt": pnl, "duration_hours": duration_hours,
                                       "exit_time": str(ts_1h)})
                        open_trade = None
                        continue

                # 시간 컷
                if duration_hours >= self._max_hours:
                    pnl = self._calc_pnl(t, close, t["quantity"])
                    equity += pnl
                    equity_curve.append(equity)
                    trades.append({**t, "exit_price": close, "exit_reason": "TIME",
                                   "pnl_usdt": pnl, "duration_hours": duration_hours,
                                   "exit_time": str(ts_1h)})
                    open_trade = None
                continue

            # ── 새 진입 시그널 탐색 ──
            if equity <= 0:
                break

            try:
                trend_sig = self._trend.analyze(df4_slice)
                if trend_sig is None or trend_sig.regime == "NEUTRAL":
                    continue
                direction = "LONG" if trend_sig.is_bullish else "SHORT"

                mom_sig = self._momentum.analyze(df1_slice)
                micro_sig = self._micro.analyze(df15_slice)
                if mom_sig is None or micro_sig is None:
                    continue

                if direction == "LONG":
                    mom_ok = mom_sig.long_trigger
                    micro_ok = micro_sig.long_confirm
                else:
                    mom_ok = mom_sig.short_trigger
                    micro_ok = micro_sig.short_confirm

                # 시그널 결합
                if self._signal_mode == "AND":
                    if not (mom_ok and micro_ok):
                        continue
                else:  # OR
                    if not (mom_ok or micro_ok):
                        continue

                entry_price = close
                stop_price = (entry_price - atr * self._atr_sl if direction == "LONG"
                              else entry_price + atr * self._atr_sl)
                tp1_price = (entry_price + atr * self._atr_tp1 if direction == "LONG"
                             else entry_price - atr * self._atr_tp1)
                tp2_price = (entry_price + atr * self._atr_tp2 if direction == "LONG"
                             else entry_price - atr * self._atr_tp2)

                # Kelly 기반 포지션 사이징 (간이)
                risk_pct = abs(entry_price - stop_price) / entry_price
                if risk_pct < 0.001:
                    risk_pct = 0.01
                risk_amount = equity * 0.25 * 0.10  # 25% Kelly × 10% max_pos
                notional = min(risk_amount / risk_pct,
                               equity * self._max_pos_pct * self._leverage)
                margin = notional / self._leverage
                if margin > equity * 0.95:
                    margin = equity * 0.95
                    notional = margin * self._leverage
                quantity = notional / entry_price

                open_trade = {
                    "symbol": symbol,
                    "direction": direction,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "tp1_price": tp1_price,
                    "tp2_price": tp2_price,
                    "quantity": quantity,
                    "notional": notional,
                    "entry_time": str(ts_1h),
                    "atr": atr,
                }
            except Exception as e:
                logger.debug(f"시그널 평가 오류: {e}")
                continue

        # 미결 포지션 청산
        if open_trade is not None and len(df_1h) > 0:
            close = float(df_1h["close"].iloc[-1])
            pnl = self._calc_pnl(open_trade, close, open_trade["quantity"])
            equity += pnl
            ts_end = df_1h.index[-1]
            duration_hours = (ts_end - pd.Timestamp(open_trade["entry_time"])).total_seconds() / 3600
            trades.append({**open_trade, "exit_price": close, "exit_reason": "END",
                           "pnl_usdt": pnl, "duration_hours": duration_hours,
                           "exit_time": str(ts_end)})

        return {
            "trades": trades,
            "final_equity": equity,
            "equity_curve": equity_curve,
            "peak_equity": peak_equity,
        }

    def _calc_pnl(self, trade: dict, exit_price: float, qty: float) -> float:
        entry = trade["entry_price"]
        direction = trade["direction"]
        fee = (entry + exit_price) * qty * self.OKX_FEE
        if direction == "LONG":
            return (exit_price - entry) * qty - fee
        else:
            return (entry - exit_price) * qty - fee


# ── 분석 함수 ──────────────────────────────────────────────

def _compute_monthly_results(
    trades: list[dict],
    initial_equity: float,
    from_date: str,
    to_date: str,
) -> list[MonthlyResult]:
    """거래 목록에서 월별 성과를 계산한다."""
    start_ts = pd.Timestamp(from_date, tz="UTC")
    monthly_results = []
    equity = initial_equity

    def _to_utc(ts_str: str) -> Optional[pd.Timestamp]:
        try:
            ts = pd.Timestamp(ts_str)
        except Exception:
            return None
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    for month_idx in range(1, 13):
        month_start = start_ts + pd.DateOffset(months=month_idx - 1)
        month_end = start_ts + pd.DateOffset(months=month_idx)
        target_equity = MONTHLY_SCHEDULE[month_idx - 1]

        month_trades = []
        for t in trades:
            exit_time_str = t.get("exit_time")
            if not exit_time_str:
                continue
            ts = _to_utc(exit_time_str)
            if ts is None:
                continue
            if month_start <= ts < month_end:
                month_trades.append(t)

        start_eq = equity
        pnl = sum(t.get("pnl_usdt", 0) for t in month_trades)
        equity += pnl
        wins = sum(1 for t in month_trades if t.get("pnl_usdt", 0) > 0)

        mr = MonthlyResult(
            month=month_idx,
            start_equity=start_eq,
            end_equity=equity,
            pnl_usdt=pnl,
            pnl_pct=pnl / start_eq if start_eq > 0 else 0.0,
            trade_count=len(month_trades),
            win_count=wins,
            target_equity=target_equity,
        )
        monthly_results.append(mr)

        # 날짜 범위 초과 시 중단
        if month_end > pd.Timestamp(to_date, tz="UTC") + pd.DateOffset(days=1):
            break

    return monthly_results


def _compute_metrics(
    trades: list[dict],
    initial: float,
    final: float,
    from_date: str,
    to_date: str,
) -> dict:
    """성과 지표 계산."""
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0, "profit_factor": 0,
            "sharpe_ratio": 0, "max_drawdown_pct": 0, "avg_duration": 0,
        }

    pnls = [t.get("pnl_usdt", 0) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]
    win_rate = len(wins) / len(pnls)
    profit_factor = sum(wins) / sum(losses) if losses else float("inf")

    # 드로다운 (트레이드별 복리 누적)
    equity = initial
    peak = initial
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak
        max_dd = max(max_dd, dd)

    # Sharpe (월 수익률 기반)
    days = (pd.Timestamp(to_date) - pd.Timestamp(from_date)).days or 365
    months = days / 30.0
    returns_usdt = [sum(pnls[i::max(1, len(pnls) // max(1, int(months)))])
                    for i in range(max(1, int(months)))]
    if len(returns_usdt) > 1:
        import statistics
        mean_r = statistics.mean(returns_usdt)
        std_r = statistics.stdev(returns_usdt)
        sharpe = (mean_r / std_r * math.sqrt(12)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    durations = [t.get("duration_hours", 0) for t in trades]
    avg_duration = sum(durations) / len(durations) if durations else 0

    total_return = (final - initial) / initial
    years = days / 365.0
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    return {
        "total_trades": len(pnls),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "avg_duration": avg_duration,
        "total_return_pct": total_return,
        "annualized_return_pct": ann_return,
    }


def _projected_months_to_10x(monthly_results: list[MonthlyResult]) -> float:
    """현재 월 평균 수익률로 10x 달성까지 소요 월 수 추정."""
    valid = [m for m in monthly_results if m.trade_count > 0]
    if not valid:
        return float("inf")
    avg_monthly = sum(m.pnl_pct for m in valid) / len(valid)
    if avg_monthly <= 0:
        return float("inf")
    # 1096 * (1 + r)^n >= 10960 → n >= log(10) / log(1+r)
    return math.log(10) / math.log(1 + avg_monthly)


# ── 메인 연구 함수 ────────────────────────────────────────

async def run_x10_research(
    from_date: str = "2024-01-01",
    to_date: str = "2024-12-31",
    scenario_filter: Optional[str] = None,
    symbol_filter: Optional[str] = None,
    output_csv: Optional[Path] = None,
) -> list[ScenarioResult]:
    """10x 타당성 연구 실행."""
    from config.settings import load_settings
    from config.strategy_params import AllStrategyParams
    from data.okx_rest_client import OKXRestClient
    from backtest.data_loader import HistoricalDataLoader

    settings = load_settings()
    rest = OKXRestClient(settings.okx)
    loader = HistoricalDataLoader(rest)
    params = AllStrategyParams.from_yaml()

    scenarios = ([scenario_filter] if scenario_filter else list(SCENARIOS.keys()))
    symbols = ([symbol_filter] if symbol_filter else CANDIDATE_SYMBOLS)

    results: list[ScenarioResult] = []

    try:
        for symbol in symbols:
            logger.info(f"\n{'='*60}")
            logger.info(f"심볼: {symbol} | 기간: {from_date} ~ {to_date}")
            logger.info(f"{'='*60}")

            # 데이터 수집 (심볼별 1회)
            logger.info(f"[{symbol}] 과거 데이터 수집 중...")
            try:
                df_4h = await loader.fetch_full_history(symbol, "4h", from_date, to_date)
                df_1h = await loader.fetch_full_history(symbol, "1h", from_date, to_date)
                df_15m = await loader.fetch_full_history(symbol, "15m", from_date, to_date)
            except Exception as e:
                logger.warning(f"[{symbol}] 데이터 수집 실패: {e}")
                continue

            if df_1h is None or len(df_1h) < 100:
                logger.warning(f"[{symbol}] 데이터 부족 — 스킵")
                continue

            for scenario_name in scenarios:
                logger.info(f"  시나리오: {scenario_name.upper()}")
                cfg = SCENARIOS[scenario_name]

                bt = X10Backtester(
                    scenario_name=scenario_name,
                    params=params,
                    initial_balance=INITIAL_CAPITAL,
                )

                raw = bt.run(df_4h, df_1h, df_15m, symbol=symbol)
                if not raw:
                    logger.warning(f"  [{scenario_name}] 결과 없음 — 스킵")
                    continue

                trades = raw.get("trades", [])
                final_equity = raw.get("final_equity", INITIAL_CAPITAL)

                metrics = _compute_metrics(trades, INITIAL_CAPITAL, final_equity, from_date, to_date)
                monthly = _compute_monthly_results(trades, INITIAL_CAPITAL, from_date, to_date)

                on_schedule = sum(1 for m in monthly if m.vs_schedule >= 1.0)
                above_schedule = sum(1 for m in monthly if m.vs_schedule > 1.1)
                proj_months = _projected_months_to_10x(monthly)

                result = ScenarioResult(
                    scenario=scenario_name,
                    symbol=symbol,
                    from_date=from_date,
                    to_date=to_date,
                    initial_equity=INITIAL_CAPITAL,
                    final_equity=final_equity,
                    total_return_pct=metrics.get("total_return_pct", 0),
                    annualized_return_pct=metrics.get("annualized_return_pct", 0),
                    total_trades=metrics["total_trades"],
                    win_rate=metrics["win_rate"],
                    profit_factor=metrics["profit_factor"],
                    sharpe_ratio=metrics["sharpe_ratio"],
                    max_drawdown_pct=metrics["max_drawdown_pct"],
                    avg_trade_duration_hours=metrics["avg_duration"],
                    monthly_results=monthly,
                    months_on_schedule=on_schedule,
                    months_above_schedule=above_schedule,
                    projected_months_to_10x=proj_months,
                )
                results.append(result)

                logger.info(
                    f"  결과: {result.verdict} | "
                    f"수익률={result.total_return_pct:+.1%} | "
                    f"승률={result.win_rate:.1%} | "
                    f"MDD={result.max_drawdown_pct:.1%} | "
                    f"타당성={result.feasibility_score:.0f}점"
                )

    finally:
        await rest.close()

    # ── 리포트 출력 ──
    _print_report(results, from_date, to_date)

    if output_csv:
        _save_csv(results, output_csv)

    return results


def _print_report(results: list[ScenarioResult], from_date: str, to_date: str) -> None:
    """콘솔 리포트 출력."""
    print("\n" + "=" * 70)
    print(f"  10x 목표 타당성 연구 리포트")
    print(f"  기간: {from_date} ~ {to_date}")
    print(f"  목표: ${INITIAL_CAPITAL:,.0f} → ${TARGET_CAPITAL:,.0f} (12개월, 월 {MONTHLY_TARGET_PCT:.2%})")
    print("=" * 70)

    if not results:
        print("  결과 없음")
        return

    # 시나리오별 요약
    for scenario in SCENARIOS.keys():
        scene_results = [r for r in results if r.scenario == scenario]
        if not scene_results:
            continue

        print(f"\n▶ 시나리오: {scenario.upper()}")
        cfg = SCENARIOS[scenario]
        print(f"  레버리지: {cfg['leverage']}x | 최대포지션: {cfg['max_position_pct']:.0%} | "
              f"신호모드: {cfg['signal_mode']} | 최대보유: {cfg['max_trade_hours']}h")
        print(f"  {'심볼':<22} {'수익률':>8} {'연환산':>8} {'승률':>7} {'MDD':>7} {'PF':>6} {'타당성':>8} {'판정'}")
        print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*8} {'-'*12}")

        for r in scene_results:
            sym = r.symbol.replace("/USDT:USDT", "")
            pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 99 else "∞"
            print(
                f"  {sym:<22} {r.total_return_pct:>+8.1%} {r.annualized_return_pct:>+8.1%} "
                f"{r.win_rate:>7.1%} {r.max_drawdown_pct:>7.1%} {pf_str:>6} "
                f"{r.feasibility_score:>8.0f}점 {r.verdict}"
            )

    # 베스트 조합
    if results:
        best = max(results, key=lambda r: r.feasibility_score)
        print(f"\n{'='*70}")
        print(f"  🏆 최고 조합: {best.symbol} × {best.scenario.upper()}")
        print(f"  타당성 점수: {best.feasibility_score:.0f}점 | {best.verdict}")
        print(f"  연환산 수익률: {best.annualized_return_pct:+.1%}")
        print(f"  스케줄 달성 월: {best.months_on_schedule}개월")
        proj = best.projected_months_to_10x
        if proj < float("inf"):
            print(f"  현재 페이스로 10x 달성: {proj:.1f}개월 예상")
        else:
            print(f"  현재 페이스로 10x 달성: 불가 (수익률 부족)")

        # 월별 스케줄 추적
        if best.monthly_results:
            print(f"\n  📅 월별 달성 현황 ({best.symbol} × {best.scenario.upper()})")
            print(f"  {'월':>4} {'시작자산':>12} {'종료자산':>12} {'수익률':>8} {'목표자산':>12} {'달성률':>8} {'거래':>5} {'승률':>7}")
            for m in best.monthly_results:
                vs = f"{m.vs_schedule:.1%}"
                wr_str = f"{m.win_rate:.1%}" if m.trade_count > 0 else "  -"
                flag = "✅" if m.target_achieved else ("🔴" if m.pnl_pct < 0 else "🟡")
                print(
                    f"  {m.month:>4} {m.start_equity:>12,.2f} {m.end_equity:>12,.2f} "
                    f"{m.pnl_pct:>+8.2%} {m.target_equity:>12,.2f} {vs:>8} "
                    f"{m.trade_count:>5} {wr_str:>7} {flag}"
                )

    print(f"\n{'='*70}\n")


def _save_csv(results: list[ScenarioResult], path: Path) -> None:
    """결과를 CSV로 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        rows.append({
            "scenario": r.scenario,
            "symbol": r.symbol,
            "from_date": r.from_date,
            "to_date": r.to_date,
            "initial_equity": r.initial_equity,
            "final_equity": round(r.final_equity, 2),
            "total_return_pct": round(r.total_return_pct * 100, 2),
            "annualized_return_pct": round(r.annualized_return_pct * 100, 2),
            "total_trades": r.total_trades,
            "win_rate": round(r.win_rate * 100, 2),
            "profit_factor": round(r.profit_factor, 3),
            "sharpe_ratio": round(r.sharpe_ratio, 3),
            "max_drawdown_pct": round(r.max_drawdown_pct * 100, 2),
            "avg_trade_hours": round(r.avg_trade_duration_hours, 1),
            "months_on_schedule": r.months_on_schedule,
            "projected_months_to_10x": round(r.projected_months_to_10x, 1),
            "feasibility_score": r.feasibility_score,
            "verdict": r.verdict,
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"결과 저장: {path}")


# ── CLI 진입점 ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="10x 타당성 연구 백테스터")
    parser.add_argument("--from", dest="from_date", default="2024-01-01")
    parser.add_argument("--to", dest="to_date", default="2024-12-31")
    parser.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                        help="특정 시나리오만 실행 (기본: 전체)")
    parser.add_argument("--symbol", help="특정 심볼만 실행 (예: AVAX/USDT:USDT)")
    parser.add_argument("--output", default="logs/x10_research.csv",
                        help="CSV 출력 경로")
    args = parser.parse_args()

    asyncio.run(run_x10_research(
        from_date=args.from_date,
        to_date=args.to_date,
        scenario_filter=args.scenario,
        symbol_filter=args.symbol,
        output_csv=Path(args.output),
    ))


if __name__ == "__main__":
    main()
