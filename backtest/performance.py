"""
백테스팅 성과 지표 계산기.
거래 기록으로 Sharpe, Sortino, 승률, 최대 드로다운 등을 계산한다.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class PerformanceReport:
    """성과 지표 리포트"""
    # 기본 통계
    total_trades: int
    win_count: int
    loss_count: int
    win_rate: float

    # 수익성
    total_pnl_usdt: float
    total_pnl_pct: float
    avg_win_usdt: float
    avg_loss_usdt: float
    profit_factor: float        # 총 수익 / 총 손실 (>1.5 목표)
    expectancy_usdt: float      # 트레이드당 기대 수익

    # 리스크 조정 지표
    sharpe_ratio: float         # >1.5 목표
    sortino_ratio: float        # >2.0 목표
    calmar_ratio: float         # 연 수익 / 최대 드로다운

    # 드로다운
    max_drawdown_pct: float     # <20% 목표
    max_drawdown_usdt: float

    # 거래 특성
    avg_trade_duration_hours: Optional[float]
    best_trade_usdt: float
    worst_trade_usdt: float

    def summary(self) -> str:
        return (
            f"총 거래: {self.total_trades} | "
            f"승률: {self.win_rate:.1%} | "
            f"PF: {self.profit_factor:.2f} | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"최대DD: {self.max_drawdown_pct:.1%} | "
            f"총P&L: {self.total_pnl_usdt:+.2f} USDT"
        )

    def is_acceptable(self, min_trades: int = 24) -> bool:
        """최소 기준 충족 여부 (과적합 방지: 최소 거래 수 기준 추가)"""
        return (
            self.total_trades >= min_trades
            and self.win_rate >= 0.45
            and self.profit_factor >= 1.8
            and self.sharpe_ratio >= 1.5
            and self.max_drawdown_pct <= 0.20
        )


class PerformanceAnalyzer:
    """거래 기록으로 성과 지표를 계산한다."""

    TRADING_DAYS_PER_YEAR = 365
    RISK_FREE_RATE = 0.0  # 암호화폐는 무위험 이자율 0 가정

    def __init__(
        self,
        trades: list[dict],
        initial_balance: float = 1000.0,
    ):
        self._trades = [t for t in trades if t.get("status") == "CLOSED" and t.get("pnl_usdt") is not None]
        self._initial_balance = initial_balance

    def calculate(self) -> PerformanceReport:
        if not self._trades:
            return self._empty_report()

        pnls = [float(t["pnl_usdt"]) for t in self._trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        total_pnl = sum(pnls)
        win_rate = len(wins) / len(pnls)
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = abs(np.mean(losses)) if losses else 0.0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float("inf")
        expectancy = win_rate * avg_win - (1 - win_rate) * avg_loss

        equity_curve = self._equity_curve(pnls)
        max_dd_pct, max_dd_usdt = self._max_drawdown(equity_curve)

        daily_returns = self._daily_returns(equity_curve)
        sharpe = self._sharpe(daily_returns)
        sortino = self._sortino(daily_returns)
        calmar = (total_pnl / self._initial_balance * 100) / (max_dd_pct * 100) if max_dd_pct > 0 else 0.0

        duration = self._avg_duration()

        return PerformanceReport(
            total_trades=len(self._trades),
            win_count=len(wins),
            loss_count=len(losses),
            win_rate=win_rate,
            total_pnl_usdt=total_pnl,
            total_pnl_pct=total_pnl / self._initial_balance,
            avg_win_usdt=avg_win,
            avg_loss_usdt=-avg_loss,
            profit_factor=profit_factor,
            expectancy_usdt=expectancy,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_usdt=max_dd_usdt,
            avg_trade_duration_hours=duration,
            best_trade_usdt=max(pnls),
            worst_trade_usdt=min(pnls),
        )

    def equity_curve(self) -> pd.Series:
        pnls = [float(t["pnl_usdt"]) for t in self._trades]
        return self._equity_curve(pnls)

    def _equity_curve(self, pnls: list[float]) -> pd.Series:
        values = [self._initial_balance]
        for pnl in pnls:
            values.append(values[-1] + pnl)
        return pd.Series(values)

    def _max_drawdown(self, equity: pd.Series) -> tuple[float, float]:
        roll_max = equity.cummax()
        drawdown = (equity - roll_max) / roll_max
        dd_usdt = (equity - roll_max).min()
        return abs(float(drawdown.min())), abs(float(dd_usdt))

    def _daily_returns(self, equity: pd.Series) -> pd.Series:
        return equity.pct_change().dropna()

    def _sharpe(self, returns: pd.Series) -> float:
        if returns.std() == 0:
            return 0.0
        annual = np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return float((returns.mean() - self.RISK_FREE_RATE) / returns.std() * annual)

    def _sortino(self, returns: pd.Series) -> float:
        downside = returns[returns < 0].std()
        if downside == 0:
            return 0.0
        annual = np.sqrt(self.TRADING_DAYS_PER_YEAR)
        return float((returns.mean() - self.RISK_FREE_RATE) / downside * annual)

    def _avg_duration(self) -> Optional[float]:
        durations = []
        for t in self._trades:
            if t.get("entry_time") and t.get("exit_time"):
                try:
                    entry = pd.Timestamp(t["entry_time"])
                    exit_ = pd.Timestamp(t["exit_time"])
                    durations.append((exit_ - entry).total_seconds() / 3600)
                except (ValueError, TypeError) as e:
                    logger.debug(f"거래 기간 계산 실패: {e}")
        return float(np.mean(durations)) if durations else None

    def _empty_report(self) -> PerformanceReport:
        return PerformanceReport(
            total_trades=0, win_count=0, loss_count=0, win_rate=0,
            total_pnl_usdt=0, total_pnl_pct=0, avg_win_usdt=0, avg_loss_usdt=0,
            profit_factor=0, expectancy_usdt=0, sharpe_ratio=0, sortino_ratio=0,
            calmar_ratio=0, max_drawdown_pct=0, max_drawdown_usdt=0,
            avg_trade_duration_hours=None, best_trade_usdt=0, worst_trade_usdt=0,
        )
