"""
몬테카를로 시뮬레이션.
거래 결과를 무작위로 셔플하여 전략의 통계적 강건성을 평가한다.
"""
import numpy as np
from dataclasses import dataclass
from loguru import logger


@dataclass
class MonteCarloResult:
    """몬테카를로 시뮬레이션 결과"""
    n_simulations: int
    median_pnl: float
    pct_5_pnl: float           # 5th percentile P&L (worst case)
    pct_95_pnl: float          # 95th percentile P&L (best case)
    median_max_drawdown: float
    pct_95_max_drawdown: float  # 95th percentile MDD (worst case)
    prob_profitable: float      # 수익 확률 (PnL > 0 비율)
    prob_mdd_under_20: float    # MDD < 20% 확률

    def summary(self) -> str:
        return (
            f"몬테카를로 ({self.n_simulations}회)\n"
            f"  P&L 중앙값: ${self.median_pnl:,.2f} | "
            f"5%ile: ${self.pct_5_pnl:,.2f} | 95%ile: ${self.pct_95_pnl:,.2f}\n"
            f"  MDD 중앙값: {self.median_max_drawdown:.1%} | "
            f"95%ile: {self.pct_95_max_drawdown:.1%}\n"
            f"  수익 확률: {self.prob_profitable:.1%} | "
            f"MDD<20% 확률: {self.prob_mdd_under_20:.1%}"
        )

    def is_robust(self) -> bool:
        """강건성 기준 충족 여부"""
        return (
            self.prob_profitable >= 0.6
            and self.pct_95_max_drawdown <= 0.25
            and self.pct_5_pnl > -100  # 최악 5%ile에서도 $100 이상 손실 아님
        )


class MonteCarloSimulator:
    """거래 결과 순서 셔플 몬테카를로"""

    def __init__(self, initial_balance: float = 1000.0, n_simulations: int = 1000):
        self._initial = initial_balance
        self._n_sims = n_simulations

    def run(self, trade_pnls: list[float]) -> MonteCarloResult:
        """거래 P&L 리스트를 입력받아 몬테카를로 시뮬레이션을 실행한다.

        Args:
            trade_pnls: 각 거래의 P&L (USDT)

        Returns:
            MonteCarloResult
        """
        if not trade_pnls or len(trade_pnls) < 5:
            logger.warning("몬테카를로: 거래 수 부족 (최소 5건 필요)")
            return MonteCarloResult(
                n_simulations=0, median_pnl=0, pct_5_pnl=0, pct_95_pnl=0,
                median_max_drawdown=0, pct_95_max_drawdown=0,
                prob_profitable=0, prob_mdd_under_20=0,
            )

        pnl_array = np.array(trade_pnls)
        n_trades = len(pnl_array)
        rng = np.random.default_rng(42)

        final_pnls = []
        max_drawdowns = []

        for _ in range(self._n_sims):
            shuffled = rng.permutation(pnl_array)
            equity = np.empty(n_trades + 1)
            equity[0] = self._initial
            for j in range(n_trades):
                equity[j + 1] = equity[j] + shuffled[j]

            final_pnls.append(equity[-1] - self._initial)

            # Max drawdown 계산
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / np.where(peak > 0, peak, 1)
            max_drawdowns.append(abs(dd.min()))

        final_pnls = np.array(final_pnls)
        max_drawdowns = np.array(max_drawdowns)

        result = MonteCarloResult(
            n_simulations=self._n_sims,
            median_pnl=float(np.median(final_pnls)),
            pct_5_pnl=float(np.percentile(final_pnls, 5)),
            pct_95_pnl=float(np.percentile(final_pnls, 95)),
            median_max_drawdown=float(np.median(max_drawdowns)),
            pct_95_max_drawdown=float(np.percentile(max_drawdowns, 95)),
            prob_profitable=float(np.mean(final_pnls > 0)),
            prob_mdd_under_20=float(np.mean(max_drawdowns < 0.20)),
        )

        logger.info(f"몬테카를로 완료:\n{result.summary()}")
        return result
