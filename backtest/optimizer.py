"""
전략 파라미터 최적화기 (그리드 서치).
strategy_params.yaml의 optimizer_grid 범위로 최적 파라미터를 탐색한다.

과적합 방지 개선:
- 최소 거래 수 미달 조합은 제외
- Sharpe ratio에 자유도 페널티 적용 (Adjusted Sharpe)
- 파라미터 수 대비 거래 수 비율 기록
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from backtest.backtester import Backtester
from backtest.performance import PerformanceReport
from config.strategy_params import AllStrategyParams, TrendFilterParams, MomentumParams, MicroParams, RiskParams


class GridOptimizer:
    """파라미터 그리드 서치 (정규화 적용)"""

    def __init__(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        initial_balance: float = 1000.0,
        min_trades: int = None,
    ):
        self._df_4h = df_4h
        self._df_1h = df_1h
        self._df_15m = df_15m
        self._initial = initial_balance
        # min_trades를 YAML에서 로드 (인자 미지정 시)
        if min_trades is None:
            yaml_path = Path(__file__).parent.parent / "config" / "strategy_params.yaml"
            with open(yaml_path) as f:
                cfg = yaml.safe_load(f)
            self._min_trades = cfg.get("min_trades", 15)
        else:
            self._min_trades = min_trades
        self._results: list[dict] = []

    def run(self) -> pd.DataFrame:
        """그리드 서치를 실행하고 결과 DataFrame을 반환한다."""
        grid = self._load_grid()
        combinations = list(itertools.product(*grid.values()))
        keys = list(grid.keys())
        n_params = len(keys)

        logger.info(f"그리드 서치 시작: {len(combinations)}개 조합 (파라미터 {n_params}개, 최소 거래 {self._min_trades}회)")

        for i, combo in enumerate(combinations):
            params_dict = dict(zip(keys, combo))
            try:
                params = self._build_params(params_dict)
                bt = Backtester(params, self._initial)
                report = bt.run(self._df_4h, self._df_1h, self._df_15m)
                row = {**params_dict, **self._report_to_dict(report, n_params)}
                self._results.append(row)

                if (i + 1) % 10 == 0:
                    logger.info(f"진행: {i + 1}/{len(combinations)}")
            except (KeyError, ValueError, ZeroDivisionError, TypeError) as e:
                logger.warning(f"조합 {params_dict} 실패: {e}")

        if not self._results:
            return pd.DataFrame()

        df = pd.DataFrame(self._results)

        # 최소 거래 수 미달 조합 필터링
        valid = df[df["total_trades"] >= self._min_trades]
        if valid.empty:
            logger.warning(f"최소 거래 수({self._min_trades}회) 충족 조합 없음. 전체 결과 반환.")
            df = df.sort_values("adjusted_sharpe", ascending=False)
        else:
            df = valid.sort_values("adjusted_sharpe", ascending=False)

        logger.info(f"\n최적 파라미터 Top 5 (Adjusted Sharpe 기준):\n{df.head(5).to_string()}")
        return df

    def _load_grid(self) -> dict:
        yaml_path = Path(__file__).parent.parent / "config" / "strategy_params.yaml"
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("optimizer_grid", {})

    def _build_params(self, p: dict) -> AllStrategyParams:
        from config.strategy_params import (
            FiltersParams, ScoringParams, SentimentParams, MacroParams,
            BacktestParams, RegimeParams, TradingHoursParams,
        )
        return AllStrategyParams(
            trend=TrendFilterParams(
                ema_fast=p.get("ema_fast", 21),
                ema_mid=p.get("ema_mid", 55),
                ema_slow=200,
                supertrend_multiplier=p.get("supertrend_multiplier", 3.0),
            ),
            momentum=MomentumParams(
                rsi_long_min=p.get("rsi_long_min", 40),
                rsi_long_max=p.get("rsi_long_max", 65),
            ),
            micro=MicroParams(),
            risk=RiskParams(
                atr_sl_multiplier=p.get("atr_sl_multiplier", 1.5),
                atr_tp1_multiplier=p.get("atr_tp1_multiplier", 2.0),
                atr_tp2_multiplier=p.get("atr_tp2_multiplier", 3.5),
            ),
            filters=FiltersParams(),
            scoring=ScoringParams(),
            sentiment=SentimentParams(),
            macro=MacroParams(),
            backtest=BacktestParams.from_yaml(),
            regime=RegimeParams(),
            trading_hours=TradingHoursParams(),
        )

    @staticmethod
    def _adjusted_sharpe(sharpe: float, n_trades: int, n_params: int) -> float:
        """자유도 보정 Sharpe: 거래 수가 적거나 파라미터가 많을수록 페널티 부여.

        공식: adjusted = sharpe × sqrt(1 - n_params / n_trades)
        거래 수가 파라미터 수 이하이면 0 반환.
        """
        if n_trades <= n_params or np.isnan(sharpe) or np.isinf(sharpe):
            return 0.0
        penalty = np.sqrt(1.0 - n_params / n_trades)
        return sharpe * penalty

    def _report_to_dict(self, r: PerformanceReport, n_params: int) -> dict:
        adj_sharpe = self._adjusted_sharpe(r.sharpe_ratio, r.total_trades, n_params)
        return {
            "total_trades": r.total_trades,
            "win_rate": r.win_rate,
            "profit_factor": r.profit_factor,
            "sharpe_ratio": r.sharpe_ratio,
            "adjusted_sharpe": round(adj_sharpe, 4),
            "max_drawdown_pct": r.max_drawdown_pct,
            "total_pnl_usdt": r.total_pnl_usdt,
            "trades_per_param": round(r.total_trades / max(n_params, 1), 1),
            "acceptable": r.is_acceptable(min_trades=self._min_trades if hasattr(r, 'is_acceptable') else 24),
        }
