"""
파라미터 민감도 분석.

핵심 파라미터를 ±N% 변동시켜 Sharpe 변화율을 측정한다.
단일 파라미터 변동으로 Sharpe가 급변하면 과적합 징후로 판정한다.
"""
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
import pandas as pd
from loguru import logger

from backtest.backtester import Backtester
from backtest.performance import PerformanceReport
from config.strategy_params import AllStrategyParams, SensitivityParams


@dataclass
class ParamSensitivity:
    """단일 파라미터의 민감도 결과"""
    name: str
    base_value: float
    low_value: float
    high_value: float
    base_sharpe: float
    low_sharpe: float
    high_sharpe: float
    max_sharpe_change_pct: float  # 최대 Sharpe 변화율 (%)
    is_stable: bool               # 안정적인지 여부


@dataclass
class SensitivityReport:
    """민감도 분석 종합 결과"""
    params: list[ParamSensitivity]
    overall_stable: bool
    unstable_params: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["파라미터 민감도 분석:"]
        lines.append(f"  {'파라미터':<25} {'기본값':>8} {'−변동':>8} {'＋변동':>8} {'Sharpe변화':>10} {'안정':>4}")
        lines.append(f"  {'─'*67}")
        for p in self.params:
            stable = "✓" if p.is_stable else "✗"
            lines.append(
                f"  {p.name:<25} {p.base_value:>8.3f} {p.low_sharpe:>8.2f} {p.high_sharpe:>8.2f} "
                f"{p.max_sharpe_change_pct:>+9.1f}% {stable:>4}"
            )
        verdict = "안정" if self.overall_stable else f"불안정 ({', '.join(self.unstable_params)})"
        lines.append(f"\n  종합 판정: {verdict}")
        return "\n".join(lines)


class SensitivityAnalyzer:
    """파라미터 민감도 분석기"""

    # 분석 대상 파라미터 (optimizer_grid에서 사용하는 것들)
    PARAM_MAP = {
        "supertrend_multiplier": ("trend", "supertrend_multiplier"),
        "atr_sl_multiplier": ("risk", "atr_sl_multiplier"),
        "atr_tp1_multiplier": ("risk", "atr_tp1_multiplier"),
        "atr_tp2_multiplier": ("risk", "atr_tp2_multiplier"),
    }

    def __init__(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        params: AllStrategyParams,
        initial_balance: float = 1000.0,
        sensitivity_params: SensitivityParams = None,
    ):
        self._df_4h = df_4h
        self._df_1h = df_1h
        self._df_15m = df_15m
        self._params = params
        self._initial = initial_balance
        self._sp = sensitivity_params or getattr(params, 'sensitivity', None) or SensitivityParams()

    def analyze(self) -> SensitivityReport:
        """모든 대상 파라미터에 대해 민감도 분석을 실행한다."""
        # 기준 Sharpe 계산
        base_sharpe = self._run_backtest(self._params)
        logger.info(f"민감도 분석: 기준 Sharpe = {base_sharpe:.2f}")

        results: list[ParamSensitivity] = []

        for param_name, (section, attr) in self.PARAM_MAP.items():
            base_val = getattr(getattr(self._params, section), attr)

            low_val = base_val * (1 - self._sp.perturbation_pct)
            high_val = base_val * (1 + self._sp.perturbation_pct)

            # −변동 백테스트
            low_params = self._perturb(section, attr, low_val)
            low_sharpe = self._run_backtest(low_params)

            # +변동 백테스트
            high_params = self._perturb(section, attr, high_val)
            high_sharpe = self._run_backtest(high_params)

            # Sharpe 변화율 계산
            if abs(base_sharpe) > 0.01:
                change_low = abs(low_sharpe - base_sharpe) / abs(base_sharpe) * 100
                change_high = abs(high_sharpe - base_sharpe) / abs(base_sharpe) * 100
                max_change = max(change_low, change_high)
            else:
                max_change = 0.0

            is_stable = max_change < self._sp.max_sharpe_drop_pct * 100

            result = ParamSensitivity(
                name=param_name,
                base_value=base_val,
                low_value=low_val,
                high_value=high_val,
                base_sharpe=base_sharpe,
                low_sharpe=low_sharpe,
                high_sharpe=high_sharpe,
                max_sharpe_change_pct=max_change,
                is_stable=is_stable,
            )
            results.append(result)
            logger.info(
                f"  {param_name}: base={base_val:.3f} → "
                f"low({low_val:.3f})={low_sharpe:.2f}, high({high_val:.3f})={high_sharpe:.2f}, "
                f"변화={max_change:.1f}% {'안정' if is_stable else '불안정'}"
            )

        unstable = [r.name for r in results if not r.is_stable]
        report = SensitivityReport(
            params=results,
            overall_stable=len(unstable) == 0,
            unstable_params=unstable,
        )
        logger.info(f"\n{report.summary()}")
        return report

    def _perturb(self, section: str, attr: str, new_value: float) -> AllStrategyParams:
        """파라미터를 변경한 복사본을 반환한다."""
        params_copy = deepcopy(self._params)
        section_obj = getattr(params_copy, section)
        setattr(section_obj, attr, new_value)
        return params_copy

    def _run_backtest(self, params: AllStrategyParams) -> float:
        """백테스트를 실행하고 Sharpe ratio를 반환한다."""
        try:
            bt = Backtester(params, self._initial)
            report = bt.run(self._df_4h, self._df_1h, self._df_15m)
            sharpe = report.sharpe_ratio
            if np.isnan(sharpe) or np.isinf(sharpe):
                return 0.0
            return sharpe
        except (KeyError, ValueError, ZeroDivisionError) as e:
            logger.warning(f"민감도 백테스트 실패: {e}")
            return 0.0
