"""
아웃오브샘플(OOS) 프로그레시브 검증기.

3단계 검증 파이프라인:
  Stage 1 (Discovery): 완화 기준, min_trades=10, total_pnl > 0
  Stage 2 (Walk-Forward): 45/15일 윈도우, robustness ≥ 40
  Stage 3 (OOS Test): 마지막 20% 데이터로 최종 검증
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from loguru import logger

from backtest.backtester import Backtester
from backtest.performance import PerformanceAnalyzer, PerformanceReport
from backtest.walk_forward import WalkForwardValidator, WalkForwardResult
from config.strategy_params import AllStrategyParams


@dataclass
class StageResult:
    """단일 검증 단계 결과"""
    stage: int
    passed: bool
    reason: str
    total_trades: int = 0
    total_pnl_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    robustness_score: float = 0.0  # Stage 2 전용


@dataclass
class OOSValidationResult:
    """OOS 검증 종합 결과"""
    stages: list[StageResult]
    final_passed: bool
    oos_report: Optional[PerformanceReport] = None
    wf_result: Optional[WalkForwardResult] = None

    def summary(self) -> str:
        lines = ["OOS 프로그레시브 검증 결과:"]
        for s in self.stages:
            status = "PASS" if s.passed else "FAIL"
            lines.append(f"  Stage {s.stage} [{status}]: {s.reason}")
        verdict = "PASS" if self.final_passed else "FAIL"
        lines.append(f"  최종 판정: {verdict}")
        return "\n".join(lines)


class OutOfSampleValidator:
    """3단계 프로그레시브 검증기"""

    def __init__(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        initial_balance: float = 1000.0,
        oos_pct: float = 0.20,
    ):
        self._initial = initial_balance

        # 데이터 분할: 앞 80% = 인샘플, 뒤 20% = OOS
        n = len(df_15m)
        split_idx = int(n * (1 - oos_pct))
        split_time = df_15m.index[split_idx]

        self._is_4h = df_4h[df_4h.index < split_time]
        self._is_1h = df_1h[df_1h.index < split_time]
        self._is_15m = df_15m.iloc[:split_idx]

        self._oos_4h = df_4h[df_4h.index >= split_time]
        self._oos_1h = df_1h[df_1h.index >= split_time]
        self._oos_15m = df_15m.iloc[split_idx:]

        # OOS에도 지표 워밍업 데이터 필요 → 4H/1H는 인샘플 끝부분 포함
        self._oos_4h_full = df_4h
        self._oos_1h_full = df_1h

        logger.info(
            f"OOS 분할: 인샘플 {len(self._is_15m)}개 15M "
            f"({self._is_15m.index[0].date()}~{self._is_15m.index[-1].date()}) | "
            f"OOS {len(self._oos_15m)}개 15M "
            f"({self._oos_15m.index[0].date()}~{self._oos_15m.index[-1].date()})"
        )

    def run(
        self,
        params: AllStrategyParams,
        min_trades_stage1: int = 10,
        robustness_threshold: float = 40.0,
        train_days: int = 45,
        test_days: int = 15,
    ) -> OOSValidationResult:
        """3단계 프로그레시브 검증을 실행한다."""
        stages: list[StageResult] = []

        # ── Stage 1: Discovery (인샘플 전체 백테스트) ──
        logger.info("=== Stage 1: Discovery ===")
        bt = Backtester(params, self._initial)
        is_report = bt.run(self._is_4h, self._is_1h, self._is_15m)

        if is_report.total_trades < min_trades_stage1:
            stage1 = StageResult(
                stage=1, passed=False,
                reason=f"거래 수 부족: {is_report.total_trades} < {min_trades_stage1}",
                total_trades=is_report.total_trades,
                total_pnl_pct=is_report.total_pnl_pct,
                sharpe_ratio=is_report.sharpe_ratio,
            )
            stages.append(stage1)
            return OOSValidationResult(stages=stages, final_passed=False)

        if is_report.total_pnl_pct <= 0:
            stage1 = StageResult(
                stage=1, passed=False,
                reason=f"인샘플 손실: PnL={is_report.total_pnl_pct:.2%}",
                total_trades=is_report.total_trades,
                total_pnl_pct=is_report.total_pnl_pct,
                sharpe_ratio=is_report.sharpe_ratio,
            )
            stages.append(stage1)
            return OOSValidationResult(stages=stages, final_passed=False)

        stage1 = StageResult(
            stage=1, passed=True,
            reason=f"거래 {is_report.total_trades}회, PnL={is_report.total_pnl_pct:.2%}, Sharpe={is_report.sharpe_ratio:.2f}",
            total_trades=is_report.total_trades,
            total_pnl_pct=is_report.total_pnl_pct,
            sharpe_ratio=is_report.sharpe_ratio,
        )
        stages.append(stage1)
        logger.info(f"  Stage 1 PASS: {stage1.reason}")

        # ── Stage 2: Walk-Forward (인샘플 내에서) ──
        logger.info("=== Stage 2: Walk-Forward ===")
        wf = WalkForwardValidator(
            self._is_4h, self._is_1h, self._is_15m,
            initial_balance=self._initial,
            train_days=train_days,
            test_days=test_days,
        )
        wf_result = wf.run(params)

        if wf_result.total_windows == 0:
            stage2 = StageResult(
                stage=2, passed=False,
                reason="워크포워드 윈도우 생성 불가 (데이터 부족)",
                robustness_score=0.0,
            )
            stages.append(stage2)
            return OOSValidationResult(stages=stages, final_passed=False, wf_result=wf_result)

        if wf_result.robustness_score < robustness_threshold:
            stage2 = StageResult(
                stage=2, passed=False,
                reason=f"견고성 부족: {wf_result.robustness_score:.0f} < {robustness_threshold}",
                robustness_score=wf_result.robustness_score,
            )
            stages.append(stage2)
            return OOSValidationResult(stages=stages, final_passed=False, wf_result=wf_result)

        stage2 = StageResult(
            stage=2, passed=True,
            reason=f"견고성={wf_result.robustness_score:.0f}, 윈도우={wf_result.total_windows}, 수익윈도우={wf_result.test_positive_windows}",
            robustness_score=wf_result.robustness_score,
        )
        stages.append(stage2)
        logger.info(f"  Stage 2 PASS: {stage2.reason}")

        # ── Stage 3: OOS 최종 검증 ──
        logger.info("=== Stage 3: Out-of-Sample Test ===")
        bt_oos = Backtester(params, self._initial)
        oos_report = bt_oos.run(self._oos_4h_full, self._oos_1h_full, self._oos_15m)

        reasons = []
        passed = True

        if oos_report.total_trades == 0:
            passed = False
            reasons.append("OOS 거래 없음")
        else:
            if oos_report.sharpe_ratio <= 0:
                passed = False
                reasons.append(f"OOS Sharpe={oos_report.sharpe_ratio:.2f} ≤ 0")
            if oos_report.max_drawdown_pct > 0.25:
                passed = False
                reasons.append(f"OOS DD={oos_report.max_drawdown_pct:.1%} > 25%")
            if oos_report.total_pnl_pct <= 0:
                passed = False
                reasons.append(f"OOS PnL={oos_report.total_pnl_pct:.2%} ≤ 0")

        if passed:
            reason_str = (
                f"OOS 거래={oos_report.total_trades}, "
                f"PnL={oos_report.total_pnl_pct:.2%}, "
                f"Sharpe={oos_report.sharpe_ratio:.2f}, "
                f"DD={oos_report.max_drawdown_pct:.1%}"
            )
        else:
            reason_str = "; ".join(reasons)

        stage3 = StageResult(
            stage=3, passed=passed,
            reason=reason_str,
            total_trades=oos_report.total_trades,
            total_pnl_pct=oos_report.total_pnl_pct,
            sharpe_ratio=oos_report.sharpe_ratio,
            max_drawdown_pct=oos_report.max_drawdown_pct,
        )
        stages.append(stage3)
        logger.info(f"  Stage 3 {'PASS' if passed else 'FAIL'}: {reason_str}")

        return OOSValidationResult(
            stages=stages,
            final_passed=passed,
            oos_report=oos_report,
            wf_result=wf_result,
        )
