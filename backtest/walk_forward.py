"""
워크포워드 검증 (Walk-Forward Validation).

전체 데이터를 학습(Train) + 검증(Test) 구간으로 나누어
학습 구간에서 최적화된 파라미터가 검증 구간에서도 유효한지 확인한다.

사용법:
    from backtest.walk_forward import WalkForwardValidator
    wf = WalkForwardValidator(df_4h, df_1h, df_15m)
    result = wf.run()
"""
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

from backtest.backtester import Backtester
from backtest.performance import PerformanceAnalyzer, PerformanceReport
from config.strategy_params import AllStrategyParams


@dataclass
class WalkForwardWindow:
    """워크포워드 단일 윈도우 결과"""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    train_trades: int
    test_trades: int
    train_sharpe: float
    test_sharpe: float
    train_pnl_pct: float
    test_pnl_pct: float
    test_win_rate: float
    test_profit_factor: float
    test_max_drawdown_pct: float
    degradation_ratio: float  # test_sharpe / train_sharpe


@dataclass
class WalkForwardResult:
    """워크포워드 검증 종합 결과"""
    windows: list[WalkForwardWindow]
    total_windows: int
    avg_train_sharpe: float
    avg_test_sharpe: float
    avg_degradation: float      # 평균 성능 저하율 (1.0 = 동일, <1.0 = 열화)
    test_positive_windows: int  # 검증 구간에서 양의 수익을 낸 윈도우 수
    combined_test_pnl_pct: float
    robustness_score: float     # 0~100점

    def summary(self) -> str:
        return (
            f"워크포워드 검증 ({self.total_windows}개 윈도우)\n"
            f"  학습 평균 Sharpe: {self.avg_train_sharpe:.2f}\n"
            f"  검증 평균 Sharpe: {self.avg_test_sharpe:.2f}\n"
            f"  성능 유지율: {self.avg_degradation:.1%}\n"
            f"  검증 수익 윈도우: {self.test_positive_windows}/{self.total_windows}\n"
            f"  검증 합산 수익률: {self.combined_test_pnl_pct:.2%}\n"
            f"  견고성 점수: {self.robustness_score:.0f}/100"
        )

    def is_robust(self) -> bool:
        """견고성 기준 충족 여부: 점수 50 이상이면 과적합 위험 낮음"""
        return self.robustness_score >= 50.0


class WalkForwardValidator:
    """워크포워드 검증 실행기"""

    def __init__(
        self,
        df_4h: pd.DataFrame,
        df_1h: pd.DataFrame,
        df_15m: pd.DataFrame,
        initial_balance: float = 1000.0,
        train_days: int = 45,
        test_days: int = 15,
        min_trades_per_window: int = 3,
        step_days: int = None,
    ):
        self._df_4h = df_4h
        self._df_1h = df_1h
        self._df_15m = df_15m
        self._initial = initial_balance
        self._train_days = train_days
        self._test_days = test_days
        self._min_trades = min_trades_per_window
        self._step_days = step_days if step_days is not None else test_days

    def run(self, params: Optional[AllStrategyParams] = None) -> WalkForwardResult:
        """워크포워드 검증을 실행한다.

        params가 None이면 yaml에서 기본 파라미터를 로드한다.
        """
        if params is None:
            params = AllStrategyParams.from_yaml()

        windows = self._generate_windows()
        if not windows:
            logger.warning("워크포워드 윈도우를 생성할 수 없습니다. 데이터가 부족합니다.")
            return self._empty_result()

        logger.info(f"워크포워드 검증 시작: {len(windows)}개 윈도우 (학습 {self._train_days}일 / 검증 {self._test_days}일)")

        results: list[WalkForwardWindow] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"  윈도우 {i+1}: 학습 {train_start.date()}~{train_end.date()} / 검증 {test_start.date()}~{test_end.date()}")

            # 학습 구간 백테스트
            train_report = self._run_period(params, train_start, train_end)
            # 검증 구간 백테스트
            test_report = self._run_period(params, test_start, test_end)

            # 성능 저하율 계산
            if train_report.sharpe_ratio > 0 and not np.isinf(train_report.sharpe_ratio):
                degradation = test_report.sharpe_ratio / train_report.sharpe_ratio
            else:
                degradation = 0.0

            wf_window = WalkForwardWindow(
                window_id=i + 1,
                train_start=str(train_start.date()),
                train_end=str(train_end.date()),
                test_start=str(test_start.date()),
                test_end=str(test_end.date()),
                train_trades=train_report.total_trades,
                test_trades=test_report.total_trades,
                train_sharpe=round(train_report.sharpe_ratio, 2),
                test_sharpe=round(test_report.sharpe_ratio, 2),
                train_pnl_pct=round(train_report.total_pnl_pct, 4),
                test_pnl_pct=round(test_report.total_pnl_pct, 4),
                test_win_rate=round(test_report.win_rate, 4),
                test_profit_factor=round(test_report.profit_factor, 2),
                test_max_drawdown_pct=round(test_report.max_drawdown_pct, 4),
                degradation_ratio=round(degradation, 4),
            )
            results.append(wf_window)

            logger.info(
                f"    학습: {train_report.total_trades}거래 Sharpe={train_report.sharpe_ratio:.2f} | "
                f"검증: {test_report.total_trades}거래 Sharpe={test_report.sharpe_ratio:.2f} | "
                f"유지율: {degradation:.1%}"
            )

        return self._aggregate(results)

    def _generate_windows(self) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """롤링 윈도우 생성: 학습 → 검증을 반복"""
        if self._df_15m.empty:
            return []

        data_start = self._df_15m.index[0]
        data_end = self._df_15m.index[-1]
        total_days = (data_end - data_start).days
        window_days = self._train_days + self._test_days

        if total_days < window_days:
            logger.warning(f"데이터 기간({total_days}일)이 윈도우({window_days}일)보다 짧습니다.")
            return []

        windows = []
        current = data_start

        while True:
            train_start = current
            train_end = current + timedelta(days=self._train_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self._test_days)

            if test_end > data_end:
                break

            windows.append((train_start, train_end, test_start, test_end))
            current = current + timedelta(days=self._step_days)  # 슬라이딩 윈도우

        return windows

    def _run_period(
        self,
        params: AllStrategyParams,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> PerformanceReport:
        """특정 기간만 잘라서 백테스트를 실행한다."""
        df_4h = self._df_4h[(self._df_4h.index >= start) & (self._df_4h.index <= end)]
        df_1h = self._df_1h[(self._df_1h.index >= start) & (self._df_1h.index <= end)]
        df_15m = self._df_15m[(self._df_15m.index >= start) & (self._df_15m.index <= end)]

        # 지표 계산에 필요한 룩백 데이터를 포함 (start 이전 데이터)
        # 4H EMA200: 200×4h = 800h ≈ 33.3일 + 여유 → 40일
        lookback_start = start - timedelta(days=40)
        df_4h_full = self._df_4h[(self._df_4h.index >= lookback_start) & (self._df_4h.index <= end)]
        df_1h_full = self._df_1h[(self._df_1h.index >= lookback_start) & (self._df_1h.index <= end)]
        df_15m_full = self._df_15m[(self._df_15m.index >= lookback_start) & (self._df_15m.index <= end)]

        if len(df_4h_full) < 50 or len(df_1h_full) < 30 or len(df_15m_full) < 20:
            from backtest.performance import PerformanceAnalyzer
            return PerformanceAnalyzer([], self._initial).calculate()

        bt = Backtester(params, self._initial)
        return bt.run(df_4h_full, df_1h_full, df_15m_full)

    def _aggregate(self, windows: list[WalkForwardWindow]) -> WalkForwardResult:
        """윈도우 결과를 종합한다."""
        if not windows:
            return self._empty_result()

        train_sharpes = [w.train_sharpe for w in windows]
        test_sharpes = [w.test_sharpe for w in windows]
        test_pnls = [w.test_pnl_pct for w in windows]
        degradations = [w.degradation_ratio for w in windows if w.train_sharpe > 0]

        avg_train = float(np.mean(train_sharpes))
        avg_test = float(np.mean(test_sharpes))
        avg_deg = float(np.mean(degradations)) if degradations else 0.0
        positive = sum(1 for p in test_pnls if p > 0)
        combined_pnl = float(np.sum(test_pnls))

        # 견고성 점수 (0~100)
        # 기준: 검증 수익률, 성능 유지율, 양의 수익 윈도우 비율
        score = self._robustness_score(windows, avg_deg, positive)

        result = WalkForwardResult(
            windows=windows,
            total_windows=len(windows),
            avg_train_sharpe=round(avg_train, 2),
            avg_test_sharpe=round(avg_test, 2),
            avg_degradation=round(avg_deg, 4),
            test_positive_windows=positive,
            combined_test_pnl_pct=round(combined_pnl, 4),
            robustness_score=round(score, 1),
        )

        logger.info(f"\n{result.summary()}")
        return result

    @staticmethod
    def _robustness_score(
        windows: list[WalkForwardWindow],
        avg_degradation: float,
        positive_windows: int,
    ) -> float:
        """견고성 점수 계산 (0~100)

        구성:
        - 검증 수익 윈도우 비율 (40점): 전체 윈도우 중 양의 수익 비율
        - 성능 유지율 (30점): 학습 대비 검증 Sharpe 유지율 (>1.0 캡)
        - 검증 Sharpe 양수 비율 (30점): 검증 Sharpe > 0인 윈도우 비율
        """
        n = len(windows)
        if n == 0:
            return 0.0

        # 1. 검증 수익 비율 (40점)
        pnl_score = (positive_windows / n) * 40

        # 2. 성능 유지율 (30점): 1.0 초과는 캡, 음수는 0
        deg_capped = min(max(avg_degradation, 0.0), 1.0)
        deg_score = deg_capped * 30

        # 3. 검증 Sharpe 양수 비율 (30점)
        sharpe_positive = sum(1 for w in windows if w.test_sharpe > 0)
        sharpe_score = (sharpe_positive / n) * 30

        return pnl_score + deg_score + sharpe_score

    def _empty_result(self) -> WalkForwardResult:
        return WalkForwardResult(
            windows=[],
            total_windows=0,
            avg_train_sharpe=0.0,
            avg_test_sharpe=0.0,
            avg_degradation=0.0,
            test_positive_windows=0,
            combined_test_pnl_pct=0.0,
            robustness_score=0.0,
        )
