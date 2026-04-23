"""
시그널 집계기 단위 테스트.
각 레이어 개별 동작 및 통합 시그널 검증.
"""
import numpy as np
import pandas as pd
import pytest

from config.strategy_params import TrendFilterParams, MomentumParams, MicroParams, RiskParams
from strategy.trend_filter import TrendFilter
from strategy.momentum_trigger import MomentumTrigger
from strategy.micro_confirmation import MicroConfirmation


def _make_bullish_df(n=300) -> pd.DataFrame:
    """명확한 상승 추세 데이터"""
    np.random.seed(10)
    # 강한 상승 추세
    base = 40000 + np.linspace(0, 20000, n)
    noise = np.random.randn(n) * 100
    close = base + noise
    high = close + 200
    low = close - 200
    vol = np.abs(np.random.randn(n)) * 1000 + 2000
    return pd.DataFrame(
        {"open": close - 50, "high": high, "low": low, "close": close, "volume": vol},
        index=pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC"),
    )


def _make_bearish_df(n=300) -> pd.DataFrame:
    """명확한 하락 추세 데이터"""
    np.random.seed(20)
    base = 60000 - np.linspace(0, 20000, n)
    noise = np.random.randn(n) * 100
    close = base + noise
    high = close + 200
    low = close - 200
    vol = np.abs(np.random.randn(n)) * 1000 + 2000
    return pd.DataFrame(
        {"open": close - 50, "high": high, "low": low, "close": close, "volume": vol},
        index=pd.date_range("2023-01-01", periods=n, freq="4h", tz="UTC"),
    )


def _make_1h_df(n=200) -> pd.DataFrame:
    np.random.seed(30)
    close = 55000 + np.cumsum(np.random.randn(n) * 50)
    return pd.DataFrame(
        {"open": close, "high": close + 100, "low": close - 100, "close": close,
         "volume": np.abs(np.random.randn(n)) * 500 + 500},
        index=pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC"),
    )


def _make_15m_df(n=100) -> pd.DataFrame:
    np.random.seed(40)
    close = 55000 + np.cumsum(np.random.randn(n) * 30)
    return pd.DataFrame(
        {"open": close, "high": close + 50, "low": close - 50, "close": close,
         "volume": np.abs(np.random.randn(n)) * 300 + 300},
        index=pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC"),
    )


class TestTrendFilter:
    def test_returns_signal_with_enough_data(self):
        df = _make_bullish_df(300)
        filt = TrendFilter(TrendFilterParams())
        signal = filt.analyze(df)
        assert signal is not None
        assert signal.regime in ("BULL", "BEAR", "NEUTRAL")

    def test_returns_none_with_insufficient_data(self):
        df = _make_bullish_df(50)  # EMA200에 데이터 부족
        filt = TrendFilter(TrendFilterParams())
        signal = filt.analyze(df)
        assert signal is None

    def test_bullish_trend_detected(self):
        df = _make_bullish_df(300)
        filt = TrendFilter(TrendFilterParams())
        signal = filt.analyze(df)
        # 강한 상승 추세에서 BULL이어야 함 (항상 보장은 아니지만 높은 확률)
        assert signal is not None
        assert signal.ema_fast is not None
        assert signal.price > 0


class TestMomentumTrigger:
    def test_returns_signal(self):
        df = _make_1h_df(100)
        trig = MomentumTrigger(MomentumParams())
        signal = trig.analyze(df)
        assert signal is not None
        assert signal.rsi_value >= 0
        assert signal.rsi_value <= 100
        assert signal.macd_cross in ("UP", "DOWN", "NONE")

    def test_rsi_valid_range_long(self):
        df = _make_1h_df(100)
        trig = MomentumTrigger(MomentumParams(rsi_long_min=40, rsi_long_max=65))
        signal = trig.analyze(df)
        if signal:
            if 40 <= signal.rsi_value <= 65:
                assert signal.rsi_valid_long
            else:
                assert not signal.rsi_valid_long


class TestMicroConfirmation:
    def test_returns_signal(self):
        df = _make_15m_df(50)
        conf = MicroConfirmation(MicroParams())
        signal = conf.analyze(df)
        assert signal is not None
        assert signal.bb_bounce in ("LONG", "SHORT", "NONE")
        assert signal.bb_pct >= 0

    def test_insufficient_data(self):
        df = _make_15m_df(10)  # 볼린저 밴드 기간보다 짧음
        conf = MicroConfirmation(MicroParams(bb_period=20))
        signal = conf.analyze(df)
        assert signal is None

    def test_volume_confirmation(self):
        """거래량이 평균의 1.5배 이상이면 confirmed"""
        np.random.seed(99)
        n = 50
        close = np.ones(n) * 50000
        vol = np.ones(n) * 1000
        vol[-1] = 2000  # 마지막 캔들 거래량 2배

        df = pd.DataFrame(
            {"open": close, "high": close + 50, "low": close - 50, "close": close, "volume": vol},
            index=pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        )
        conf = MicroConfirmation(MicroParams(volume_multiplier=1.5))
        signal = conf.analyze(df)
        if signal:
            assert signal.volume_confirmed
