"""
기술 지표 함수 단위 테스트.
외부 API 없이 샘플 데이터로 검증한다.
"""
import numpy as np
import pandas as pd
import pytest

from strategy.indicators import (
    calc_atr,
    calc_bollinger,
    calc_ema,
    calc_macd,
    calc_rsi,
    calc_supertrend,
    detect_macd_cross,
)


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """300개 샘플 OHLCV 데이터"""
    np.random.seed(42)
    n = 300
    close = 50000 + np.cumsum(np.random.randn(n) * 200)
    high = close + np.abs(np.random.randn(n) * 100)
    low = close - np.abs(np.random.randn(n) * 100)
    volume = np.abs(np.random.randn(n)) * 1000 + 500

    return pd.DataFrame(
        {"open": close - 50, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
    )


class TestEMA:
    def test_length(self, sample_ohlcv):
        ema = calc_ema(sample_ohlcv["close"], 21)
        assert len(ema) == len(sample_ohlcv)

    def test_no_nan_after_period(self, sample_ohlcv):
        period = 21
        ema = calc_ema(sample_ohlcv["close"], period)
        assert not ema.iloc[period:].isna().any()

    def test_ema200_no_nan(self, sample_ohlcv):
        # ewm(adjust=False) 방식은 첫 값부터 계산되므로 NaN 없음
        ema = calc_ema(sample_ohlcv["close"], 200)
        assert not ema.isna().any()


class TestRSI:
    def test_range(self, sample_ohlcv):
        rsi = calc_rsi(sample_ohlcv["close"], 14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_length(self, sample_ohlcv):
        rsi = calc_rsi(sample_ohlcv["close"])
        assert len(rsi) == len(sample_ohlcv)


class TestMACD:
    def test_columns(self, sample_ohlcv):
        macd = calc_macd(sample_ohlcv["close"])
        assert set(macd.columns) == {"macd", "macd_hist", "macd_signal"}

    def test_length(self, sample_ohlcv):
        macd = calc_macd(sample_ohlcv["close"])
        assert len(macd) == len(sample_ohlcv)

    def test_cross_detection_up(self):
        """히스토그램이 음→양으로 바뀔 때 'UP' 반환"""
        macd_df = pd.DataFrame({"macd": [0, 0], "macd_hist": [-1.0, 0.5], "macd_signal": [0, 0]})
        assert detect_macd_cross(macd_df) == "UP"

    def test_cross_detection_down(self):
        """히스토그램이 양→음으로 바뀔 때 'DOWN' 반환"""
        macd_df = pd.DataFrame({"macd": [0, 0], "macd_hist": [1.0, -0.5], "macd_signal": [0, 0]})
        assert detect_macd_cross(macd_df) == "DOWN"

    def test_cross_detection_none(self):
        """방향 변화 없을 때 'NONE' 반환"""
        macd_df = pd.DataFrame({"macd": [0, 0], "macd_hist": [1.0, 0.5], "macd_signal": [0, 0]})
        assert detect_macd_cross(macd_df) == "NONE"


class TestBollingerBands:
    def test_columns(self, sample_ohlcv):
        bb = calc_bollinger(sample_ohlcv["close"])
        assert {"bb_upper", "bb_mid", "bb_lower"}.issubset(bb.columns)

    def test_upper_above_lower(self, sample_ohlcv):
        bb = calc_bollinger(sample_ohlcv["close"])
        valid = bb.dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_mid_between(self, sample_ohlcv):
        bb = calc_bollinger(sample_ohlcv["close"])
        valid = bb.dropna()
        assert (valid["bb_mid"] <= valid["bb_upper"]).all()
        assert (valid["bb_mid"] >= valid["bb_lower"]).all()


class TestATR:
    def test_positive(self, sample_ohlcv):
        atr = calc_atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        valid = atr.dropna()
        assert (valid > 0).all()

    def test_length(self, sample_ohlcv):
        atr = calc_atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert len(atr) == len(sample_ohlcv)


class TestSupertrend:
    def test_direction_values(self, sample_ohlcv):
        st = calc_supertrend(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        valid_dir = st["supertrend_dir"].dropna()
        assert set(valid_dir.unique()).issubset({1, -1})

    def test_columns(self, sample_ohlcv):
        st = calc_supertrend(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert "supertrend" in st.columns
        assert "supertrend_dir" in st.columns
