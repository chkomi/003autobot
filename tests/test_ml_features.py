"""
ml_features.py 테스트.

가장 중요한 검증: **lookahead 누설 없음**.
임의 row i의 피처 값이 ohlcv[i+1:]를 변경해도 동일해야 함 (현재 bar 포함, 미래 제외).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.ml_features import (
    WARMUP_BARS,
    FeatureConfig,
    build_features,
    feature_columns,
    target_column,
)


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """결정적 1분봉 합성 데이터 (1000 bars)."""
    rng = np.random.default_rng(seed=42)
    n = 1000
    base = 50000.0
    returns = rng.normal(loc=0, scale=0.001, size=n).cumsum()
    close = base * np.exp(returns)
    open_ = np.r_[base, close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0005, n)))
    volume = rng.uniform(10, 200, size=n)

    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def test_feature_columns_present(synthetic_ohlcv):
    feats = build_features(synthetic_ohlcv)
    cols = feature_columns()
    missing = [c for c in cols if c not in feats.columns]
    assert not missing, f"누락 피처: {missing}"


def test_target_present_with_horizons(synthetic_ohlcv):
    feats = build_features(synthetic_ohlcv)
    cfg = FeatureConfig()
    for n in cfg.target_horizons:
        assert f"y_{n}" in feats.columns
    # primary target 컬럼 존재
    assert target_column() in feats.columns


def test_no_lookahead_leakage(synthetic_ohlcv):
    """row i의 피처 값이 i 이후 데이터에 의존하지 않음을 검증.

    전체 데이터로 빌드한 피처와, i+1까지만 자른 데이터로 빌드한 피처가
    row i에서 동일해야 함.
    """
    cfg = FeatureConfig()
    full_feat = build_features(synthetic_ohlcv, cfg)
    cols = feature_columns(cfg)

    # 워밍업 충분한 후, 가장 마지막 ~50 행 사이의 임의 indices 체크
    test_indices = [WARMUP_BARS + 50, WARMUP_BARS + 100, len(synthetic_ohlcv) - 100]
    for i in test_indices:
        truncated = synthetic_ohlcv.iloc[: i + 1]  # 현재 bar 포함, 미래 제외
        trunc_feat = build_features(truncated, cfg)

        for col in cols:
            full_val = full_feat[col].iloc[i]
            trunc_val = trunc_feat[col].iloc[i]
            if pd.isna(full_val) and pd.isna(trunc_val):
                continue
            assert np.isclose(full_val, trunc_val, equal_nan=True), (
                f"누설 의심: row {i} col '{col}' "
                f"full={full_val} truncated={trunc_val}"
            )


def test_target_uses_future(synthetic_ohlcv):
    """타겟 y_N은 명시적으로 future-looking이어야 함."""
    feat_full = build_features(synthetic_ohlcv)
    feat_trunc = build_features(synthetic_ohlcv.iloc[:500])

    # row 100의 y_15는 close[115]에 의존 → truncated[:500]에서는 같지만,
    # row 490의 y_15는 close[505]에 의존 → truncated에서는 NaN 이어야 함
    primary = target_column()
    assert pd.isna(feat_trunc[primary].iloc[490]), \
        f"잘린 데이터의 row 490 y_15는 NaN이어야 함 (got {feat_trunc[primary].iloc[490]})"
    # full에서는 값 있어야 함
    assert not pd.isna(feat_full[primary].iloc[490])


def test_warmup_drop_no_nan_in_features(synthetic_ohlcv):
    """워밍업 이후 row의 피처들에 NaN이 없어야 함 (타겟 제외)."""
    feat = build_features(synthetic_ohlcv)
    feat_post = feat.iloc[WARMUP_BARS:]
    cols = feature_columns()
    nan_counts = feat_post[cols].isna().sum()
    nonzero = nan_counts[nan_counts > 0]
    assert nonzero.empty, f"워밍업 후 NaN 있음: {nonzero.to_dict()}"


def test_returns_are_log(synthetic_ohlcv):
    feat = build_features(synthetic_ohlcv)
    # ret_1m at row i = log(close[i]/close[i-1])
    expected = np.log(synthetic_ohlcv["close"].iloc[100] / synthetic_ohlcv["close"].iloc[99])
    assert np.isclose(feat["ret_1m"].iloc[100], expected)


def test_calendar_features_in_range(synthetic_ohlcv):
    feat = build_features(synthetic_ohlcv)
    assert feat["hour_sin"].between(-1, 1).all()
    assert feat["hour_cos"].between(-1, 1).all()
    assert feat["dow_sin"].between(-1, 1).all()
    assert feat["dow_cos"].between(-1, 1).all()
    assert set(feat["is_weekend"].dropna().unique()).issubset({0, 1})


def test_adx_bucket_categories(synthetic_ohlcv):
    feat = build_features(synthetic_ohlcv).iloc[WARMUP_BARS:]
    assert set(feat["adx_bucket"].dropna().unique()).issubset({0, 1, 2})
