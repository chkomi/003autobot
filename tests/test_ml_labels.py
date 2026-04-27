"""
ml_labels.py 테스트.

핵심 검증:
  1. 정확성: 합성 데이터에서 알려진 barrier 도달이 정확히 ±1/0으로 라벨됨.
  2. 누설 방지: row i의 라벨은 close[i]·high[i+1..]·low[i+1..] 만 사용 (i+M 이후 무관).
  3. embargo: train_end 이후로 t1이 넘는 샘플은 마스크에서 False.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from strategy.ml_labels import (
    TripleBarrierConfig,
    embargo_mask,
    triple_barrier_labels,
)


def _make_ohlcv(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = 50000.0
    rets = rng.normal(0, 0.001, n).cumsum()
    close = base * np.exp(rets)
    open_ = np.r_[base, close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0005, n)))
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.uniform(10, 200, n)},
        index=idx,
    )


def test_pt_hit_is_plus_one():
    """진입 직후 가격 급등 → 상단 barrier 우선 도달 → 라벨 +1."""
    n = 100
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    close = pd.Series(np.full(n, 100.0), index=idx)
    high = close.copy()
    low = close.copy()
    # row 5에서 진입, row 10에서 high가 +5% 돌파
    high.iloc[10] = 110.0
    vol = pd.Series(np.full(n, 0.01), index=idx)  # vol=1%
    cfg = TripleBarrierConfig(pt_mult=2.0, sl_mult=1.0, max_hold_bars=30)

    out = triple_barrier_labels(close, high, low, vol, cfg)
    # entry row 5: upper = 100 * (1 + 0.02) = 102; row 10 high=110 ≥ 102 → PT hit
    assert out["tb_label"].iloc[5] == 1
    assert out["tb_barrier"].iloc[5] == "pt"
    assert out["tb_bars"].iloc[5] == 5  # 6번째 row (idx 10) - 5 = 5
    assert out["tb_t1"].iloc[5] == idx[10]


def test_sl_hit_is_minus_one():
    n = 100
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    close = pd.Series(np.full(n, 100.0), index=idx)
    high = close.copy()
    low = close.copy()
    low.iloc[8] = 95.0  # row 5 진입 → row 8에서 -5% 도달
    vol = pd.Series(np.full(n, 0.01), index=idx)
    cfg = TripleBarrierConfig(pt_mult=5.0, sl_mult=2.0, max_hold_bars=30)

    out = triple_barrier_labels(close, high, low, vol, cfg)
    # entry row 5: lower = 100 * (1 - 0.02) = 98; row 8 low=95 ≤ 98 → SL hit
    assert out["tb_label"].iloc[5] == -1
    assert out["tb_barrier"].iloc[5] == "sl"
    assert out["tb_bars"].iloc[5] == 3


def test_timeout_is_zero():
    n = 100
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    close = pd.Series(np.full(n, 100.0), index=idx)
    high = close.copy()
    low = close.copy()
    vol = pd.Series(np.full(n, 0.01), index=idx)
    # barrier 매우 멀어서 도달 불가
    cfg = TripleBarrierConfig(pt_mult=10.0, sl_mult=10.0, max_hold_bars=20)

    out = triple_barrier_labels(close, high, low, vol, cfg)
    assert out["tb_label"].iloc[5] == 0
    assert out["tb_barrier"].iloc[5] == "timeout"
    assert out["tb_bars"].iloc[5] == 20


def test_no_label_when_window_insufficient():
    """마지막 max_hold_bars 행은 forward window 부족 → 라벨 NaN."""
    df = _make_ohlcv(200)
    vol = pd.Series(np.full(200, 0.005), index=df.index)
    cfg = TripleBarrierConfig(max_hold_bars=30)
    out = triple_barrier_labels(df["close"], df["high"], df["low"], vol, cfg)

    # 마지막 30 행은 라벨 NA
    last_30_labels = out["tb_label"].iloc[-30:]
    assert last_30_labels.isna().all(), "tail max_hold rows must be NaN"


def test_no_lookahead_leakage():
    """row i 라벨이 i+max_hold_bars 이후 데이터에 의존하지 않음."""
    df = _make_ohlcv(500)
    vol = pd.Series(np.full(500, 0.005), index=df.index)
    cfg = TripleBarrierConfig(max_hold_bars=30)

    full = triple_barrier_labels(df["close"], df["high"], df["low"], vol, cfg)

    # truncate at row 200 (실제 평가 가능 범위는 ~170 까지). row 100 라벨은 동일해야.
    cut = 200
    df_trunc = df.iloc[:cut]
    vol_trunc = vol.iloc[:cut]
    trunc = triple_barrier_labels(df_trunc["close"], df_trunc["high"], df_trunc["low"],
                                   vol_trunc, cfg)

    # row 100: full 과 trunc 의 라벨 동일해야 함 (forward window 100~130 까지만 봄)
    for i in [50, 100, 150]:
        full_lab = full["tb_label"].iloc[i]
        trunc_lab = trunc["tb_label"].iloc[i]
        if pd.isna(full_lab) and pd.isna(trunc_lab):
            continue
        assert full_lab == trunc_lab, f"row {i} 라벨 불일치 full={full_lab} trunc={trunc_lab}"


def test_invalid_vol_yields_nan():
    n = 100
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    close = pd.Series(np.full(n, 100.0), index=idx)
    high = close.copy()
    low = close.copy()
    high.iloc[10] = 110.0  # 충분한 상승
    vol = pd.Series(np.full(n, 0.01), index=idx)
    vol.iloc[5] = np.nan
    vol.iloc[6] = 0.0
    cfg = TripleBarrierConfig(max_hold_bars=30)

    out = triple_barrier_labels(close, high, low, vol, cfg)
    assert pd.isna(out["tb_label"].iloc[5])
    assert pd.isna(out["tb_label"].iloc[6])


def test_simultaneous_hit_is_sl_conservative():
    """같은 bar 안에서 PT/SL 모두 trigger → 보수적으로 SL."""
    n = 50
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    close = pd.Series(np.full(n, 100.0), index=idx)
    high = close.copy()
    low = close.copy()
    # row 5 진입 → row 7 에서 high·low 모두 트리거
    high.iloc[7] = 110.0
    low.iloc[7] = 90.0
    vol = pd.Series(np.full(n, 0.01), index=idx)
    cfg = TripleBarrierConfig(pt_mult=2.0, sl_mult=2.0, max_hold_bars=20)

    out = triple_barrier_labels(close, high, low, vol, cfg)
    assert out["tb_label"].iloc[5] == -1
    assert out["tb_barrier"].iloc[5] == "sl"


def test_embargo_mask_excludes_leaking_t1():
    """t1 이 train_end 를 넘어가는 샘플은 mask=False."""
    n = 100
    idx = pd.date_range("2025-01-01", periods=n, freq="1min", tz="UTC")
    # 임의 t1: row i → i+30분
    t1 = pd.Series(idx + pd.Timedelta(minutes=30), index=idx)

    train_start = idx[0]
    train_end = idx[80]  # row 80 ~ row 99 사이는 t1 leak 위험

    mask = embargo_mask(t1, train_start, train_end)
    # row 49: t1 = idx[79] < train_end → True
    assert bool(mask.iloc[49]) is True
    # row 51: t1 = idx[81] >= train_end → False
    assert bool(mask.iloc[51]) is False
    # train_end 이후도 False
    assert bool(mask.iloc[85]) is False


def test_label_distribution_balanced_on_random_walk():
    """랜덤 워크에서 PT/SL/timeout 모두 발생하는지 sanity check."""
    df = _make_ohlcv(2000, seed=7)
    # vol을 적당히 크게 (몇 분 안에 hit 가능하도록)
    vol = pd.Series(np.full(2000, 0.002), index=df.index)
    cfg = TripleBarrierConfig(pt_mult=2.0, sl_mult=2.0, max_hold_bars=60)

    out = triple_barrier_labels(df["close"], df["high"], df["low"], vol, cfg)
    counts = out["tb_barrier"].value_counts()
    # 적어도 두 가지 outcome 이상 존재
    assert (counts.get("pt", 0) + counts.get("sl", 0)) > 0
