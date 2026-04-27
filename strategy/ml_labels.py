"""
Triple Barrier Labeling (López de Prado, AFML 2018, Ch.3).

각 진입 시점 i에 대해 세 가지 barrier 중 어느 것이 먼저 닿는지로 라벨링한다.
  - 상단 barrier (profit-take): entry * (1 + pt_mult * vol_i)
  - 하단 barrier (stop-loss):   entry * (1 - sl_mult * vol_i)
  - 수직 barrier (max hold):    i + max_hold_bars

라벨 정의:
  +1 : 상단 barrier 먼저 도달 (long 성공)
  -1 : 하단 barrier 먼저 도달 (long 실패)
   0 : 시간 만료 (timeout)

설계 원칙:
  - 모든 barrier 검사는 row i+1 ~ i+max_hold_bars 사이의 high/low 만 참조
    → 진입 bar 자체에서는 trigger 되지 않음 (현실적 진입 가정).
  - 변동성은 호출 시점에 결정된 vol_i (예: realized_vol_60 또는 ATR)를 사용.
    학습 시점 leakage 방지를 위해 i 시점까지 사용 가능한 값만 vol_i로 넘겨야 함.
  - 라벨은 future-looking이므로 학습용 타겟. 추론 시 drop.
  - embargo: barrier 도달 시각 t1[i]가 학습-테스트 경계를 넘으면 그 샘플은 학습에서
    제외해야 한다. 본 함수는 t1 컬럼(절대 timestamp)을 반환만 하고, embargo 적용은
    호출 측 walk-forward 로직 책임.

성능: 3M 행 / max_hold=60 기준 numpy 벡터화로 ~10초 내 완료.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class TripleBarrierConfig:
    pt_mult: float = 2.0           # 상단 barrier = entry * (1 + pt_mult * vol)
    sl_mult: float = 1.0           # 하단 barrier = entry * (1 - sl_mult * vol)
    max_hold_bars: int = 60        # 수직 barrier (분 단위, 1m bar 기준)
    min_vol: float = 1e-6          # vol < min_vol 인 row는 라벨 NaN
    label_neutral_zero_ret: bool = True  # timeout 시 0 라벨 부여 (False면 ret 부호로 ±1)


def triple_barrier_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    vol: pd.Series,
    cfg: TripleBarrierConfig | None = None,
) -> pd.DataFrame:
    """Triple Barrier 라벨 계산.

    Parameters
    ----------
    close, high, low : pd.Series
        UTC DatetimeIndex 정렬된 1m OHLCV.
    vol : pd.Series
        row i 시점 진입 시 사용할 변동성. 보통 realized_vol_60 (분당 std)나
        atr_14_pct (분당 ATR/close 비율). 본 함수는 비율로 가정.
    cfg : TripleBarrierConfig

    Returns
    -------
    DataFrame with columns:
      tb_label  : Int8 ∈ {-1, 0, 1, NaN}
      tb_ret    : float, log return at barrier touch
      tb_bars   : Int32, barrier 도달까지 bar 수 (1 = 다음 bar에서 hit)
      tb_t1     : datetime64[ns, UTC], barrier 도달 timestamp
      tb_barrier: object, 'pt' | 'sl' | 'timeout' | 'na'
    """
    cfg = cfg or TripleBarrierConfig()
    n = len(close)
    if n == 0:
        return _empty_result(close.index)

    if not (len(high) == n and len(low) == n and len(vol) == n):
        raise ValueError("close/high/low/vol 길이 불일치")

    close_v = close.to_numpy(dtype=np.float64)
    high_v = high.to_numpy(dtype=np.float64)
    low_v = low.to_numpy(dtype=np.float64)
    vol_v = vol.to_numpy(dtype=np.float64)

    M = cfg.max_hold_bars

    # 결과 버퍼
    label = np.full(n, np.iinfo(np.int8).min, dtype=np.int8)  # sentinel = "missing"
    label_valid = np.zeros(n, dtype=bool)
    ret_arr = np.full(n, np.nan, dtype=np.float64)
    bars_arr = np.zeros(n, dtype=np.int32)
    barrier_arr = np.full(n, "na", dtype=object)

    # 평가 가능한 event 범위: i ∈ [0, n - M - 1]
    # 마지막 M 개 row는 forward window가 부족 → 라벨 NaN.
    n_events = n - M
    if n_events <= 0:
        return _build_dataframe(close.index, label, label_valid, ret_arr, bars_arr, barrier_arr)

    # forward windows: high[i+1 .. i+M], shape (n_events, M)
    # sliding_window_view(arr, M) 는 arr[k : k+M] 형태 → arr=high[1:], 시작 index k=i 가
    # high[i+1 : i+1+M] 와 동일.
    fwd_high = sliding_window_view(high_v[1:], window_shape=M)[:n_events]
    fwd_low = sliding_window_view(low_v[1:], window_shape=M)[:n_events]
    fwd_close = sliding_window_view(close_v[1:], window_shape=M)[:n_events]

    entry = close_v[:n_events]
    vol_e = vol_v[:n_events]

    valid_vol = np.isfinite(vol_e) & (vol_e > cfg.min_vol)

    upper = np.where(valid_vol, entry * (1.0 + cfg.pt_mult * vol_e), np.nan)
    lower = np.where(valid_vol, entry * (1.0 - cfg.sl_mult * vol_e), np.nan)

    # barrier 도달 mask: 비교 대상이 NaN이면 False
    pt_mask = fwd_high >= upper[:, None]
    sl_mask = fwd_low <= lower[:, None]

    pt_any = pt_mask.any(axis=1)
    sl_any = sl_mask.any(axis=1)

    pt_idx = pt_mask.argmax(axis=1)  # all-False이면 0이지만 pt_any로 가려냄
    sl_idx = sl_mask.argmax(axis=1)

    # 무한대 sentinel (M = "도달 안 함")
    pt_idx_eff = np.where(pt_any, pt_idx, M)
    sl_idx_eff = np.where(sl_any, sl_idx, M)

    pt_first = pt_idx_eff < sl_idx_eff
    sl_first = sl_idx_eff < pt_idx_eff
    timeout = (~pt_any) & (~sl_any)
    # 동시 도달 (pt_idx == sl_idx): bar 안에서 high·low 모두 trigger → 보수적으로 SL 우선
    both_hit = (pt_idx_eff == sl_idx_eff) & pt_any & sl_any

    # 결과 채우기 (event 범위만)
    sub_label = np.zeros(n_events, dtype=np.int8)
    sub_valid = valid_vol.copy()
    sub_ret = np.full(n_events, np.nan, dtype=np.float64)
    sub_bars = np.zeros(n_events, dtype=np.int32)
    sub_barrier = np.full(n_events, "na", dtype=object)

    # PT 먼저
    idx_pt = pt_first & valid_vol & ~both_hit
    sub_label[idx_pt] = 1
    sub_bars[idx_pt] = pt_idx_eff[idx_pt] + 1
    sub_ret[idx_pt] = np.log(upper[idx_pt] / entry[idx_pt])
    sub_barrier[idx_pt] = "pt"

    # SL 먼저 (또는 동시 hit → SL 보수)
    idx_sl = (sl_first | both_hit) & valid_vol
    sub_label[idx_sl] = -1
    sub_bars[idx_sl] = sl_idx_eff[idx_sl] + 1
    sub_ret[idx_sl] = np.log(lower[idx_sl] / entry[idx_sl])
    sub_barrier[idx_sl] = "sl"

    # Timeout
    idx_to = timeout & valid_vol
    last_close = fwd_close[:, -1]
    sub_bars[idx_to] = M
    sub_ret[idx_to] = np.log(last_close[idx_to] / entry[idx_to])
    sub_barrier[idx_to] = "timeout"
    if cfg.label_neutral_zero_ret:
        sub_label[idx_to] = 0
    else:
        sub_label[idx_to] = np.sign(sub_ret[idx_to]).astype(np.int8)

    # invalid vol → label NaN sentinel
    sub_valid &= (idx_pt | idx_sl | idx_to)

    # 전체 배열에 반영
    label[:n_events] = sub_label
    label_valid[:n_events] = sub_valid
    ret_arr[:n_events] = sub_ret
    bars_arr[:n_events] = sub_bars
    barrier_arr[:n_events] = sub_barrier

    return _build_dataframe(close.index, label, label_valid, ret_arr, bars_arr, barrier_arr)


def _build_dataframe(
    index: pd.Index,
    label: np.ndarray,
    label_valid: np.ndarray,
    ret_arr: np.ndarray,
    bars_arr: np.ndarray,
    barrier_arr: np.ndarray,
) -> pd.DataFrame:
    label_series = pd.Series(label.astype(np.float32), index=index)
    label_series[~label_valid] = np.nan
    label_series = label_series.astype("Int8")

    if isinstance(index, pd.DatetimeIndex):
        bars = bars_arr.astype("int64")
        offset = pd.to_timedelta(bars, unit="m")
        t1 = index + offset
        t1 = pd.Series(t1, index=index)
        t1[~label_valid] = pd.NaT
    else:
        t1 = pd.Series([pd.NaT] * len(index), index=index)

    return pd.DataFrame(
        {
            "tb_label": label_series,
            "tb_ret": pd.Series(ret_arr, index=index),
            "tb_bars": pd.Series(bars_arr, index=index).astype("Int32"),
            "tb_t1": t1,
            "tb_barrier": pd.Series(barrier_arr, index=index),
        }
    )


def _empty_result(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tb_label": pd.Series(dtype="Int8"),
            "tb_ret": pd.Series(dtype="float64"),
            "tb_bars": pd.Series(dtype="Int32"),
            "tb_t1": pd.Series(dtype="datetime64[ns, UTC]"),
            "tb_barrier": pd.Series(dtype="object"),
        },
        index=index,
    )


def embargo_mask(
    t1: pd.Series,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> pd.Series:
    """학습용 mask: row i 의 barrier touch 시각 t1[i] 가 train_end를 넘기면 제외.

    walk-forward에서 학습 구간 끝에 닿은 라벨이 미래 정보로 누설되는 것을 방지.
    embargo 길이는 자동으로 max(tb_bars) ≈ max_hold_bars 분.
    """
    train_start = pd.Timestamp(train_start, tz="UTC") if train_start.tzinfo is None else train_start.tz_convert("UTC")
    train_end = pd.Timestamp(train_end, tz="UTC") if train_end.tzinfo is None else train_end.tz_convert("UTC")

    valid = t1.notna()
    in_range = (t1.index >= train_start) & (t1.index < train_end) & valid
    no_leak = t1 < train_end
    return pd.Series(in_range & no_leak, index=t1.index)
