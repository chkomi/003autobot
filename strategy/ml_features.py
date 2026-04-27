"""
ML 백테스트용 1분봉 피처 엔지니어링.

핵심 원칙:
  - 모든 피처는 row i 시점에서 ohlcv[:i+1] (현재 bar 포함, 미래 제외)만 사용.
  - 타겟 y_N은 future-looking이며 학습 시에만 사용. 추론 시 drop.
  - shift(-N) 사용 시 마지막 N 행은 NaN → 학습에서 제외 필요.

재사용 모듈:
  - strategy.indicators: calc_ema, calc_rsi, calc_macd, calc_bollinger,
                         calc_atr, calc_adx, calc_obv, calc_stochastic
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from strategy.indicators import (
    calc_adx,
    calc_atr,
    calc_bollinger,
    calc_ema,
    calc_macd,
    calc_obv,
    calc_rsi,
    calc_stochastic,
)


# 가장 긴 lookback (EMA 200) + 안전 마진. 워밍업이 부족한 row는 drop 대상.
WARMUP_BARS = 240


@dataclass
class FeatureConfig:
    return_horizons: tuple[int, ...] = (1, 5, 15, 60, 240)
    rsi_periods: tuple[int, ...] = (14, 60)
    ema_periods: tuple[int, ...] = (20, 50, 200)
    atr_periods: tuple[int, ...] = (14, 60)
    realized_vol_windows: tuple[int, ...] = (15, 60, 240)
    bb_period: int = 20
    bb_std: float = 2.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stoch_k: int = 14
    stoch_d: int = 3
    adx_period: int = 14
    volume_zscore_windows: tuple[int, ...] = (60, 240)
    target_horizons: tuple[int, ...] = (5, 15, 60)  # 분 단위 forward log return
    primary_target: int = 15  # 학습/백테스트의 메인 타겟
    # Triple Barrier (Phase 0.2). enable=False면 기존 회귀 타겟만 출력.
    triple_barrier_enable: bool = True
    triple_barrier_pt_mult: float = 2.0
    triple_barrier_sl_mult: float = 1.0
    triple_barrier_max_hold: int = 60
    triple_barrier_vol_col: str = "realized_vol_60"  # 변동성 입력 컬럼 (분당 std)


def build_features(df: pd.DataFrame, cfg: FeatureConfig | None = None) -> pd.DataFrame:
    """OHLCV DataFrame → 피처 + 타겟 컬럼 추가된 DataFrame.

    인풋 df 인덱스는 UTC DatetimeIndex이어야 하며 컬럼은 [open, high, low, close, volume].
    """
    cfg = cfg or FeatureConfig()
    out = df.copy()
    close, high, low, vol, op = out["close"], out["high"], out["low"], out["volume"], out["open"]

    # ── Returns ─────────────────────────────────────────────────────
    log_close = np.log(close)
    for n in cfg.return_horizons:
        out[f"ret_{n}m"] = log_close - log_close.shift(n)

    # ── Momentum ───────────────────────────────────────────────────
    for p in cfg.rsi_periods:
        out[f"rsi_{p}"] = calc_rsi(close, p)

    macd_df = calc_macd(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    out["macd"] = macd_df["macd"]
    out["macd_signal"] = macd_df["macd_signal"]
    out["macd_hist"] = macd_df["macd_hist"]

    stoch_df = calc_stochastic(high, low, close, cfg.stoch_k, cfg.stoch_d)
    out["stoch_k"] = stoch_df["stoch_k"]
    out["stoch_d"] = stoch_df["stoch_d"]

    # ── Trend / EMA ─────────────────────────────────────────────────
    for p in cfg.ema_periods:
        ema = calc_ema(close, p)
        out[f"ema_{p}"] = ema
        out[f"ema_{p}_dist"] = (close - ema) / close.replace(0, np.nan)

    # EMA 크로스 시그널 (단/중기, 중/장기) — 부호만
    if {20, 50, 200}.issubset(cfg.ema_periods):
        out["ema_20_50_diff"] = (out["ema_20"] - out["ema_50"]) / close.replace(0, np.nan)
        out["ema_50_200_diff"] = (out["ema_50"] - out["ema_200"]) / close.replace(0, np.nan)

    # ── Volatility ──────────────────────────────────────────────────
    ret_1 = out[f"ret_1m"] if "ret_1m" in out.columns else log_close.diff()
    for p in cfg.atr_periods:
        atr = calc_atr(high, low, close, p)
        out[f"atr_{p}"] = atr
        out[f"atr_{p}_pct"] = atr / close.replace(0, np.nan)

    for w in cfg.realized_vol_windows:
        out[f"realized_vol_{w}"] = ret_1.rolling(window=w, min_periods=w).std()

    bb = calc_bollinger(close, cfg.bb_period, cfg.bb_std)
    out["bb_bandwidth"] = bb["bb_bandwidth"]
    out["bb_pct"] = bb["bb_pct"]

    # Parkinson volatility (high-low range based, sqrt(252*1440)? 단순 비율로):
    log_hl = np.log(high / low.replace(0, np.nan))
    out["parkinson_60"] = (log_hl ** 2).rolling(window=60, min_periods=60).mean().pipe(np.sqrt)

    # ── Volume ─────────────────────────────────────────────────────
    for w in cfg.volume_zscore_windows:
        mean = vol.rolling(window=w, min_periods=w).mean()
        std = vol.rolling(window=w, min_periods=w).std().replace(0, np.nan)
        out[f"vol_zscore_{w}"] = (vol - mean) / std

    out["dollar_volume"] = vol * close
    obv = calc_obv(close, vol)
    out["obv_slope_60"] = (obv - obv.shift(60)) / 60.0

    # ── Microstructure ─────────────────────────────────────────────
    out["hl_range_pct"] = (high - low) / close.replace(0, np.nan)
    bar_range = (high - low).replace(0, np.nan)
    out["body_pct_of_range"] = (close - op) / bar_range
    out["body_sign"] = np.sign(close - op)
    out["body_sum_5"] = (close - op).rolling(window=5, min_periods=5).sum()

    # ── Regime ─────────────────────────────────────────────────────
    adx_df = calc_adx(high, low, close, cfg.adx_period)
    out["adx"] = adx_df["adx"]
    out["plus_di"] = adx_df["plus_di"]
    out["minus_di"] = adx_df["minus_di"]
    # bucket: 0=ranging(<15), 1=neutral(15-25), 2=trending(>=25)
    out["adx_bucket"] = pd.cut(
        out["adx"], bins=[-np.inf, 15.0, 25.0, np.inf], labels=[0, 1, 2]
    ).astype("Int8")

    # ── Calendar ───────────────────────────────────────────────────
    idx = out.index
    if isinstance(idx, pd.DatetimeIndex):
        hour = idx.hour
        dow = idx.dayofweek
        out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        out["is_weekend"] = ((dow >= 5).astype("int8"))
        out["is_us_session"] = (((hour >= 13) & (hour < 21)).astype("int8"))   # ~9am-5pm EST UTC
        out["is_asia_session"] = (((hour >= 0) & (hour < 8)).astype("int8"))

    # ── Targets (forward looking) ───────────────────────────────────
    for n in cfg.target_horizons:
        out[f"y_{n}"] = np.log(close.shift(-n) / close.replace(0, np.nan))

    # ── Triple Barrier 라벨 (Phase 0.2) ────────────────────────────
    if cfg.triple_barrier_enable:
        from strategy.ml_labels import TripleBarrierConfig, triple_barrier_labels
        vol_col = cfg.triple_barrier_vol_col
        if vol_col not in out.columns:
            raise KeyError(f"triple_barrier_vol_col={vol_col} 가 features 에 없음")
        tb_cfg = TripleBarrierConfig(
            pt_mult=cfg.triple_barrier_pt_mult,
            sl_mult=cfg.triple_barrier_sl_mult,
            max_hold_bars=cfg.triple_barrier_max_hold,
        )
        tb = triple_barrier_labels(
            close=out["close"],
            high=out["high"],
            low=out["low"],
            vol=out[vol_col],
            cfg=tb_cfg,
        )
        out["tb_label"] = tb["tb_label"]
        out["tb_ret"] = tb["tb_ret"]
        out["tb_bars"] = tb["tb_bars"]
        out["tb_t1"] = tb["tb_t1"]
        out["tb_barrier"] = tb["tb_barrier"]

    return out


def feature_columns(cfg: FeatureConfig | None = None) -> list[str]:
    """build_features 결과에서 학습/추론 입력으로 쓸 컬럼 리스트 (타겟·OHLCV 제외)."""
    cfg = cfg or FeatureConfig()
    cols: list[str] = []
    for n in cfg.return_horizons:
        cols.append(f"ret_{n}m")
    for p in cfg.rsi_periods:
        cols.append(f"rsi_{p}")
    cols += ["macd", "macd_signal", "macd_hist", "stoch_k", "stoch_d"]
    for p in cfg.ema_periods:
        cols += [f"ema_{p}_dist"]
    if {20, 50, 200}.issubset(cfg.ema_periods):
        cols += ["ema_20_50_diff", "ema_50_200_diff"]
    for p in cfg.atr_periods:
        cols += [f"atr_{p}_pct"]
    for w in cfg.realized_vol_windows:
        cols.append(f"realized_vol_{w}")
    cols += ["bb_bandwidth", "bb_pct", "parkinson_60"]
    for w in cfg.volume_zscore_windows:
        cols.append(f"vol_zscore_{w}")
    cols += ["obv_slope_60"]  # dollar_volume 제외 (스케일 큼)
    cols += [
        "hl_range_pct", "body_pct_of_range", "body_sign", "body_sum_5",
        "adx", "plus_di", "minus_di", "adx_bucket",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "is_weekend", "is_us_session", "is_asia_session",
    ]
    return cols


def target_column(cfg: FeatureConfig | None = None) -> str:
    cfg = cfg or FeatureConfig()
    return f"y_{cfg.primary_target}"


def build_features_for_dir(
    raw_dir: Path,
    out_dir: Path,
    cfg: FeatureConfig | None = None,
) -> dict[str, int]:
    """월별 raw OHLCV Parquet → 월별 feature Parquet.

    각 월 빌드 시 직전 WARMUP_BARS 행을 끌어와 워밍업한 뒤 해당 월만 출력.
    빌드 후 row 수를 dict로 반환.
    """
    cfg = cfg or FeatureConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(raw_dir.glob("*.parquet"))
    if not raw_files:
        raise FileNotFoundError(f"{raw_dir} 비어있음")

    # 모든 월을 단일 DataFrame으로 로드 후 한 번에 빌드 → 가장 정확한 결과.
    # 데이터셋이 ~3M 행이면 메모리 OK (~200MB).
    logger.info(f"raw {len(raw_files)}개 월 로딩 중...")
    df = pd.concat([pd.read_parquet(p) for p in raw_files]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    logger.info(f"raw rows: {len(df):,}")

    feat = build_features(df, cfg)
    # warmup 이전 행 drop (NaN 제거)
    feat = feat.iloc[WARMUP_BARS:]
    logger.info(f"feature rows (warmup drop): {len(feat):,}, cols: {feat.shape[1]}")

    # 월별 분할 저장
    written: dict[str, int] = {}
    months = feat.index.to_period("M").unique()
    for month in tqdm(months, desc="월별 저장"):
        mask = feat.index.to_period("M") == month
        sub = feat.loc[mask]
        if sub.empty:
            continue
        path = out_dir / f"{month}.parquet"
        sub.to_parquet(path, compression="zstd", engine="pyarrow")
        written[str(month)] = len(sub)

    return written


def load_feature_range(
    feature_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    start = pd.Timestamp(start, tz="UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = pd.Timestamp(end, tz="UTC") if end.tzinfo is None else end.tz_convert("UTC")

    files = sorted(feature_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(feature_dir)

    frames = []
    for p in files:
        # filename: YYYY-MM.parquet
        try:
            month = pd.Period(p.stem, freq="M")
        except Exception:
            continue
        ms = month.to_timestamp(tz="UTC")
        me = (month + 1).to_timestamp(tz="UTC")
        if me <= start or ms >= end:
            continue
        frames.append(pd.read_parquet(p))

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.loc[(df.index >= start) & (df.index < end)]


def _cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        default="data_cache/ohlcv_1m/BTC-USDT-SWAP",
        help="월별 raw Parquet 디렉토리",
    )
    parser.add_argument(
        "--out-dir",
        default="features/btc_1m_v1",
        help="피처 출력 디렉토리",
    )
    args = parser.parse_args()
    written = build_features_for_dir(Path(args.raw_dir), Path(args.out_dir))
    logger.info(f"완료. 월별 row 수: {written}")
    logger.info(f"총 row 수: {sum(written.values()):,}")


if __name__ == "__main__":
    _cli()
