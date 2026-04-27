"""
1분봉 ML 추론 기반 백테스터 (BTC 전용 v1).

설계 개요:
  - 입력: 월별 raw OHLCV Parquet + 월별 feature Parquet + 월별 LGB 모델
  - 매 bar i (close 기준)에서 해당 월의 모델로 ŷ 예측
  - 임계값 (fold metadata의 threshold_long / threshold_short) 통과 시 진입
    체결 가격은 i+1 bar의 open (latency model)
  - exit: 모델 시그널 flip OR 60-bar (1시간) timeout
  - fee 0.05% / slippage 0.05%/side / funding 0.01% per 8h
  - 포지션 사이징: balance × position_pct × leverage
  - notional 기반 PnL 산출

기존 backtest/backtester.py(15분 룰베이스)는 그대로 두고 별도 파일로 추가.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from tqdm.auto import tqdm

from strategy.ml_features import FeatureConfig, feature_columns


@dataclass
class MLBacktestConfig:
    initial_balance: float = 1000.0
    leverage: int = 5
    position_pct: float = 0.10              # 잔고의 10%를 노출량으로
    fee_pct: float = 0.0005                 # OKX taker 0.05%
    slippage_pct: float = 0.0005            # 0.05%/side
    funding_per_8h_pct: float = 0.0001      # 0.01% per 8h (constant assumption)
    max_hold_bars: int = 60                 # 1시간 timeout
    feature_cfg: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass
class Trade:
    fold_id: str
    direction: int       # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    quantity: float = 0.0
    notional: float = 0.0
    fee_usdt: float = 0.0
    funding_usdt: float = 0.0
    pnl_usdt: Optional[float] = None
    return_pct: Optional[float] = None


def _funding_charge(
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    direction: int,
    notional: float,
    rate_per_8h: float,
) -> float:
    """포지션 보유 중 funding 발생 시각(00, 08, 16 UTC)마다 비용 청구.

    부호 가정: long은 funding rate가 양수일 때 비용, 음수일 때 수령.
    상수 모델이므로 항상 비용 발생 (보수적). long/short 모두 동일 부호로 차감.
    v2에서 실제 funding history와 부호 모델 적용.
    """
    if exit_time <= entry_time:
        return 0.0
    # funding instants: every 8h at 00:00, 08:00, 16:00 UTC
    e0 = entry_time.tz_convert("UTC") if entry_time.tzinfo else entry_time.tz_localize("UTC")
    x0 = exit_time.tz_convert("UTC") if exit_time.tzinfo else exit_time.tz_localize("UTC")
    # find first funding instant after entry
    h0 = e0.floor("8h")
    if h0 <= e0:
        h0 = h0 + pd.Timedelta(hours=8)
    n_charges = 0
    cur = h0
    while cur < x0:
        n_charges += 1
        cur = cur + pd.Timedelta(hours=8)
    return n_charges * rate_per_8h * notional


def _pick_fold_for(ts: pd.Timestamp, fold_ids_sorted: list[str]) -> Optional[str]:
    """test_start ≤ ts < test_end 인 fold를 찾음.

    fold_ids_sorted: ['2022-07', '2022-08', ...] (test month 기준).
    각 fold는 자신의 test month 동안만 유효. test month 이전이면 fold 미존재(skip).
    """
    if not fold_ids_sorted:
        return None
    target = f"{ts.year:04d}-{ts.month:02d}"
    if target in fold_ids_sorted:
        return target
    return None  # train-only 구간이면 거래 안 함


def run_ml_backtest(
    raw_dir: Path,
    feature_dir: Path,
    model_dir: Path,
    cfg: Optional[MLBacktestConfig] = None,
) -> dict:
    cfg = cfg or MLBacktestConfig()

    # ── 모델 + 메타 로드 ─────────────────────────────────────────
    meta = json.loads((model_dir / "metadata.json").read_text())
    fold_meta_by_id = {f["fold_id"]: f for f in meta["folds"]}
    fold_ids = sorted(fold_meta_by_id.keys())
    boosters: dict[str, lgb.Booster] = {
        f["fold_id"]: lgb.Booster(model_file=str(model_dir / Path(f["model_path"]).name))
        for f in meta["folds"]
    }
    feat_cols: list[str] = meta["feature_columns"]
    logger.info(f"{len(fold_ids)}개 fold 모델 로드. 첫 fold={fold_ids[0]}, 마지막={fold_ids[-1]}")

    # ── raw + feature 데이터 로드 ───────────────────────────────
    raw_files = sorted(raw_dir.glob("*.parquet"))
    feat_files = sorted(feature_dir.glob("*.parquet"))
    if not raw_files or not feat_files:
        raise FileNotFoundError("raw or feature dir 비어있음")
    raw_df = pd.concat([pd.read_parquet(p) for p in raw_files]).sort_index()
    raw_df = raw_df[~raw_df.index.duplicated(keep="last")]
    feat_df = pd.concat([pd.read_parquet(p) for p in feat_files]).sort_index()
    feat_df = feat_df[~feat_df.index.duplicated(keep="last")]

    # 첫 test fold의 시작 이후만 거래 대상
    first_test_start = pd.Timestamp(meta["folds"][0]["test_start"])
    last_test_end = pd.Timestamp(meta["folds"][-1]["test_end"])
    raw_df = raw_df.loc[(raw_df.index >= first_test_start) & (raw_df.index <= last_test_end)]
    feat_df = feat_df.loc[(feat_df.index >= first_test_start) & (feat_df.index <= last_test_end)]
    common_idx = raw_df.index.intersection(feat_df.index)
    raw_df = raw_df.loc[common_idx]
    feat_df = feat_df.loc[common_idx]
    logger.info(f"백테스트 구간 rows: {len(raw_df):,}")

    # 피처 행렬 (numpy)
    X_full = feat_df[feat_cols].astype("float32").values
    open_arr = raw_df["open"].values
    close_arr = raw_df["close"].values
    times = raw_df.index

    # 사전 예측: 각 fold 구간만 각 모델로 inference
    y_hat_full = np.full(len(raw_df), np.nan, dtype="float32")
    for fold in meta["folds"]:
        fid = fold["fold_id"]
        ts_start = pd.Timestamp(fold["test_start"])
        ts_end = pd.Timestamp(fold["test_end"])
        mask = (times >= ts_start) & (times < ts_end)
        if mask.any():
            y_hat_full[mask] = boosters[fid].predict(X_full[mask])
    logger.info("예측 완료.")

    # ── event loop ───────────────────────────────────────────────
    balance = cfg.initial_balance
    open_trade: Optional[Trade] = None
    trades: list[Trade] = []
    equity_records: list[tuple[pd.Timestamp, float]] = []

    n = len(raw_df)
    pbar = tqdm(total=n, desc="ML 백테스트", dynamic_ncols=True)
    try:
        for i in range(n - 1):  # 마지막 bar는 다음 open 없으므로 진입 불가
            t = times[i]
            y_hat = y_hat_full[i]
            equity_records.append((t, balance + (_unrealized_pnl(open_trade, close_arr[i]) if open_trade else 0.0)))

            # 1) 포지션 관리
            if open_trade is not None:
                hold_bars = i - open_trade.entry_idx  # type: ignore[attr-defined]
                exit_reason: Optional[str] = None
                # signal flip
                if not np.isnan(y_hat):
                    fid = _pick_fold_for(t, fold_ids)
                    if fid is not None:
                        thr_long = fold_meta_by_id[fid]["threshold_long"]
                        thr_short = fold_meta_by_id[fid]["threshold_short"]
                        if open_trade.direction == 1 and y_hat < thr_short:
                            exit_reason = "model_flip"
                        elif open_trade.direction == -1 and y_hat > thr_long:
                            exit_reason = "model_flip"
                # timeout
                if exit_reason is None and hold_bars >= cfg.max_hold_bars:
                    exit_reason = "timeout"

                if exit_reason is not None:
                    # 체결가: 다음 bar open
                    exit_idx = i + 1
                    fill_price = open_arr[exit_idx] * (1 - cfg.slippage_pct * open_trade.direction)
                    realized = _close_trade(open_trade, fill_price, times[exit_idx], cfg, exit_reason)
                    balance += realized
                    trades.append(open_trade)
                    open_trade = None

            # 2) 신규 진입 (포지션 없을 때 + fold 유효 + ŷ 정상)
            if open_trade is None and not np.isnan(y_hat):
                fid = _pick_fold_for(t, fold_ids)
                if fid is not None:
                    thr_long = fold_meta_by_id[fid]["threshold_long"]
                    thr_short = fold_meta_by_id[fid]["threshold_short"]
                    direction = 0
                    if y_hat > thr_long:
                        direction = 1
                    elif y_hat < thr_short:
                        direction = -1
                    if direction != 0:
                        entry_idx = i + 1
                        if entry_idx >= n:
                            continue
                        raw_entry = open_arr[entry_idx]
                        fill_price = raw_entry * (1 + cfg.slippage_pct * direction)
                        notional = balance * cfg.position_pct * cfg.leverage
                        qty = notional / fill_price
                        fee_in = notional * cfg.fee_pct
                        balance -= fee_in
                        open_trade = Trade(
                            fold_id=fid,
                            direction=direction,
                            entry_time=times[entry_idx],
                            entry_price=fill_price,
                            quantity=qty,
                            notional=notional,
                            fee_usdt=fee_in,
                        )
                        # entry_idx attr (for hold_bars)
                        open_trade.entry_idx = entry_idx  # type: ignore[attr-defined]
            pbar.update(1)
    finally:
        pbar.close()

    # 강제 청산
    if open_trade is not None:
        last_price = close_arr[-1]
        realized = _close_trade(open_trade, last_price, times[-1], cfg, "force_close")
        balance += realized
        trades.append(open_trade)

    equity_records.append((times[-1], balance))
    eq_df = pd.DataFrame(equity_records, columns=["time", "equity"]).set_index("time")

    trades_records = [_trade_to_dict(t) for t in trades]
    trades_df = pd.DataFrame(trades_records)

    summary = _summary(eq_df, trades_df, cfg.initial_balance)
    logger.info(
        f"백테스트 완료. trades={len(trades)} "
        f"final={balance:.2f} "
        f"return={(balance/cfg.initial_balance-1)*100:.2f}% "
        f"mean_rank_ic={meta['summary']['mean_rank_ic']:.4f}"
    )
    return {
        "summary": summary,
        "equity_curve": eq_df,
        "trades": trades_df,
        "model_metadata": meta,
        "model_dir": str(model_dir),
        "config": {
            "initial_balance": cfg.initial_balance,
            "leverage": cfg.leverage,
            "position_pct": cfg.position_pct,
            "fee_pct": cfg.fee_pct,
            "slippage_pct": cfg.slippage_pct,
            "funding_per_8h_pct": cfg.funding_per_8h_pct,
            "max_hold_bars": cfg.max_hold_bars,
        },
    }


def _unrealized_pnl(trade: Trade, mark_price: float) -> float:
    if trade is None or trade.quantity == 0:
        return 0.0
    return (mark_price - trade.entry_price) * trade.quantity * trade.direction


def _close_trade(
    trade: Trade, fill_price: float, exit_time: pd.Timestamp,
    cfg: MLBacktestConfig, reason: str,
) -> float:
    """trade를 mutate하여 청산 정보 기록. 잔고 변동분 반환."""
    trade.exit_price = fill_price
    trade.exit_time = exit_time
    trade.exit_reason = reason
    fee_out = abs(trade.quantity * fill_price) * cfg.fee_pct
    trade.fee_usdt += fee_out
    funding = _funding_charge(
        trade.entry_time, exit_time, trade.direction, trade.notional, cfg.funding_per_8h_pct
    )
    trade.funding_usdt = funding
    gross_pnl = (fill_price - trade.entry_price) * trade.quantity * trade.direction
    # balance 변동: gross_pnl - fee_out - funding (entry fee는 진입 시점에 이미 차감됨)
    realized = gross_pnl - fee_out - funding
    # trade.pnl_usdt는 사용자가 보는 trade-level net P&L (entry+exit fee 모두 포함)
    trade.pnl_usdt = gross_pnl - trade.fee_usdt - trade.funding_usdt
    trade.return_pct = realized / trade.notional if trade.notional else 0.0
    return realized


def _trade_to_dict(t: Trade) -> dict:
    return {
        "fold_id": t.fold_id,
        "direction": t.direction,
        "entry_time": t.entry_time,
        "exit_time": t.exit_time,
        "entry_price": t.entry_price,
        "exit_price": t.exit_price,
        "exit_reason": t.exit_reason,
        "quantity": t.quantity,
        "notional": t.notional,
        "fee_usdt": t.fee_usdt,
        "funding_usdt": t.funding_usdt,
        "pnl_usdt": t.pnl_usdt,
        "return_pct": t.return_pct,
        "hold_minutes": (t.exit_time - t.entry_time).total_seconds() / 60.0 if t.exit_time else None,
    }


def _summary(eq_df: pd.DataFrame, trades_df: pd.DataFrame, initial: float) -> dict:
    # 일별 PnL 기준 Sharpe
    daily = eq_df["equity"].resample("1D").last().dropna()
    daily_ret = daily.pct_change().dropna()
    sharpe_daily = (
        float(daily_ret.mean() / daily_ret.std() * np.sqrt(365))
        if daily_ret.std() > 0 else 0.0
    )
    # 분 단위 Sharpe (참고용 — autocorrelation으로 inflate됨)
    minute_ret = eq_df["equity"].pct_change().dropna()
    sharpe_minute = (
        float(minute_ret.mean() / minute_ret.std() * np.sqrt(365 * 24 * 60))
        if minute_ret.std() > 0 else 0.0
    )
    # max drawdown
    cummax = eq_df["equity"].cummax()
    dd = (eq_df["equity"] - cummax) / cummax
    max_dd = float(abs(dd.min())) if len(dd) else 0.0

    final = float(eq_df["equity"].iloc[-1]) if len(eq_df) else initial
    if not trades_df.empty:
        wins = trades_df[trades_df["pnl_usdt"] > 0]
        losses = trades_df[trades_df["pnl_usdt"] <= 0]
        win_rate = len(wins) / len(trades_df)
        avg_win = float(wins["pnl_usdt"].mean()) if len(wins) else 0.0
        avg_loss = float(losses["pnl_usdt"].mean()) if len(losses) else 0.0
        pf = (
            float(wins["pnl_usdt"].sum() / abs(losses["pnl_usdt"].sum()))
            if len(losses) and losses["pnl_usdt"].sum() != 0 else float("inf")
        )
        avg_hold = float(trades_df["hold_minutes"].mean())
    else:
        win_rate = avg_win = avg_loss = pf = avg_hold = 0.0

    calmar = ((final / initial) - 1) / max_dd if max_dd > 0 else 0.0
    return {
        "n_trades": int(len(trades_df)),
        "final_balance": final,
        "total_return_pct": (final / initial - 1) * 100 if initial else 0.0,
        "sharpe_daily": sharpe_daily,
        "sharpe_minute": sharpe_minute,
        "max_drawdown_pct": max_dd * 100,
        "calmar_ratio": calmar,
        "win_rate": win_rate,
        "avg_win_usdt": avg_win,
        "avg_loss_usdt": avg_loss,
        "profit_factor": pf,
        "avg_hold_minutes": avg_hold,
    }


def _cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data_cache/ohlcv_1m/BTC-USDT-SWAP")
    parser.add_argument("--feature-dir", default="features/btc_1m_v1")
    parser.add_argument("--model-dir", default="models/btc_lgb_v1")
    parser.add_argument("--results-dir", default="backtest/results/btc_ml_v1")
    parser.add_argument("--initial-balance", type=float, default=1000.0)
    parser.add_argument("--leverage", type=int, default=5)
    parser.add_argument("--position-pct", type=float, default=0.10)
    args = parser.parse_args()

    cfg = MLBacktestConfig(
        initial_balance=args.initial_balance,
        leverage=args.leverage,
        position_pct=args.position_pct,
    )
    res = run_ml_backtest(
        Path(args.raw_dir), Path(args.feature_dir), Path(args.model_dir), cfg,
    )

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res["equity_curve"].to_parquet(out_dir / "equity_curve.parquet", compression="zstd")
    res["trades"].to_parquet(out_dir / "trades.parquet", compression="zstd")
    (out_dir / "summary.json").write_text(json.dumps(res["summary"], indent=2, default=str))
    logger.info(f"결과 저장: {out_dir}")
    logger.info(f"summary: {json.dumps(res['summary'], indent=2)}")

    # 리포트 생성 시도 (있으면)
    try:
        from backtest.ml_report import generate_report
        generate_report(out_dir, res)
        logger.info(f"리포트 생성 완료: {out_dir / 'report.html'}")
    except Exception as e:
        logger.warning(f"리포트 생성 실패(무시 가능): {e}")


if __name__ == "__main__":
    _cli()
