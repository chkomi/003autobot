"""
ML 백테스트 리포트 생성기.
HTML (단일 파일, base64 PNG 임베드) + JSON + CSV trade log + feature_importance PNG.
"""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

# matplotlib backend: Agg (no display)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_HEADER_BANNER = (
    "Backtest assumes 1m bar resolution; live execution incurs additional 1-60s "
    "latency post-signal which will degrade Sharpe by ~10-25%."
)


def _png_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _equity_png(eq_df: pd.DataFrame, initial: float) -> tuple[str, plt.Figure]:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(eq_df.index, eq_df["equity"], lw=1.0, color="#0d6efd")
    ax.axhline(initial, color="grey", linestyle=":", lw=0.8)
    ax.set_title("Equity Curve")
    ax.set_xlabel("UTC")
    ax.set_ylabel("Balance (USDT)")
    ax.grid(alpha=0.3)
    return _png_b64(fig), fig


def _drawdown_png(eq_df: pd.DataFrame) -> str:
    cummax = eq_df["equity"].cummax()
    dd = (eq_df["equity"] - cummax) / cummax * 100
    fig, ax = plt.subplots(figsize=(11, 2.5))
    ax.fill_between(eq_df.index, dd.values, 0, color="#dc3545", alpha=0.35)
    ax.plot(eq_df.index, dd.values, lw=0.7, color="#dc3545")
    ax.set_title("Drawdown (%)")
    ax.set_xlabel("UTC")
    ax.set_ylabel("DD %")
    ax.grid(alpha=0.3)
    return _png_b64(fig)


def _feature_importance_png(model_dir: Path, feat_cols: list[str]) -> Optional[str]:
    """fold별 .lgb 모델에서 gain-based 중요도 평균 → 상위 25개 barplot."""
    files = sorted(model_dir.glob("fold_*.lgb"))
    if not files:
        return None
    importances = np.zeros(len(feat_cols), dtype=float)
    for p in files:
        booster = lgb.Booster(model_file=str(p))
        gains = booster.feature_importance(importance_type="gain")
        if len(gains) == len(feat_cols):
            importances += gains
    importances /= len(files)
    order = np.argsort(importances)[::-1][:25]
    names = [feat_cols[i] for i in order]
    vals = importances[order]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(names))[::-1], vals[::-1], color="#198754")
    ax.set_yticks(range(len(names))[::-1])
    ax.set_yticklabels(names[::-1])
    ax.set_xlabel("avg gain (across folds)")
    ax.set_title("Top 25 Feature Importance (LightGBM gain)")
    ax.grid(alpha=0.3, axis="x")
    return _png_b64(fig)


def _per_year_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["year"] = df["entry_time"].dt.year
    grp = df.groupby("year")
    out = pd.DataFrame({
        "n_trades": grp.size(),
        "win_rate": grp["pnl_usdt"].apply(lambda s: (s > 0).mean()),
        "total_pnl": grp["pnl_usdt"].sum(),
        "avg_pnl": grp["pnl_usdt"].mean(),
        "avg_hold_min": grp["hold_minutes"].mean(),
    })
    return out.round(4)


def _direction_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    df = trades_df.copy()
    df["side"] = df["direction"].map({1: "LONG", -1: "SHORT"})
    grp = df.groupby("side")
    out = pd.DataFrame({
        "n_trades": grp.size(),
        "win_rate": grp["pnl_usdt"].apply(lambda s: (s > 0).mean()),
        "total_pnl": grp["pnl_usdt"].sum(),
        "avg_pnl": grp["pnl_usdt"].mean(),
    })
    return out.round(4)


def _exit_reason_table(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    grp = trades_df.groupby("exit_reason")
    out = pd.DataFrame({
        "n_trades": grp.size(),
        "win_rate": grp["pnl_usdt"].apply(lambda s: (s > 0).mean()),
        "total_pnl": grp["pnl_usdt"].sum(),
    })
    return out.round(4)


def _fold_metrics_table(model_meta: dict) -> pd.DataFrame:
    rows = []
    for f in model_meta.get("folds", []):
        rows.append({
            "fold": f["fold_id"],
            "n_train": f["n_train"],
            "n_test": f["n_test"],
            "rmse": f["rmse"],
            "ic": f["ic"],
            "rank_ic": f["rank_ic"],
            "sign_acc": f["sign_accuracy"],
            "decile_bp": f["decile_spread"] * 1e4,
            "thr_long": f["threshold_long"],
            "thr_short": f["threshold_short"],
        })
    return pd.DataFrame(rows).round(5)


def generate_report(out_dir: Path, results: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    eq_df: pd.DataFrame = results["equity_curve"]
    trades_df: pd.DataFrame = results["trades"]
    summary: dict = results["summary"]
    model_meta: dict = results["model_metadata"]
    cfg: dict = results["config"]

    feat_cols = model_meta.get("feature_columns", [])

    eq_png, _ = _equity_png(eq_df, cfg["initial_balance"])
    dd_png = _drawdown_png(eq_df)
    model_dir_str = results.get("model_dir")
    fi_png: Optional[str] = None
    if model_dir_str:
        mdir = Path(model_dir_str)
        if mdir.exists():
            fi_png = _feature_importance_png(mdir, feat_cols)

    year_tbl = _per_year_table(trades_df)
    side_tbl = _direction_table(trades_df)
    exit_tbl = _exit_reason_table(trades_df)
    fold_tbl = _fold_metrics_table(model_meta)

    # JSON 출력
    (out_dir / "report.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "config": cfg,
        "model_summary": model_meta.get("summary", {}),
        "per_year": year_tbl.to_dict(orient="index") if not year_tbl.empty else {},
        "per_direction": side_tbl.to_dict(orient="index") if not side_tbl.empty else {},
        "per_exit_reason": exit_tbl.to_dict(orient="index") if not exit_tbl.empty else {},
    }, indent=2, default=str))

    # CSV trade log
    if not trades_df.empty:
        trades_df.to_csv(out_dir / "trades.csv", index=False)

    # 별도 PNG (참조 용)
    if fi_png:
        (out_dir / "feature_importance.png").write_bytes(base64.b64decode(fi_png))

    html = _render_html(
        summary=summary, cfg=cfg, model_meta=model_meta,
        eq_png=eq_png, dd_png=dd_png, fi_png=fi_png,
        year_tbl=year_tbl, side_tbl=side_tbl,
        exit_tbl=exit_tbl, fold_tbl=fold_tbl,
    )
    out_path = out_dir / "report.html"
    out_path.write_text(html)
    return out_path


def _table_html(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<i>(데이터 없음)</i>"
    return df.to_html(border=0, classes="tbl", escape=False)


def _render_html(
    *, summary: dict, cfg: dict, model_meta: dict,
    eq_png: str, dd_png: str, fi_png: Optional[str],
    year_tbl: pd.DataFrame, side_tbl: pd.DataFrame,
    exit_tbl: pd.DataFrame, fold_tbl: pd.DataFrame,
) -> str:
    summary_pretty = json.dumps(summary, indent=2, default=str)
    cfg_pretty = json.dumps(cfg, indent=2)
    model_summary = json.dumps(model_meta.get("summary", {}), indent=2)

    fi_block = (
        f'<img src="data:image/png;base64,{fi_png}" style="max-width:100%"/>'
        if fi_png else "<i>모델 디렉토리 미발견 → 생략</i>"
    )
    return f"""<!doctype html>
<html lang="ko">
<head><meta charset="utf-8"/>
<title>BTC ML Backtest Report</title>
<style>
body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif; max-width:1200px; margin:24px auto; padding:0 16px; color:#222; }}
h1 {{ font-size:1.6rem; margin-bottom:0.2em; }}
h2 {{ margin-top:1.6em; border-bottom:2px solid #eee; padding-bottom:4px; }}
.banner {{ background:#fff3cd; border:1px solid #ffe69c; padding:10px 14px; border-radius:6px; color:#664d03; }}
pre {{ background:#f6f8fa; padding:10px; border-radius:6px; overflow-x:auto; font-size:0.85rem; }}
table.tbl {{ border-collapse:collapse; }}
table.tbl th, table.tbl td {{ padding:6px 12px; text-align:right; }}
table.tbl th {{ background:#f1f3f5; }}
table.tbl tr:nth-child(odd) {{ background:#fcfcfc; }}
.metric {{ display:inline-block; margin:6px 12px 6px 0; }}
.metric-label {{ font-size:0.85rem; color:#666; }}
.metric-value {{ font-size:1.2rem; font-weight:600; }}
img {{ max-width:100%; }}
</style>
</head>
<body>
<h1>BTC ML 백테스트 리포트</h1>
<p class="banner"><b>주의:</b> {_HEADER_BANNER}</p>

<h2>핵심 지표</h2>
<div>
  <div class="metric"><div class="metric-label">최종 잔고</div><div class="metric-value">{summary['final_balance']:.2f} USDT</div></div>
  <div class="metric"><div class="metric-label">총 수익률</div><div class="metric-value">{summary['total_return_pct']:+.2f}%</div></div>
  <div class="metric"><div class="metric-label">Sharpe (일)</div><div class="metric-value">{summary['sharpe_daily']:.2f}</div></div>
  <div class="metric"><div class="metric-label">Sharpe (분 - 참고)</div><div class="metric-value">{summary['sharpe_minute']:.2f}</div></div>
  <div class="metric"><div class="metric-label">Max DD</div><div class="metric-value">{summary['max_drawdown_pct']:.2f}%</div></div>
  <div class="metric"><div class="metric-label">Calmar</div><div class="metric-value">{summary['calmar_ratio']:.2f}</div></div>
  <div class="metric"><div class="metric-label">거래 수</div><div class="metric-value">{summary['n_trades']}</div></div>
  <div class="metric"><div class="metric-label">승률</div><div class="metric-value">{summary['win_rate']*100:.1f}%</div></div>
  <div class="metric"><div class="metric-label">PF</div><div class="metric-value">{summary['profit_factor']:.2f}</div></div>
  <div class="metric"><div class="metric-label">평균 보유</div><div class="metric-value">{summary['avg_hold_minutes']:.1f}분</div></div>
</div>

<h2>자본 곡선</h2>
<img src="data:image/png;base64,{eq_png}"/>
<h2>드로다운</h2>
<img src="data:image/png;base64,{dd_png}"/>

<h2>연도별 성과</h2>
{_table_html(year_tbl)}

<h2>방향별 성과</h2>
{_table_html(side_tbl)}

<h2>청산 사유별</h2>
{_table_html(exit_tbl)}

<h2>Walk-Forward Fold 메트릭</h2>
{_table_html(fold_tbl)}

<h2>Feature Importance</h2>
{fi_block}

<h2>모델 학습 요약</h2>
<pre>{model_summary}</pre>

<h2>백테스트 설정</h2>
<pre>{cfg_pretty}</pre>

<h2>전체 summary</h2>
<pre>{summary_pretty}</pre>

</body></html>
"""
