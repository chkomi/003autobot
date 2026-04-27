"""
v1 full baseline 봉인 스크립트.

models/btc_lgb_v1_full/ 와 backtest/results/btc_ml_v1_full/ 가 모두 존재한다는 가정 하에
backtest/results/btc_ml_v1_full/baseline_manifest.json 를 생성한다.

이후 모든 ML 개선(Phase 0+) 은 이 baseline 과 비교한다.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_MANIFEST = PROJECT_ROOT / "data_cache" / "ohlcv_1m" / "baseline_manifest.json"
MODEL_DIR = PROJECT_ROOT / "models" / "btc_lgb_v1_full"
RESULTS_DIR = PROJECT_ROOT / "backtest" / "results" / "btc_ml_v1_full"
FEATURE_DIR = PROJECT_ROOT / "features" / "btc_1m_v1_full"
OUTPUT = RESULTS_DIR / "baseline_manifest.json"


def _git_info() -> dict:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=PROJECT_ROOT, text=True
        ).strip()

    sha = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=PROJECT_ROOT, text=True
    ).splitlines()
    modified = sorted(line[3:] for line in status if line.startswith(" M"))
    new = sorted(line[3:] for line in status if line.startswith("??"))
    return {
        "sha": sha,
        "branch": branch,
        "uncommitted_modified": modified,
        "uncommitted_new": new,
    }


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _monthly_hit_rate(per_year: dict) -> dict:
    hit = {}
    for year, stats in per_year.items():
        hit[year] = {
            "n_trades": stats.get("n_trades"),
            "win_rate": stats.get("win_rate"),
            "total_pnl": stats.get("total_pnl"),
            "avg_pnl": stats.get("avg_pnl"),
        }
    return hit


def main() -> None:
    if not RESULTS_DIR.exists():
        raise SystemExit(f"missing results dir: {RESULTS_DIR}")
    if not MODEL_DIR.exists():
        raise SystemExit(f"missing model dir: {MODEL_DIR}")
    if not FEATURE_DIR.exists():
        raise SystemExit(f"missing feature dir: {FEATURE_DIR}")

    raw = _load_json(RAW_MANIFEST) if RAW_MANIFEST.exists() else {}
    summary = _load_json(RESULTS_DIR / "summary.json")
    report = _load_json(RESULTS_DIR / "report.json")
    metadata = _load_json(MODEL_DIR / "metadata.json")

    feature_files = sorted(p.name for p in FEATURE_DIR.glob("*.parquet"))
    fold_files = sorted(p.name for p in MODEL_DIR.glob("fold_*.lgb"))

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": "v4-phase-1-baseline-sealed",
        "purpose": "BTC 1m 2020+ v1 full ML baseline (frozen reference for all Phase 0+ experiments)",
        "git": _git_info(),
        "raw_data": raw.get("data", {}),
        "raw_validation_gates": raw.get("validation_gates", {}),
        "features": {
            "dir": str(FEATURE_DIR.relative_to(PROJECT_ROOT)),
            "month_count": len(feature_files),
            "files_first": feature_files[:3],
            "files_last": feature_files[-3:],
            "feature_columns": metadata.get("feature_columns", []),
            "target_column": metadata.get("target_column"),
        },
        "model": {
            "dir": str(MODEL_DIR.relative_to(PROJECT_ROOT)),
            "n_folds": len(fold_files),
            "lgb_params": metadata.get("lgb_params"),
            "walk_forward_config": metadata.get("walk_forward_config"),
            "model_summary": report.get("model_summary"),
        },
        "backtest": {
            "dir": str(RESULTS_DIR.relative_to(PROJECT_ROOT)),
            "config": report.get("config"),
            "summary": summary,
            "per_year": report.get("per_year"),
            "monthly_hit_rate": _monthly_hit_rate(report.get("per_year", {})),
        },
        "live_candidate_thresholds": {
            "rank_ic_min": 0.025,
            "daily_sharpe_min": 1.5,
            "profit_factor_min": 1.25,
            "max_drawdown_pct_max": 25.0,
        },
        "notes": [
            "Phase 0 의 모든 변경(Triple Barrier, Vol-Norm, Multi-horizon, Meta-Label, Cost-aware EV, Funding/OI, On-chain) 효과는 이 baseline 과 비교해 측정한다.",
            "본 baseline 은 fee 0.05% / slippage 0.05% / funding 0.0001%/8h 가정에서 EV 음수.",
            "rank-IC 는 양수지만 cost 후 PF<1 → 라이브 후보 아님.",
        ],
    }

    OUTPUT.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"sealed: {OUTPUT}")


if __name__ == "__main__":
    main()
