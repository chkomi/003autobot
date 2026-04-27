"""
Walk-forward LightGBM 회귀 학습기 (BTC 1m → next-15m log return 예측).

설계:
  - expanding window (max 36개월 cap)
  - 매월 fold 생성: 학습=과거 18~36개월, 테스트=다음 1개월, embargo=마지막 60분 drop
  - inner-validation: 학습 데이터의 마지막 10%로 early stopping
  - 모델 저장: models/btc_lgb_v1/fold_YYYY-MM.lgb + metadata.json
  - 메트릭: RMSE, IC (Pearson), Rank IC (Spearman), sign accuracy, decile spread
  - threshold: 학습 fold prediction의 75/25 percentile

본 모듈은 strategy/ml_filter.py(post-trade 분류 + sklearn)와 별개임.
이쪽은 entry-decision 회귀 + LightGBM.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import spearmanr
from tqdm.auto import tqdm

from strategy.ml_features import (
    FeatureConfig,
    feature_columns,
    load_feature_range,
    target_column,
)


@dataclass
class WalkForwardConfig:
    train_window_months: int = 18  # 초기 학습 윈도우 길이
    train_window_max_months: int = 36  # expanding cap
    test_window_months: int = 1
    embargo_minutes: int = 60  # 학습 끝 N분 drop (forward target 누설 방지)
    threshold_quantile: float = 0.75  # 75% percentile of |y_hat| → long, -75% → short
    primary_target: int = 15
    feature_cfg: FeatureConfig = field(default_factory=FeatureConfig)


DEFAULT_LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 500,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 1.0,
    "verbose": -1,
    "num_threads": 0,  # 0 = OpenMP default
}


@dataclass
class FoldResult:
    fold_id: str  # "YYYY-MM" (test month)
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    rmse: float
    ic: float          # Pearson 상관 (예측 vs 실제)
    rank_ic: float     # Spearman 상관
    sign_accuracy: float
    decile_spread: float
    threshold_long: float
    threshold_short: float
    model_path: str
    n_boost_used: int

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "n_train": self.n_train,
            "n_test": self.n_test,
            "rmse": float(self.rmse),
            "ic": float(self.ic),
            "rank_ic": float(self.rank_ic),
            "sign_accuracy": float(self.sign_accuracy),
            "decile_spread": float(self.decile_spread),
            "threshold_long": float(self.threshold_long),
            "threshold_short": float(self.threshold_short),
            "model_path": self.model_path,
            "n_boost_used": self.n_boost_used,
        }


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _drop_train_rows(train: pd.DataFrame, target_col: str, embargo_minutes: int) -> pd.DataFrame:
    """학습 데이터에서 (a) 타겟 NaN 제거 (forward window 미존재) + (b) 마지막 embargo_minutes 제거."""
    train = train.dropna(subset=[target_col])
    if not train.empty and embargo_minutes > 0:
        cutoff = train.index.max() - pd.Timedelta(minutes=embargo_minutes)
        train = train.loc[train.index <= cutoff]
    return train


def _generate_folds(
    feat_df: pd.DataFrame,
    cfg: WalkForwardConfig,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """(train_start, train_end_exclusive, test_start, test_end_exclusive) 튜플 리스트."""
    if feat_df.empty:
        return []
    data_start = feat_df.index.min().normalize()
    data_end = feat_df.index.max()

    # 첫 fold의 test_start = data_start + train_window_months
    test_start = (
        pd.Timestamp(year=data_start.year, month=data_start.month, day=1, tz="UTC")
        + pd.DateOffset(months=cfg.train_window_months)
    )

    folds = []
    while True:
        test_end = test_start + pd.DateOffset(months=cfg.test_window_months)
        if test_start >= data_end:
            break
        train_end = test_start  # exclusive
        train_start_min = train_end - pd.DateOffset(months=cfg.train_window_max_months)
        train_start = max(train_start_min, data_start)
        if test_end > data_end + pd.Timedelta(minutes=1):
            test_end = data_end + pd.Timedelta(minutes=1)
        folds.append((train_start, train_end, test_start, test_end))
        test_start = test_end
    return folds


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        ic = 0.0
        rank_ic = 0.0
    else:
        ic = float(np.corrcoef(y_true, y_pred)[0, 1])
        rank_ic = float(spearmanr(y_true, y_pred).statistic)
    sign_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    # decile spread: top vs bottom decile of predicted, realized return
    n = len(y_pred)
    if n >= 100:
        order = np.argsort(y_pred)
        bot = y_true[order[: n // 10]].mean()
        top = y_true[order[-n // 10 :]].mean()
        decile_spread = float(top - bot)
    else:
        decile_spread = 0.0
    return {
        "rmse": rmse,
        "ic": ic,
        "rank_ic": rank_ic,
        "sign_accuracy": sign_acc,
        "decile_spread": decile_spread,
    }


def _train_one_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feat_cols: list[str],
    target_col: str,
    cfg: WalkForwardConfig,
    out_dir: Path,
) -> FoldResult:
    fold_id = f"{test.index.min().year:04d}-{test.index.min().month:02d}"

    train_clean = _drop_train_rows(train, target_col, cfg.embargo_minutes)

    # inner validation: 마지막 10% 시간순
    n_train = len(train_clean)
    cutoff = int(n_train * 0.9)
    inner_train = train_clean.iloc[:cutoff]
    inner_val = train_clean.iloc[cutoff:]

    X_tr = inner_train[feat_cols].astype("float32").values
    y_tr = inner_train[target_col].astype("float32").values
    X_va = inner_val[feat_cols].astype("float32").values
    y_va = inner_val[target_col].astype("float32").values

    dtrain = lgb.Dataset(X_tr, y_tr)
    dval = lgb.Dataset(X_va, y_va, reference=dtrain)

    booster = lgb.train(
        DEFAULT_LGB_PARAMS,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    n_boost_used = booster.best_iteration or booster.num_trees()

    # 테스트 평가
    test_clean = test.dropna(subset=[target_col])
    X_te = test_clean[feat_cols].astype("float32").values
    y_te = test_clean[target_col].astype("float32").values
    y_pred = booster.predict(X_te, num_iteration=n_boost_used)

    metrics = _eval_metrics(y_te, y_pred)

    # threshold: 학습 fold(전체) 예측 분포의 75/25 percentile
    X_train_all = train_clean[feat_cols].astype("float32").values
    y_pred_train = booster.predict(X_train_all, num_iteration=n_boost_used)
    pos = y_pred_train[y_pred_train > 0]
    neg = y_pred_train[y_pred_train < 0]
    threshold_long = float(np.quantile(pos, cfg.threshold_quantile)) if len(pos) > 100 else 0.001
    threshold_short = -float(np.quantile(np.abs(neg), cfg.threshold_quantile)) if len(neg) > 100 else -0.001

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"fold_{fold_id}.lgb"
    booster.save_model(str(model_path), num_iteration=n_boost_used)

    return FoldResult(
        fold_id=fold_id,
        train_start=train_clean.index.min(),
        train_end=train_clean.index.max(),
        test_start=test_clean.index.min(),
        test_end=test_clean.index.max(),
        n_train=len(train_clean),
        n_test=len(test_clean),
        rmse=metrics["rmse"],
        ic=metrics["ic"],
        rank_ic=metrics["rank_ic"],
        sign_accuracy=metrics["sign_accuracy"],
        decile_spread=metrics["decile_spread"],
        threshold_long=threshold_long,
        threshold_short=threshold_short,
        model_path=str(model_path.relative_to(model_path.parent.parent)),
        n_boost_used=n_boost_used,
    )


def train_walk_forward(
    feature_dir: Path,
    out_dir: Path,
    cfg: Optional[WalkForwardConfig] = None,
) -> dict:
    cfg = cfg or WalkForwardConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_cols = feature_columns(cfg.feature_cfg)
    target_col = target_column(cfg.feature_cfg)

    # 전체 피처 한 번에 로드 (Parquet → in-memory)
    files = sorted(feature_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"{feature_dir} 비어있음")
    logger.info(f"피처 {len(files)}개 월 로딩...")
    feat_df = pd.concat([pd.read_parquet(p) for p in files]).sort_index()
    feat_df = feat_df[~feat_df.index.duplicated(keep="last")]
    logger.info(f"feature rows: {len(feat_df):,}, cols: {feat_df.shape[1]}")

    folds = _generate_folds(feat_df, cfg)
    logger.info(f"총 {len(folds)}개 fold 생성")

    results: list[FoldResult] = []
    for train_s, train_e, test_s, test_e in tqdm(folds, desc="walk-forward folds"):
        train = feat_df.loc[(feat_df.index >= train_s) & (feat_df.index < train_e)]
        test = feat_df.loc[(feat_df.index >= test_s) & (feat_df.index < test_e)]
        if len(train) < 1000 or len(test) < 100:
            logger.warning(f"fold skip (n_train={len(train)}, n_test={len(test)})")
            continue
        res = _train_one_fold(train, test, feat_cols, target_col, cfg, out_dir)
        results.append(res)
        logger.info(
            f"[{res.fold_id}] n_train={res.n_train:,} n_test={res.n_test:,} "
            f"rmse={res.rmse:.5f} IC={res.ic:.4f} rankIC={res.rank_ic:.4f} "
            f"sign={res.sign_accuracy:.3f} decile={res.decile_spread*1e4:+.2f}bp "
            f"thr_long={res.threshold_long:.5f} thr_short={res.threshold_short:.5f}"
        )

    metadata = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "feature_columns": feat_cols,
        "target_column": target_col,
        "lgb_params": DEFAULT_LGB_PARAMS,
        "walk_forward_config": {
            "train_window_months": cfg.train_window_months,
            "train_window_max_months": cfg.train_window_max_months,
            "test_window_months": cfg.test_window_months,
            "embargo_minutes": cfg.embargo_minutes,
            "threshold_quantile": cfg.threshold_quantile,
        },
        "folds": [r.to_dict() for r in results],
        "summary": {
            "n_folds": len(results),
            "mean_rmse": float(np.mean([r.rmse for r in results])) if results else 0.0,
            "mean_ic": float(np.mean([r.ic for r in results])) if results else 0.0,
            "mean_rank_ic": float(np.mean([r.rank_ic for r in results])) if results else 0.0,
            "mean_sign_accuracy": float(np.mean([r.sign_accuracy for r in results])) if results else 0.0,
            "mean_decile_spread": float(np.mean([r.decile_spread for r in results])) if results else 0.0,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    logger.info(
        f"완료. {len(results)}개 fold. "
        f"평균 rank-IC = {metadata['summary']['mean_rank_ic']:.4f}"
    )
    return metadata


def load_fold_models(model_dir: Path) -> tuple[dict[str, lgb.Booster], dict]:
    """metadata.json + .lgb 모델 모두 로드. 백테스트가 사용."""
    meta = json.loads((model_dir / "metadata.json").read_text())
    boosters: dict[str, lgb.Booster] = {}
    for fold in meta["folds"]:
        path = model_dir / Path(fold["model_path"]).name
        boosters[fold["fold_id"]] = lgb.Booster(model_file=str(path))
    return boosters, meta


def _cli():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-dir", default="features/btc_1m_v1",
        help="strategy/ml_features.py가 만든 월별 피처 디렉토리",
    )
    parser.add_argument(
        "--out-dir", default="models/btc_lgb_v1",
        help="모델 저장 디렉토리",
    )
    parser.add_argument("--train-months", type=int, default=18)
    parser.add_argument("--max-train-months", type=int, default=36)
    args = parser.parse_args()
    cfg = WalkForwardConfig(
        train_window_months=args.train_months,
        train_window_max_months=args.max_train_months,
    )
    train_walk_forward(Path(args.feature_dir), Path(args.out_dir), cfg)


if __name__ == "__main__":
    _cli()
