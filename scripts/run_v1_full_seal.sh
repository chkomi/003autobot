#!/bin/bash
# Phase -1.4 + Phase -1.5: feature 재빌드 → v1 full 학습 → 백테스트 → baseline 봉인.
# 작업 시간 예상: 2~4시간 (raw 76개월, walk-forward 56 fold).

set -euo pipefail

cd "/Users/yun/Documents/Business/003. Autobot"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/v1_full_seal_${TS}"
mkdir -p "${LOG_DIR}"

MAIN_LOG="${LOG_DIR}/run.log"
exec >>"${MAIN_LOG}" 2>&1

echo "=============================="
echo "[$(date '+%F %T')] v1 full seal 시작"
echo "log dir: ${LOG_DIR}"
echo "=============================="

echo "[$(date '+%F %T')] Phase -1.4: feature 재빌드 (-> features/btc_1m_v1_full)"
python3 -m strategy.ml_features \
    --raw-dir data_cache/ohlcv_1m/BTC-USDT-SWAP \
    --out-dir features/btc_1m_v1_full \
    > "${LOG_DIR}/01_feature_build.log" 2>&1

echo "[$(date '+%F %T')] Phase -1.5a: walk-forward 학습 (-> models/btc_lgb_v1_full)"
python3 -m strategy.ml_model \
    --feature-dir features/btc_1m_v1_full \
    --out-dir models/btc_lgb_v1_full \
    > "${LOG_DIR}/02_model_train.log" 2>&1

echo "[$(date '+%F %T')] Phase -1.5b: backtest (-> backtest/results/btc_ml_v1_full)"
python3 -m backtest.ml_backtester \
    --raw-dir data_cache/ohlcv_1m/BTC-USDT-SWAP \
    --feature-dir features/btc_1m_v1_full \
    --model-dir models/btc_lgb_v1_full \
    --results-dir backtest/results/btc_ml_v1_full \
    > "${LOG_DIR}/03_backtest.log" 2>&1

echo "[$(date '+%F %T')] Phase -1.5c: baseline_manifest.json 봉인"
python3 scripts/seal_v1_baseline.py \
    > "${LOG_DIR}/04_seal.log" 2>&1

echo "=============================="
echo "[$(date '+%F %T')] 완료"
echo "manifest: backtest/results/btc_ml_v1_full/baseline_manifest.json"
echo "=============================="
