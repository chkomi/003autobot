#!/bin/bash
cd "/Users/yun/Documents/Business/003. Autobot"

echo "=============================="
echo "  Feature 재빌드 시작 (2020+)"
echo "=============================="
echo ""
echo "raw 데이터: data_cache/ohlcv_1m/BTC-USDT-SWAP"
echo "출력 경로:  features/btc_1m_v1"
echo "예상 시간:  1~2시간"
echo ""

python3 -m strategy.ml_features \
    --raw-dir data_cache/ohlcv_1m/BTC-USDT-SWAP \
    --out-dir features/btc_1m_v1

echo ""
echo "=============================="
echo "  완료! features/btc_1m_v1 확인"
echo "=============================="
