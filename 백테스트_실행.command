#!/bin/bash
# OKX 전종목 백테스트 실행 스크립트
# 더블클릭하면 Terminal에서 실행됩니다.

cd "/Users/yun/Documents/Business/003. Autobot"
echo "============================================"
echo "OKX 전종목 백테스트 시작 (2024-01-01 ~ 2026-04-23)"
echo "완료까지 30~90분 소요될 수 있습니다."
echo "============================================"
python3 run_full_backtest.py
echo ""
echo "완료! report/backtest_results.csv 를 확인하세요."
read -p "엔터를 누르면 창이 닫힙니다..."
