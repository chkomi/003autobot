#!/bin/bash
cd "/Users/yun/Documents/Business/003. Autobot"

echo "=============================="
echo "  Autobot 재시작 스크립트"
echo "=============================="

# 기존 봇 프로세스 종료
echo ""
echo "기존 봇 프로세스 종료 중..."
pkill -f "python.*main.py" 2>/dev/null && echo "기존 프로세스 종료됨" || echo "실행 중인 프로세스 없음"
sleep 2

# 봇 재시작
echo ""
echo "봇 재시작 중..."
python3 main.py
