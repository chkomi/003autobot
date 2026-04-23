#!/bin/bash
cd "/Users/yun/Documents/Business/003. Autobot"

echo "======================================"
echo "  Autobot 전체 설치 및 재시작"
echo "======================================"
echo ""

# 1) 기존 봇 프로세스 종료
echo "[ 1/3 ] 기존 봇 프로세스 종료 중..."
pkill -f "python.*main.py" 2>/dev/null && echo "       기존 프로세스 종료됨" || echo "       실행 중인 프로세스 없음"
sleep 2

# 2) LaunchAgent 자동 재시작 데몬 설치
echo "[ 2/3 ] 자동 재시작 데몬 설치 중..."
bash "/Users/yun/Documents/Business/003. Autobot/setup_autostart.sh"

echo ""
echo "[ 3/3 ] 완료!"
echo "       봇이 LaunchAgent로 백그라운드에서 실행됩니다."
echo "       맥 재시작 및 크래시 시 자동으로 복구됩니다."
echo ""
echo "로그 확인: tail -f ~/Documents/Business/003.\ Autobot/logs/launchd_stdout.log"
