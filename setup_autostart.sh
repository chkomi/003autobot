#!/bin/bash
# =============================================================
# Autobot macOS LaunchAgent 자동 시작 설정 스크립트
# 사용법: bash setup_autostart.sh
# 실행 후 맥을 재시작하거나 직접 launchctl로 서비스를 시작하세요.
# =============================================================

set -e

PLIST_LABEL="com.autobot"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_FILE="$PLIST_DIR/$PLIST_LABEL.plist"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

# python3 경로 자동 감지 (pyenv, homebrew, 시스템 순)
if command -v python3 &>/dev/null; then
    PYTHON_PATH="$(command -v python3)"
else
    echo "❌ python3를 찾을 수 없습니다. Python 3를 설치한 뒤 다시 실행하세요."
    exit 1
fi

echo "🔧 설정 시작..."
echo "   프로젝트: $SCRIPT_DIR"
echo "   Python:   $PYTHON_PATH"
echo "   plist:    $PLIST_FILE"
echo "   로그:     $LOG_DIR/"
echo ""

# 디렉터리 생성
mkdir -p "$PLIST_DIR"
mkdir -p "$LOG_DIR"

# 기존 서비스 언로드 (오류 무시)
if launchctl list | grep -q "$PLIST_LABEL" 2>/dev/null; then
    echo "⏹  기존 서비스 언로드 중..."
    launchctl unload "$PLIST_FILE" 2>/dev/null || true
fi

# plist 파일 생성
cat > "$PLIST_FILE" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <!-- 서비스 식별자 -->
    <key>Label</key>
    <string>${PLIST_LABEL}</string>

    <!-- 실행할 명령어 -->
    <key>ProgramArguments</key>
    <array>
        <string>${PYTHON_PATH}</string>
        <string>${SCRIPT_DIR}/main.py</string>
    </array>

    <!-- 작업 디렉터리 -->
    <key>WorkingDirectory</key>
    <string>${SCRIPT_DIR}</string>

    <!-- 로그인 시 자동 시작 -->
    <key>RunAtLoad</key>
    <true/>

    <!-- 크래시/종료 시 자동 재시작 -->
    <key>KeepAlive</key>
    <true/>

    <!-- 재시작 쿨다운: 60초 (너무 빠른 반복 재시작 방지) -->
    <key>ThrottleInterval</key>
    <integer>60</integer>

    <!-- 표준 출력 로그 -->
    <key>StandardOutPath</key>
    <string>${LOG_DIR}/launchd_stdout.log</string>

    <!-- 표준 에러 로그 -->
    <key>StandardErrorPath</key>
    <string>${LOG_DIR}/launchd_stderr.log</string>

    <!-- 환경 변수 -->
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin</string>
        <key>HOME</key>
        <string>${HOME}</string>
        <key>LANG</key>
        <string>en_US.UTF-8</string>
    </dict>
</dict>
</plist>
PLIST_EOF

echo "✅ plist 파일 생성 완료"

# 서비스 로드
launchctl load "$PLIST_FILE"
echo "✅ LaunchAgent 등록 완료 — 봇이 백그라운드에서 실행 중입니다."

echo ""
echo "📋 서비스 관리 명령어:"
echo "   상태 확인: launchctl list | grep autobot"
echo "   서비스 중지: launchctl unload ~/Library/LaunchAgents/com.autobot.plist"
echo "   서비스 시작: launchctl load ~/Library/LaunchAgents/com.autobot.plist"
echo "   로그 보기:  tail -f \"$LOG_DIR/launchd_stdout.log\""
echo "   에러 보기:  tail -f \"$LOG_DIR/launchd_stderr.log\""
echo ""
echo "🔄 봇은 크래시나 맥 재시작 시 자동으로 재시작됩니다."
