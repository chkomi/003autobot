#!/bin/bash
# =============================================================
# Cloudflare Tunnel LaunchAgent 자동시작 설정
# 실행: bash setup_tunnel_autostart.sh
#
# 이 스크립트는 Mac 로그인 시 터널을 자동으로 시작하는
# LaunchAgent를 등록합니다. URL은 재시작마다 바뀝니다.
# URL 고정을 원하면 Cloudflare 계정 + 도메인이 필요합니다.
# =============================================================

set -e

LABEL="com.autobot.tunnel"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST_FILE="$PLIST_DIR/$LABEL.plist"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$LOG_DIR"

# cloudflared 경로 확인
CLOUDFLARED_PATH=$(command -v cloudflared 2>/dev/null || echo "")
if [ -z "$CLOUDFLARED_PATH" ]; then
    echo "❌ cloudflared가 설치되어 있지 않습니다."
    echo "   먼저 start_tunnel.command를 실행하거나:"
    echo "   brew install cloudflare/cloudflare/cloudflared"
    exit 1
fi

echo "🔧 LaunchAgent 등록 중..."
echo "   cloudflared: $CLOUDFLARED_PATH"
echo "   plist: $PLIST_FILE"

cat > "$PLIST_FILE" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$LABEL</string>

  <key>ProgramArguments</key>
  <array>
    <string>$CLOUDFLARED_PATH</string>
    <string>tunnel</string>
    <string>--url</string>
    <string>http://localhost:8080</string>
    <string>--logfile</string>
    <string>$LOG_DIR/tunnel.log</string>
    <string>--no-autoupdate</string>
  </array>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <dict>
    <key>NetworkState</key>
    <true/>
  </dict>

  <key>StandardOutPath</key>
  <string>$LOG_DIR/tunnel_stdout.log</string>

  <key>StandardErrorPath</key>
  <string>$LOG_DIR/tunnel_stderr.log</string>

  <key>ThrottleInterval</key>
  <integer>30</integer>
</dict>
</plist>
EOF

# 기존 서비스 언로드 (오류 무시)
launchctl unload "$PLIST_FILE" 2>/dev/null || true

# 새 서비스 로드
launchctl load "$PLIST_FILE"

echo ""
echo "✅ 터널 LaunchAgent 등록 완료!"
echo ""
echo "   로그인 시 자동으로 터널이 시작됩니다."
echo ""
echo "   URL 확인 (30초 후):"
echo "   grep trycloudflare $LOG_DIR/tunnel.log"
echo ""
echo "   서비스 중지:"
echo "   launchctl unload ~/Library/LaunchAgents/$LABEL.plist"
