#!/bin/bash
# =============================================================
# Cloudflare Quick Tunnel 실행 스크립트
# 더블클릭으로 실행 — Mac API를 인터넷에 안전하게 노출
# =============================================================

cd "$(dirname "$0")"
LOG_DIR="./logs"
TUNNEL_LOG="$LOG_DIR/tunnel.log"
URL_FILE="$LOG_DIR/tunnel_url.txt"

mkdir -p "$LOG_DIR"

echo "🔧 Cloudflare Tunnel 시작 중..."

# ── cloudflared 설치 확인 ────────────────────────────────────
if ! command -v cloudflared &>/dev/null; then
    echo ""
    echo "📦 cloudflared 미설치 → Homebrew로 설치합니다..."
    if ! command -v brew &>/dev/null; then
        echo "❌ Homebrew가 없습니다. https://brew.sh 에서 먼저 설치하세요."
        read -p "계속하려면 Enter..."
        exit 1
    fi
    brew install cloudflare/cloudflare/cloudflared
    echo "✅ cloudflared 설치 완료"
fi

# ── 기존 터널 프로세스 종료 ───────────────────────────────────
pkill -f "cloudflared tunnel" 2>/dev/null && echo "이전 터널 종료됨" || true
sleep 1

# ── 터널 시작 (백그라운드, 로그 파일에 기록) ─────────────────
echo "" > "$TUNNEL_LOG"
cloudflared tunnel --url http://localhost:8080 \
    --logfile "$TUNNEL_LOG" \
    --no-autoupdate &
TUNNEL_PID=$!
echo $TUNNEL_PID > "$LOG_DIR/tunnel.pid"

echo ""
echo "⏳ 터널 URL 확인 중 (최대 15초)..."

# URL이 로그에 나타날 때까지 대기
for i in $(seq 1 30); do
    sleep 0.5
    URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1)
    if [ -n "$URL" ]; then
        break
    fi
done

if [ -z "$URL" ]; then
    echo ""
    echo "⚠️  URL 자동 감지 실패. 로그 확인: $TUNNEL_LOG"
    echo "   수동으로 URL을 확인하고 Vercel 환경변수에 설정하세요."
    cat "$TUNNEL_LOG" | grep -i "trycloudflare\|error\|ERR" | tail -10
else
    echo "$URL" > "$URL_FILE"
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  ✅ 터널 실행 중!                                        ║"
    echo "║                                                          ║"
    echo "║  URL: $URL"
    echo "║                                                          ║"
    echo "║  ⚠️  이 URL을 Vercel 환경변수에 설정하세요:             ║"
    echo "║     AUTOBOT_API_URL = (위 URL)                          ║"
    echo "║                                                          ║"
    echo "║  Mac 재시작 시 URL이 바뀌므로 Vercel에서 다시 업데이트  ║"
    echo "║  해야 합니다. URL 고정은 setup_tunnel_autostart.sh 참고 ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "💡 API 확인 (브라우저에서 열어보세요):"
    echo "   $URL/api/v1/ops/health"
fi

echo ""
echo "터널 PID: $TUNNEL_PID | 로그: $TUNNEL_LOG"
echo "종료하려면 이 창을 닫고: pkill -f 'cloudflared tunnel'"
echo ""
read -p "이 창을 닫으면 터널도 종료됩니다. 계속 열어두세요. Enter로 확인..."
wait $TUNNEL_PID
