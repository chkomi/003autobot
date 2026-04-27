/**
 * 003 Autobot — 메인 대시보드
 *
 * Next.js Server Component: 빌드/요청 시 FastAPI에서 직접 데이터를 가져온다.
 * CORS 문제 없음 (서버 → 서버 요청).
 * ISR(revalidate: 60)으로 60초마다 자동 갱신.
 *
 * 환경변수 (Vercel 대시보드에 설정):
 *   AUTOBOT_API_URL   FastAPI 서버 URL (e.g. https://xyz.trycloudflare.com)
 *   AUTOBOT_API_KEY   X-API-Key 값 (AUTOBOT_API_API_KEY 와 동일)
 */

// ── 타입 정의 ─────────────────────────────────────────────────

interface LeaderboardEntry {
  rank: number;
  symbol: string;
  long_score: number;
  short_score: number;
  direction: string;
  signal_strength: string;
  last_price: number | null;
  change_24h: number | null;
}

interface HealthStatus {
  status: string;
}

interface ApiResponse<T> {
  data: T;
  generated_at: string;
}

// ── 데이터 페칭 ───────────────────────────────────────────────

const API_URL = process.env.AUTOBOT_API_URL ?? "";
const API_KEY = process.env.AUTOBOT_API_KEY ?? "";

async function fetchLeaderboard(): Promise<LeaderboardEntry[]> {
  if (!API_URL) return [];
  try {
    const res = await fetch(
      `${API_URL}/api/v1/market/leaderboard?direction=LONG&limit=20`,
      {
        headers: { "X-API-Key": API_KEY },
        next: { revalidate: 60 },
      }
    );
    if (!res.ok) return [];
    const json: ApiResponse<LeaderboardEntry[]> = await res.json();
    return json.data ?? [];
  } catch {
    return [];
  }
}

async function fetchHealth(): Promise<HealthStatus | null> {
  if (!API_URL) return null;
  try {
    const res = await fetch(`${API_URL}/api/v1/ops/health`, {
      next: { revalidate: 30 },
    });
    if (!res.ok) return null;
    const json: ApiResponse<HealthStatus> = await res.json();
    return json.data ?? null;
  } catch {
    return null;
  }
}

// ── 유틸 ──────────────────────────────────────────────────────

function fmtScore(v: number) {
  return v.toFixed(1);
}

function fmtPrice(v: number | null) {
  if (v == null) return "—";
  return v >= 1000
    ? "$" + v.toLocaleString("en-US", { maximumFractionDigits: 0 })
    : "$" + v.toFixed(3);
}

function fmtChange(v: number | null) {
  if (v == null) return "";
  const sign = v >= 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

function directionLabel(d: string) {
  if (d === "LONG") return "Long";
  if (d === "SHORT") return "Short";
  return "Neutral";
}

// ── 페이지 ────────────────────────────────────────────────────

export default async function Home() {
  const [leaderboard, health] = await Promise.all([
    fetchLeaderboard(),
    fetchHealth(),
  ]);

  const apiLive = health?.status === "ok";
  const symbolCount = leaderboard.length;

  const reportCards = [
    {
      label: "Live Symbols",
      value: symbolCount > 0 ? String(symbolCount) : "—",
      note: "Ranked by LONG conviction score across OKX USDT perpetuals",
    },
    {
      label: "API Status",
      value: apiLive ? "Online" : "Offline",
      note: apiLive
        ? "FastAPI + scoring engine connected via Cloudflare Tunnel"
        : "AUTOBOT_API_URL 환경변수를 Vercel에 설정하세요.",
    },
    {
      label: "Score Refresh",
      value: "60s",
      note: "Leaderboard revalidates every 60 seconds via Next.js ISR",
    },
    {
      label: "Strategy",
      value: "v2",
      note: "Trend · Momentum · Volume · Volatility · Sentiment · Macro",
    },
  ];

  return (
    <main className="page-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">003 Autobot</p>
          <h1>OKX long and short conviction, backtests, and reporting.</h1>
          <p className="lede">
            Live scoring from{" "}
            {symbolCount > 0 ? `${symbolCount} symbols` : "OKX perpetuals"} —
            powered by a 6-factor engine on your Mac, streamed via Cloudflare
            Tunnel.
          </p>
        </div>

        <div className="hero-panel">
          <div className="signal-pill">
            <span>API</span>
            <strong
              style={{ color: apiLive ? "var(--long)" : "var(--short)" }}
            >
              {apiLive ? "● Live" : "○ Offline"}
            </strong>
          </div>
          <div className="hero-grid">
            {reportCards.map((card) => (
              <article className="metric-card" key={card.label}>
                <span>{card.label}</span>
                <strong>{card.value}</strong>
                <p>{card.note}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="content-grid">
        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Leaderboard</p>
              <h2>Long / Short Opportunity Board</h2>
            </div>
            <span className="badge">
              {symbolCount > 0 ? "Live Data" : "No Data"}
            </span>
          </div>

          {leaderboard.length === 0 ? (
            <p style={{ color: "var(--muted)", padding: "24px 0" }}>
              {API_URL
                ? "API에 연결됐지만 데이터가 없습니다. 봇이 실행 중인지 확인하세요."
                : "AUTOBOT_API_URL 환경변수를 Vercel에 설정하면 실시간 데이터가 표시됩니다."}
            </p>
          ) : (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Symbol</th>
                    <th>Long</th>
                    <th>Short</th>
                    <th>Price</th>
                    <th>24h</th>
                    <th>Signal</th>
                  </tr>
                </thead>
                <tbody>
                  {leaderboard.map((row) => (
                    <tr key={row.symbol}>
                      <td style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
                        {row.rank}
                      </td>
                      <td style={{ fontWeight: 600 }}>
                        {row.symbol.replace("-USDT-SWAP", "")}
                      </td>
                      <td className="score long">{fmtScore(row.long_score)}</td>
                      <td className="score short">
                        {fmtScore(row.short_score)}
                      </td>
                      <td style={{ color: "var(--muted)" }}>
                        {fmtPrice(row.last_price)}
                      </td>
                      <td
                        style={{
                          color:
                            row.change_24h != null && row.change_24h >= 0
                              ? "var(--long)"
                              : "var(--short)",
                          fontSize: "0.9rem",
                        }}
                      >
                        {fmtChange(row.change_24h)}
                      </td>
                      <td>
                        <span
                          className="badge"
                          style={{
                            fontSize: "0.75rem",
                            padding: "4px 10px",
                            background:
                              row.direction === "LONG"
                                ? "rgba(17,122,79,0.1)"
                                : row.direction === "SHORT"
                                  ? "rgba(162,65,27,0.1)"
                                  : undefined,
                            color:
                              row.direction === "LONG"
                                ? "var(--long)"
                                : row.direction === "SHORT"
                                  ? "var(--short)"
                                  : "var(--muted)",
                          }}
                        >
                          {directionLabel(row.direction)} · {row.signal_strength}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Setup</p>
              <h2>연결 체크리스트</h2>
            </div>
          </div>
          <ol className="roadmap">
            <li>
              <code>start_tunnel.command</code> 더블클릭 → 터널 URL 복사
            </li>
            <li>
              Vercel → Settings → Environment Variables →{" "}
              <code>AUTOBOT_API_URL</code> 붙여넣기
            </li>
            <li>
              <code>AUTOBOT_API_KEY</code> 를 .env.local의{" "}
              <code>AUTOBOT_API_API_KEY</code> 값과 동일하게 설정
            </li>
            <li>Vercel에서 Redeploy 실행</li>
            <li>
              Mac 재시작마다 URL이 바뀜 →{" "}
              <code>setup_tunnel_autostart.sh</code> 로 자동시작 설정 권장
            </li>
          </ol>
        </article>
      </section>
    </main>
  );
}
