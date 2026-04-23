const leaderboard = [
  { symbol: "BTC-USDT-SWAP", longScore: 84, shortScore: 26, winRate: "61%", status: "Momentum Long" },
  { symbol: "ETH-USDT-SWAP", longScore: 73, shortScore: 35, winRate: "58%", status: "Trend Confirmed" },
  { symbol: "SOL-USDT-SWAP", longScore: 41, shortScore: 77, winRate: "55%", status: "Short Bias" },
  { symbol: "SUI-USDT-SWAP", longScore: 67, shortScore: 48, winRate: "53%", status: "Watchlist" },
];

const reportCards = [
  { label: "Realtime Universe", value: "100+", note: "USDT perpetual instruments ranked continuously" },
  { label: "Backtest Jobs", value: "24", note: "Queued, running, and completed jobs in one control plane" },
  { label: "Daily Review", value: "07:30", note: "Automated portfolio and strategy recap delivery" },
  { label: "Risk Checks", value: "9", note: "Exposure, drawdown, stale feed, and ops health validations" },
];

export default function Home() {
  return (
    <main className="page-shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">003 Autobot</p>
          <h1>OKX long and short conviction, backtests, and operator-grade reporting in one workspace.</h1>
          <p className="lede">
            The current web app is scaffolded for Vercel and ready to be connected to the Python scoring API,
            backtest worker, and reporting pipeline.
          </p>
        </div>

        <div className="hero-panel">
          <div className="signal-pill">
            <span>Universe Status</span>
            <strong>Live Snapshot Ready</strong>
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
            <span className="badge">Mock Data</span>
          </div>

          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Symbol</th>
                  <th>Long</th>
                  <th>Short</th>
                  <th>Backtest Win Rate</th>
                  <th>Signal</th>
                </tr>
              </thead>
              <tbody>
                {leaderboard.map((row) => (
                  <tr key={row.symbol}>
                    <td>{row.symbol}</td>
                    <td className="score long">{row.longScore}</td>
                    <td className="score short">{row.shortScore}</td>
                    <td>{row.winRate}</td>
                    <td>{row.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <article className="panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">Build Queue</p>
              <h2>Implementation Priorities</h2>
            </div>
          </div>

          <ol className="roadmap">
            <li>Expose Python score snapshots through versioned API endpoints.</li>
            <li>Persist score history, reports, and backtest jobs in PostgreSQL.</li>
            <li>Stream cached snapshots to the Vercel frontend on a short interval.</li>
            <li>Run long OKX ingestion and multi-symbol backtests in a dedicated worker.</li>
            <li>Replace mock data with live leaderboard, symbol detail, and report views.</li>
          </ol>
        </article>
      </section>
    </main>
  );
}
