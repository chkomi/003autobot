# Architecture

## Goal

Build a web platform that helps manage:

- real-time OKX long/short opportunity scores by symbol
- backtests and rankings
- portfolio and strategy reports
- risk checks, monitoring, and operational health
- investment performance evaluation

## Deployment Split

### 1. Vercel Web App

Use Vercel for the customer-facing frontend:

- instrument leaderboard
- symbol detail pages
- report center
- admin monitoring pages
- backtest submission UI

Recommended stack:

- Next.js App Router
- TypeScript
- server actions for simple admin flows
- polling or SSE consumption from the backend API

### 2. Python API Service

Expose application data from the existing Python core:

- `GET /api/v1/market/leaderboard`
- `GET /api/v1/market/symbols/{symbol}`
- `GET /api/v1/market/symbols/{symbol}/score`
- `POST /api/v1/backtests`
- `GET /api/v1/backtests/{jobId}`
- `GET /api/v1/reports/overview`
- `GET /api/v1/ops/health`

Responsibilities:

- read current score snapshots
- serve portfolio and symbol analytics
- control backtest jobs
- return report summaries for the web app

### 3. Worker Service

Run outside Vercel for persistent and long-running tasks:

- OKX websocket ingestion
- market snapshot refresh
- scheduled score recalculation
- multi-symbol backtests
- daily and weekly report generation

### 4. Data Stores

- PostgreSQL: symbols, scores, jobs, trades, reports, snapshots
- Redis: realtime cache, leaderboard cache, job progress, stream fan-out
- Blob storage: exported CSV, HTML, and PDF reports

## Reuse From Current Codebase

- `strategy/score_engine.py`: base scoring logic
- `dashboard/server.py`: existing FastAPI patterns and realtime concepts
- `run_full_backtest.py`: universe-wide backtest orchestration
- `backtest/`: performance analysis and validation modules
- `database/`: a useful starting point for migrations and models

## Delivery Sequence

1. Extract stable service interfaces around scoring and backtesting.
2. Add API schemas and route handlers.
3. Introduce PostgreSQL models for symbols, scores, jobs, and reports.
4. Build worker jobs for snapshot refresh and backtests.
5. Wire the Next.js frontend to the API.
6. Add auth, alerts, and operator workflows.

## Vercel Notes

Vercel is best used here for the frontend and short-lived API handlers. Persistent OKX websocket ingestion and long-running backtests should stay in a separate Python runtime.

