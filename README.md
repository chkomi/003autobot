# 003 Autobot

OKX market-data and trading-analysis workspace for:

- long/short opportunity scoring by instrument
- symbol-level backtesting
- portfolio and strategy reports
- risk checks, monitoring, and performance evaluation

This repository currently contains the Python trading core and an initial Vercel-ready web app scaffold.

## Repository Layout

- `main.py`: bot entrypoint
- `dashboard/`: current FastAPI dashboard
- `backtest/`: backtesting engine and analytics
- `strategy/`: scoring and signal logic
- `database/`: persistence and migrations
- `web/`: Next.js frontend for Vercel
- `docs/`: architecture and delivery plan

## Recommended Architecture

- `web/` on Vercel for the product UI
- Python API service for score queries, reports, and job control
- Python worker for OKX ingestion and long-running backtests
- PostgreSQL for application data
- Redis for cache, streaming snapshots, and job state
- Blob storage for exported reports and artifacts

See [architecture.md](/Users/yun/Documents/Business/003. Autobot/docs/architecture.md) and [mvp-roadmap.md](/Users/yun/Documents/Business/003. Autobot/docs/mvp-roadmap.md) for the implementation plan.

## Immediate Next Steps

1. Turn the current score and backtest logic into explicit API endpoints.
2. Move runtime state from SQLite-only storage to PostgreSQL plus Redis.
3. Connect the `web/` app to live API data instead of mock data.
4. Deploy the web app to Vercel and the Python services to a separate runtime.

