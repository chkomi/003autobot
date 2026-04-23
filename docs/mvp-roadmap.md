# MVP Roadmap

## Phase 1

Ship the first usable operator dashboard.

Scope:

- live leaderboard with long and short scores
- symbol detail page with score breakdown
- one-click backtest submission
- backtest result table
- daily overview report
- health and risk status panel

## Phase 2

Make the analytics reliable enough for daily use.

Scope:

- scheduled report generation
- result history and comparison by strategy version
- risk and data freshness alerts
- score explanation trails
- portfolio exposure summary

## Phase 3

Expand into performance management.

Scope:

- attribution by symbol and strategy
- benchmark-relative performance
- execution quality checks
- operator notes and review workflow
- report exports

## Execution Checklist

1. Finalize API contracts.
2. Stand up PostgreSQL and Redis.
3. Convert the Python scoring logic into API responses.
4. Build backtest jobs and progress tracking.
5. Connect the Next.js UI to live endpoints.
6. Add deployment pipelines for Vercel and the Python services.

