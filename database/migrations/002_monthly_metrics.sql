-- ============================================================
-- 월간 성과 메트릭 및 포트폴리오 이정표
-- ============================================================

-- 월간 성과 집계 (CAPR 목표 추적)
CREATE TABLE IF NOT EXISTS monthly_metrics (
    month         TEXT PRIMARY KEY,      -- 'YYYY-MM' (KST 기준)
    start_equity  REAL,                  -- 월초 자산
    end_equity    REAL,                  -- 월말(또는 현재) 자산
    pnl_usdt      REAL    DEFAULT 0,     -- 월간 실현 P&L 합계
    trade_count   INTEGER DEFAULT 0,
    win_count     INTEGER DEFAULT 0,
    best_day_pnl  REAL,
    worst_day_pnl REAL,
    target_pct    REAL,                  -- 해당 월의 목표 수익률 (설정값 스냅샷)
    actual_pct    REAL    DEFAULT 0,     -- 실제 수익률 = (end_equity - start_equity) / start_equity
    goal_achieved INTEGER DEFAULT 0,     -- 1 = 목표 달성
    updated_at    TEXT
);

-- 포트폴리오 이정표 (누적 수익률 기준 milestone 알림용)
CREATE TABLE IF NOT EXISTS portfolio_milestones (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    milestone_pct  REAL    NOT NULL,     -- 누적 수익률 기준 (예: 10.0 = +10%)
    equity_at_hit  REAL,
    achieved_at    TEXT    DEFAULT (datetime('now')),
    notified       INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_monthly_metrics_month ON monthly_metrics(month);
CREATE INDEX IF NOT EXISTS idx_milestones_pct ON portfolio_milestones(milestone_pct);
