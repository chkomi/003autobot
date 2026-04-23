-- ============================================================
-- OKX 자동매매봇 초기 DB 스키마
-- ============================================================

-- 거래 기록
CREATE TABLE IF NOT EXISTS trades (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id         TEXT    UNIQUE NOT NULL,
    symbol           TEXT    NOT NULL,
    direction        TEXT    NOT NULL CHECK(direction IN ('LONG', 'SHORT')),
    status           TEXT    NOT NULL CHECK(status IN ('OPEN', 'CLOSED', 'CANCELLED')),
    entry_price      REAL,
    exit_price       REAL,
    quantity         REAL    NOT NULL,
    leverage         INTEGER NOT NULL,
    stop_loss        REAL,
    take_profit_1    REAL,
    take_profit_2    REAL,
    pnl_usdt         REAL,
    pnl_pct          REAL,
    entry_time       TEXT,
    exit_time        TEXT,
    -- 청산 이유: TP1/TP2/SL/TRAILING/TIME/MANUAL/HALT
    exit_reason      TEXT,
    signal_confidence REAL,
    atr_at_entry     REAL,
    -- 알고 주문 ID (SL/TP 취소용)
    sl_algo_id       TEXT,
    tp_algo_id       TEXT,
    created_at       TEXT DEFAULT (datetime('now'))
);

-- 캔들 캐시 (REST 조회 결과 저장 → 재시작 시 재활용)
CREATE TABLE IF NOT EXISTS candle_cache (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open      REAL,
    high      REAL,
    low       REAL,
    close     REAL,
    volume    REAL,
    UNIQUE(symbol, timeframe, timestamp)
);

-- 일일 P&L 집계
CREATE TABLE IF NOT EXISTS daily_pnl (
    date        TEXT PRIMARY KEY,
    pnl_usdt    REAL DEFAULT 0,
    trade_count INTEGER DEFAULT 0,
    win_count   INTEGER DEFAULT 0,
    peak_equity REAL,
    min_equity  REAL,
    updated_at  TEXT
);

-- 계좌 자산 스냅샷 (드로다운 계산용)
CREATE TABLE IF NOT EXISTS equity_snapshots (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    equity     REAL NOT NULL,
    free       REAL,
    used       REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

-- 봇 이벤트 로그 (경고, 정지, 시그널 기록)
CREATE TABLE IF NOT EXISTS bot_events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,  -- SIGNAL/ORDER/HALT/ALERT/INFO
    level      TEXT NOT NULL,  -- INFO/WARNING/ERROR/CRITICAL
    message    TEXT,
    metadata   TEXT,           -- JSON blob
    created_at TEXT DEFAULT (datetime('now'))
);

-- ── 인덱스 ──────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_trades_status     ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_direction  ON trades(direction);
CREATE INDEX IF NOT EXISTS idx_candle_lookup     ON candle_cache(symbol, timeframe, timestamp);
CREATE INDEX IF NOT EXISTS idx_equity_time       ON equity_snapshots(created_at);
CREATE INDEX IF NOT EXISTS idx_events_type       ON bot_events(event_type, created_at);
