-- ============================================================
-- 002: TP2 컬럼 추가 (기존 DB 호환)
-- SQLite는 ALTER TABLE IF NOT EXISTS 미지원이므로
-- db_manager._migrate에서 try/except로 감싸 실행된다.
-- ============================================================
ALTER TABLE trades ADD COLUMN take_profit_2 REAL;
