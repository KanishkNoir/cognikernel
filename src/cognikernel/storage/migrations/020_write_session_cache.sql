-- Migration 020: per-session write cache — real Write/Edit/MultiEdit touches.
-- Populated by PostToolUse on every successful Write/Edit/MultiEdit.
-- Ground truth for hot_paths and the ranking backtest, replacing the
-- COMPONENT_STATUS transcript-mention proxy (see design doc §9.6 / #30).

CREATE TABLE IF NOT EXISTS write_session_cache (
    project_id         TEXT    NOT NULL,
    session_id         TEXT    NOT NULL,
    file_path          TEXT    NOT NULL,        -- canonical relative path
    first_write_at      INTEGER NOT NULL,       -- epoch ms; set on first PostToolUse write
    last_write_at       INTEGER NOT NULL,       -- epoch ms; updated on every subsequent write
    write_count         INTEGER NOT NULL DEFAULT 1,
    last_write_action   TEXT    NOT NULL DEFAULT 'Write'
                        CHECK (last_write_action IN ('Write', 'Edit', 'MultiEdit')),
    PRIMARY KEY (project_id, session_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_write_cache_session
    ON write_session_cache (project_id, session_id);

CREATE INDEX IF NOT EXISTS idx_write_cache_recency
    ON write_session_cache (project_id, last_write_at);
