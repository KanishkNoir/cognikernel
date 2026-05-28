-- Migration 007: per-session read cache for the PreToolUse hook.
-- Populated by PostToolUse:Read on every successful Read.
-- Queried by PreToolUse:Read to deny re-reads of already-cited files.

CREATE TABLE IF NOT EXISTS read_session_cache (
    project_id        TEXT    NOT NULL,
    session_id        TEXT    NOT NULL,
    file_path         TEXT    NOT NULL,        -- canonical relative path
    first_read_at     INTEGER NOT NULL,        -- epoch ms; set on first PostToolUse:Read
    last_read_at      INTEGER NOT NULL,        -- epoch ms; updated on every subsequent read
    read_count        INTEGER NOT NULL DEFAULT 1,
    last_read_outcome TEXT    NOT NULL DEFAULT 'ok'
                      CHECK (last_read_outcome IN ('ok', 'body_needed_retry')),
    PRIMARY KEY (project_id, session_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_read_cache_session
    ON read_session_cache (project_id, session_id);

CREATE INDEX IF NOT EXISTS idx_read_cache_cleanup
    ON read_session_cache (first_read_at);
