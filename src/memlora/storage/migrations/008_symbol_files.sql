-- Migration 008: file-level skeleton authority.
-- Replaces the implicit "status" semantics scattered across component_map payloads
-- with a single explicit freshness column per file. Tracks scan state authoritatively
-- so the injection can render truthful coverage statistics.

CREATE TABLE IF NOT EXISTS symbol_files (
    project_id           TEXT    NOT NULL,
    path                 TEXT    NOT NULL,          -- canonical relative path
    freshness            TEXT    NOT NULL DEFAULT 'fresh'
                         CHECK (freshness IN ('fresh', 'stale')),
    refreshed_at         INTEGER NOT NULL DEFAULT 0,   -- epoch ms of last symbol re-scan
    refreshed_in_session TEXT    NOT NULL DEFAULT '',  -- session that triggered the last refresh
    last_action          TEXT    NOT NULL DEFAULT '',  -- 'Write' | 'Edit' | 'scan' — sources B-2 header
    content_sha256       TEXT    NOT NULL DEFAULT '',  -- file content hash at refresh time
    scan_status          TEXT    NOT NULL DEFAULT 'pending'
                         CHECK (scan_status IN ('scanned', 'parse_error', 'ignored', 'pending')),
    symbol_count         INTEGER NOT NULL DEFAULT 0,   -- public symbols extracted (0 is valid)
    last_error           TEXT    NOT NULL DEFAULT '',  -- non-empty only when scan_status='parse_error'
    PRIMARY KEY (project_id, path)
);

CREATE INDEX IF NOT EXISTS idx_symbol_files_freshness
    ON symbol_files (project_id, freshness);

CREATE INDEX IF NOT EXISTS idx_symbol_files_refreshed
    ON symbol_files (project_id, refreshed_at DESC);
