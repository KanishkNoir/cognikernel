-- Migration 004: grep result cache for deduplicating redundant grep calls
CREATE TABLE IF NOT EXISTS grep_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT    NOT NULL,
    cache_key   TEXT    NOT NULL,
    pattern     TEXT    NOT NULL,
    path_filter TEXT    NOT NULL DEFAULT '',
    glob_filter TEXT    NOT NULL DEFAULT '',
    result_text TEXT    NOT NULL,
    cached_at   INTEGER NOT NULL,
    hit_count   INTEGER NOT NULL DEFAULT 0
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_grep_cache_unique
    ON grep_cache (project_id, cache_key);

CREATE INDEX IF NOT EXISTS idx_grep_cache_project
    ON grep_cache (project_id, cached_at DESC);
