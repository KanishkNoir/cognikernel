-- Migration 003: API telemetry for cache-hit tracking
CREATE TABLE IF NOT EXISTS api_telemetry (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id            TEXT NOT NULL,
    session_id            TEXT NOT NULL,
    input_tokens          INTEGER NOT NULL DEFAULT 0,
    cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens     INTEGER NOT NULL DEFAULT 0,
    output_tokens         INTEGER NOT NULL DEFAULT 0,
    ingested_at           INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_api_telemetry_unique
    ON api_telemetry (project_id, session_id);

CREATE INDEX IF NOT EXISTS idx_api_telemetry_project
    ON api_telemetry (project_id, ingested_at DESC);
