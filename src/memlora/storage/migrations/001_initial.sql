-- Migration 001: initial schema
-- Creates all tables and indexes from scratch.
-- Uses IF NOT EXISTS throughout so this file is safe to re-run.

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version',    '0');
INSERT OR IGNORE INTO meta (key, value) VALUES ('projection_version', '0');

CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    TEXT    NOT NULL,
    session_id    TEXT    NOT NULL,
    created_at    INTEGER NOT NULL,
    event_type    TEXT    NOT NULL,
    payload       TEXT    NOT NULL,
    content_hash  TEXT    NOT NULL,
    weight        REAL    NOT NULL DEFAULT 1.0,
    mention_count INTEGER NOT NULL DEFAULT 1,
    superseded_by INTEGER,
    archived      INTEGER NOT NULL DEFAULT 0,
    CHECK (event_type IN (
        'DECISION',
        'CONSTRAINT_HARD',
        'CONSTRAINT_SOFT',
        'COMPONENT_STATUS',
        'APPROACH_ABANDONED',
        'APPROACH_ABANDONED_DO_NOT_RETRY',
        'THREAD_OPEN',
        'THREAD_CLOSE'
    ))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_events_content_hash
    ON events (project_id, content_hash);

CREATE INDEX IF NOT EXISTS idx_events_project_session
    ON events (project_id, session_id);

CREATE INDEX IF NOT EXISTS idx_events_project_type_archived
    ON events (project_id, event_type, archived);

CREATE INDEX IF NOT EXISTS idx_events_weight
    ON events (project_id, archived, weight DESC);

CREATE TABLE IF NOT EXISTS state_projections (
    project_id          TEXT    PRIMARY KEY,
    built_at            INTEGER NOT NULL,
    event_id_high_water INTEGER NOT NULL,
    hard_constraints    TEXT    NOT NULL,
    ranked_decisions    TEXT    NOT NULL,
    component_map       TEXT    NOT NULL,
    graveyard           TEXT    NOT NULL,
    active_threads      TEXT    NOT NULL,
    summary             TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS extraction_failures (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id     TEXT    NOT NULL,
    session_id     TEXT    NOT NULL,
    failed_at      INTEGER NOT NULL,
    stage          TEXT    NOT NULL,
    error_message  TEXT    NOT NULL,
    raw_input_path TEXT    NOT NULL,
    retry_count    INTEGER NOT NULL DEFAULT 0
);
