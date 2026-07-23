-- Migration 005: raw evidence and event provenance.
-- Stores compressed source evidence before derived events are emitted.

CREATE TABLE IF NOT EXISTS raw_evidence (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id          TEXT    NOT NULL,
    session_id          TEXT    NOT NULL,
    source_type         TEXT    NOT NULL CHECK (source_type IN (
                            'transcript',
                            'jsonl_transcript',
                            'git_diff',
                            'tool_payload',
                            'manual'
                        )),
    source_path         TEXT    NOT NULL DEFAULT '',
    captured_at         INTEGER NOT NULL,
    content_sha256      TEXT    NOT NULL,
    content_encoding    TEXT    NOT NULL CHECK (content_encoding IN ('zlib')),
    content_blob        BLOB    NOT NULL,
    original_size_bytes INTEGER NOT NULL,
    stored_size_bytes   INTEGER NOT NULL,
    metadata            TEXT    NOT NULL DEFAULT '{}',
    UNIQUE (project_id, content_sha256)
);

CREATE INDEX IF NOT EXISTS idx_raw_evidence_project_session
    ON raw_evidence (project_id, session_id, captured_at DESC);

CREATE TABLE IF NOT EXISTS event_provenance (
    event_id             INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    raw_evidence_id      INTEGER NOT NULL REFERENCES raw_evidence(id) ON DELETE CASCADE,
    extractor_version    TEXT    NOT NULL,
    matched_phrase       TEXT,
    sentence_index       INTEGER,
    window_start         INTEGER,
    window_end           INTEGER,
    confidence           REAL,
    transformation_notes TEXT,
    created_at           INTEGER NOT NULL,
    PRIMARY KEY (event_id, raw_evidence_id)
);

CREATE INDEX IF NOT EXISTS idx_event_provenance_evidence
    ON event_provenance (raw_evidence_id);

ALTER TABLE events ADD COLUMN evidence_id INTEGER REFERENCES raw_evidence(id);

CREATE INDEX IF NOT EXISTS idx_events_evidence
    ON events (project_id, evidence_id);
