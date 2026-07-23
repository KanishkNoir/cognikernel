-- Migration 018: widen raw_evidence.source_type CHECK to include 'codex_rollout'
-- (Sprint L / L2 — cross-platform capture).
--
-- Codex CLI sessions are captured as rollout JSONL stored as evidence tagged
-- source_type='codex_rollout', which the extraction worker dispatches through the
-- Codex->prose converter. SQLite cannot ALTER a CHECK, so the table is rebuilt.
--
-- raw_evidence is a PARENT of several FKs (event_provenance, extraction_jobs,
-- enrichment_jobs, events.evidence_id). With foreign_keys ON, DROP TABLE on a
-- parent performs an implicit DELETE that cascades into children — wiping real
-- data. The migration runner therefore applies the numbered chain with
-- foreign_keys temporarily OFF (a PRAGMA is a no-op inside the migration's own
-- transaction, so it is toggled in the runner, not here). The RENAME restores the
-- original table name, so the children's "REFERENCES raw_evidence(id)" stay valid.

CREATE TABLE raw_evidence_new (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id          TEXT    NOT NULL,
    session_id          TEXT    NOT NULL,
    source_type         TEXT    NOT NULL CHECK (source_type IN (
                            'transcript',
                            'jsonl_transcript',
                            'codex_rollout',
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
    llm_extractor_version TEXT  NOT NULL DEFAULT '',
    prev_evidence_id    INTEGER,
    UNIQUE (project_id, content_sha256)
);

INSERT INTO raw_evidence_new
    (id, project_id, session_id, source_type, source_path, captured_at,
     content_sha256, content_encoding, content_blob, original_size_bytes,
     stored_size_bytes, metadata, llm_extractor_version, prev_evidence_id)
    SELECT
     id, project_id, session_id, source_type, source_path, captured_at,
     content_sha256, content_encoding, content_blob, original_size_bytes,
     stored_size_bytes, metadata, llm_extractor_version, prev_evidence_id
    FROM raw_evidence;

DROP TABLE raw_evidence;
ALTER TABLE raw_evidence_new RENAME TO raw_evidence;

CREATE INDEX IF NOT EXISTS idx_raw_evidence_project_session
    ON raw_evidence (project_id, session_id, captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_evidence_llm_version
    ON raw_evidence (project_id, llm_extractor_version);
CREATE INDEX IF NOT EXISTS idx_raw_evidence_chain
    ON raw_evidence (prev_evidence_id)
    WHERE prev_evidence_id IS NOT NULL;
