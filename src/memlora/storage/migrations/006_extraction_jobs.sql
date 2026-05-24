-- Migration 006: staged extraction jobs and acknowledgements.
-- Adds Redis/RabbitMQ-inspired lifecycle visibility without external workers.

CREATE TABLE IF NOT EXISTS extraction_jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id      TEXT    NOT NULL,
    session_id      TEXT    NOT NULL,
    evidence_id     INTEGER NOT NULL REFERENCES raw_evidence(id) ON DELETE CASCADE,
    trace_id        TEXT    NOT NULL,
    job_category    TEXT    NOT NULL CHECK (job_category IN (
                         'extract.transcript',
                         'extract.git',
                         'symbols.parse',
                         'projection.rebuild',
                         'hydrate.render',
                         'lookup.answer'
                     )),
    stage           TEXT    NOT NULL CHECK (stage IN (
                         'OBSERVED',
                         'PARSED',
                         'CLASSIFIED',
                         'MERGED',
                         'PROJECTED',
                         'INJECTED',
                         'COMPLETED'
                     )),
    state           TEXT    NOT NULL CHECK (state IN (
                         'queued',
                         'claimed',
                         'running',
                         'partial',
                         'retryable_failure',
                         'dead_lettered',
                         'skipped_policy',
                         'superseded_job',
                         'completed'
                     )),
    failure_class   TEXT CHECK (failure_class IN (
                         'TRANSIENT',
                         'POISON_INPUT',
                         'SCHEMA_MISMATCH',
                         'EXTRACTOR_BUG',
                         'IO_MISSING',
                         'TIMEOUT'
                     )),
    last_error      TEXT,
    claimed_by      TEXT,
    claimed_at      INTEGER,
    attempts        INTEGER NOT NULL DEFAULT 0,
    max_attempts    INTEGER NOT NULL DEFAULT 3,
    soft_timeout_ms INTEGER NOT NULL DEFAULT 60000,
    hard_timeout_ms INTEGER NOT NULL DEFAULT 120000,
    created_at      INTEGER NOT NULL,
    updated_at      INTEGER NOT NULL,
    UNIQUE (project_id, session_id, evidence_id, job_category)
);

CREATE INDEX IF NOT EXISTS idx_extraction_jobs_state_stage
    ON extraction_jobs (project_id, state, stage, updated_at);

CREATE INDEX IF NOT EXISTS idx_extraction_jobs_trace
    ON extraction_jobs (trace_id);

CREATE TABLE IF NOT EXISTS extraction_job_acks (
    job_id       INTEGER NOT NULL REFERENCES extraction_jobs(id) ON DELETE CASCADE,
    stage        TEXT    NOT NULL,
    completed_at INTEGER NOT NULL,
    output_ref   TEXT,
    PRIMARY KEY (job_id, stage)
);

CREATE INDEX IF NOT EXISTS idx_extraction_job_acks_job
    ON extraction_job_acks (job_id, completed_at);
