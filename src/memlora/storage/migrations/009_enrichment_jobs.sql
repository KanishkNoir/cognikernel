-- Migration 009: LLM enrichment job lifecycle.
-- Separate from extraction_jobs because the lifecycle is different (no staged
-- pipeline) and extraction_jobs.job_category has a CHECK constraint that can't
-- be extended in-place. Versioned extractor IDs let us cleanly re-run after
-- prompt or schema changes.

CREATE TABLE IF NOT EXISTS enrichment_jobs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id        TEXT    NOT NULL,
    evidence_id       INTEGER NOT NULL REFERENCES raw_evidence(id) ON DELETE CASCADE,
    enrichment_kind   TEXT    NOT NULL CHECK (enrichment_kind IN (
                          'llm_decision_extraction',
                          'llm_constraint_inference'
                      )),
    extractor_version TEXT    NOT NULL,
    state             TEXT    NOT NULL DEFAULT 'queued'
                      CHECK (state IN (
                          'queued',
                          'claimed',
                          'completed',
                          'partial',
                          'failed',
                          'skipped'
                      )),
    error             TEXT    NOT NULL DEFAULT '',
    queued_at         INTEGER NOT NULL,
    completed_at      INTEGER NOT NULL DEFAULT 0,
    UNIQUE (project_id, evidence_id, enrichment_kind, extractor_version)
);

CREATE INDEX IF NOT EXISTS idx_enrichment_jobs_pending
    ON enrichment_jobs (project_id, state, queued_at);

CREATE INDEX IF NOT EXISTS idx_enrichment_jobs_evidence
    ON enrichment_jobs (evidence_id);
