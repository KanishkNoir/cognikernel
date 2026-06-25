-- Migration 015: chained delta evidence (Sprint I / I3)
--
-- Adds prev_evidence_id to raw_evidence so each Stop-hook firing can store
-- only the new JSONL lines (delta) rather than the full growing transcript.
-- Reconstruction: follow prev_evidence_id chain from root to leaf, concatenate
-- all content_blob bytes in order -> byte-exact full transcript.
--
-- NULL prev_evidence_id = chain root (full content stored, first firing or
-- compaction-fallback). Application enforces chain integrity; no FK constraint
-- because SQLite ALTER TABLE cannot add FK references.
--
-- Also adds last_evidence_id to ingest_cursors so session_end can continue
-- the chain on each firing without an extra query.
ALTER TABLE raw_evidence ADD COLUMN prev_evidence_id INTEGER;

CREATE INDEX IF NOT EXISTS idx_raw_evidence_chain
    ON raw_evidence (prev_evidence_id)
    WHERE prev_evidence_id IS NOT NULL;

ALTER TABLE ingest_cursors ADD COLUMN last_evidence_id INTEGER;
