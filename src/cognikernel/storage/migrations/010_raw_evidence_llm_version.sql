-- Migration 010: track LLM extractor version on raw_evidence.
-- '' = never enriched; 'llm-v1' (etc.) = enriched with that prompt/schema.
-- Bumping the version invalidates prior enrichments without dropping data.

ALTER TABLE raw_evidence ADD COLUMN llm_extractor_version TEXT NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_raw_evidence_llm_version
    ON raw_evidence (project_id, llm_extractor_version);
