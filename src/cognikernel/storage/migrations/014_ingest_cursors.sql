-- Migration 014: ingest cursors for delta extraction (Sprint I / I2)
--
-- Tracks the JSONL high-water mark per (project, session) so Stop hook
-- extractions process only new turns rather than the full growing transcript.
-- Converts O(n^2) extraction cost → O(n) across a session's lifetime.
--
-- anchor_sha256: SHA256 of the ANCHOR_LINES lines immediately before the
--   high-water mark (used to detect compaction / file rewrite). If the new
--   JSONL content at those positions doesn't match, we fall back to full
--   extraction — never skip memory to save CPU.
CREATE TABLE IF NOT EXISTS ingest_cursors (
    project_id    TEXT    NOT NULL,
    session_id    TEXT    NOT NULL,
    last_line_count INTEGER NOT NULL DEFAULT 0,
    anchor_sha256 TEXT    NOT NULL DEFAULT '',
    updated_at    INTEGER NOT NULL,
    PRIMARY KEY (project_id, session_id)
);
