-- Migration 011: ephemeral deny-timer table for the 60-second retry escape hatch.
-- A skeleton-fresh Read is denied on first attempt; if Claude retries the same
-- Read within the retry window, the second attempt is allowed and tagged as
-- 'body_needed_retry' so the read_session_cache records the legitimate body need.

CREATE TABLE IF NOT EXISTS denied_reads (
    project_id  TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    file_path   TEXT    NOT NULL,        -- canonical relative path
    denied_at   INTEGER NOT NULL,        -- epoch ms; row replaced on each new denial
    reason      TEXT    NOT NULL,        -- 'skeleton_fresh' | future reasons
    PRIMARY KEY (project_id, session_id, file_path)
);

CREATE INDEX IF NOT EXISTS idx_denied_reads_cleanup
    ON denied_reads (denied_at);
