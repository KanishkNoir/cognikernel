-- Per-rule admission-gate counters. One row per (project, session, rule);
-- repeat hits increment `count` so the table stays small regardless of how
-- noisy a session is. Feeds `cognikernel doctor` and the longitudinal
-- real-world prevention-rate claim.
CREATE TABLE IF NOT EXISTS quality_telemetry (
    project_id  TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    rule_id     TEXT    NOT NULL,
    count       INTEGER NOT NULL DEFAULT 0,
    updated_at  INTEGER NOT NULL,
    PRIMARY KEY (project_id, session_id, rule_id)
);

CREATE INDEX IF NOT EXISTS idx_quality_telemetry_project
    ON quality_telemetry (project_id, rule_id);
