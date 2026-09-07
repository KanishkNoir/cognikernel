-- Migration 021: supersession/archival transitions + commit anchor.
--
-- The store records supersession and archival as STATES (superseded_by,
-- archived) but not as TRANSITIONS, so the belief set as of an arbitrary
-- time cannot be reconstructed -- the nearest approximation, inferring the
-- time from the superseding event's own created_at, is wrong whenever
-- supersession is applied in a later backfill or a projection rebuild.
--
-- captured_at_sha anchors a claim to the commit it was captured against,
-- which is what lets drift be computed as a diff against HEAD rather than
-- inferred. supersede_reason names which gate fired (lexical, subject_key,
-- cross_encoder, cross_type_priority, decision_key) for the debugger --
-- a fixed vocabulary, not free text.
--
-- NULL is the honest value for pre-021 rows and for captures outside a git
-- work tree -- these columns are never backfilled with a guess. A guessed
-- sha or inferred timestamp would make future replay confidently wrong
-- rather than honestly bounded.
ALTER TABLE events ADD COLUMN superseded_at    INTEGER;
ALTER TABLE events ADD COLUMN archived_at      INTEGER;
ALTER TABLE events ADD COLUMN captured_at_sha  TEXT;
ALTER TABLE events ADD COLUMN supersede_reason TEXT;

CREATE INDEX IF NOT EXISTS idx_events_superseded_at
    ON events (project_id, superseded_at)
    WHERE superseded_at IS NOT NULL;
