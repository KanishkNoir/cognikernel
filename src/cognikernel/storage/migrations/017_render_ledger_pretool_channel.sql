-- Migration 017: widen render_ledger channel CHECK to include 'pretool' (Sprint K / K2)
--
-- PreToolUse JIT surfacing (the "bind at the action point" fix) records which
-- prohibitions it surfaced on a new 'pretool' channel, so the same prohibition
-- is not re-surfaced repeatedly in a session and the exposure is auditable.
-- rendered_event_ids() is already channel-agnostic; only the CHECK constraint
-- needs widening. SQLite cannot ALTER a CHECK, so rebuild the (small,
-- session-scoped) table preserving existing rows.
ALTER TABLE render_ledger RENAME TO render_ledger_old;

CREATE TABLE render_ledger (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    event_id    INTEGER NOT NULL,
    channel     TEXT    NOT NULL CHECK (channel IN ('block', 'ck1', 'recall', 'pretool')),
    rendered_at INTEGER NOT NULL
);

INSERT INTO render_ledger (id, project_id, session_id, event_id, channel, rendered_at)
    SELECT id, project_id, session_id, event_id, channel, rendered_at
    FROM render_ledger_old;

DROP TABLE render_ledger_old;

CREATE UNIQUE INDEX IF NOT EXISTS idx_render_ledger_dedup
    ON render_ledger (project_id, session_id, event_id, channel);
