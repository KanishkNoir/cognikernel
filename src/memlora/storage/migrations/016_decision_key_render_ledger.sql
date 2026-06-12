-- Migration 016: decision keys + render ledger (Sprint J / J2.1)
--
-- decision_key: the normalized topic axis a choice-family event is ABOUT
-- ("alias default", "counter limit rate"), derived at extraction time by
-- extraction/decision_key.py and backfilled lazily for pre-016 rows (NULL =
-- not yet derived; '' = derivation found no key). The projection groups
-- same-key DECISION/CONSTRAINT_HARD/CONSTRAINT_SOFT events and renders the
-- latest highest-authority value as canonical (latest-wins READ semantics) —
-- a structural currency guarantee independent of pairwise supersession.
-- Read-time only and reversible: no event is archived or superseded by this.
--
-- render_ledger: which events were verbatim exposed to which session, per
-- channel (block = session-start context block, ck1 = per-prompt injection,
-- recall = MCP pull [schema-ready; not written until MCP calls carry a
-- session identity]). Powers CK-1's "not already in context" redundancy
-- filter and exposure auditing. Observability state, never load-bearing:
-- an empty ledger only risks re-injecting something already shown.
--
-- Plain SQL only — FTS5 (J1.1) is deliberately NOT here; virtual-table
-- support varies by SQLite build and lives in application-level ensure_fts.
ALTER TABLE events ADD COLUMN decision_key TEXT;

CREATE INDEX IF NOT EXISTS idx_events_decision_key
    ON events (project_id, decision_key)
    WHERE decision_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS render_ledger (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id  TEXT    NOT NULL,
    session_id  TEXT    NOT NULL,
    event_id    INTEGER NOT NULL,
    channel     TEXT    NOT NULL CHECK (channel IN ('block', 'ck1', 'recall')),
    rendered_at INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_render_ledger_dedup
    ON render_ledger (project_id, session_id, event_id, channel);
