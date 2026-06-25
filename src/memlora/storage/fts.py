"""SQLite FTS5 lexical index over event descriptions (J1.1).

The lexical axis of hybrid retrieval. A plain (self-contained) FTS5 table is
used deliberately:
  - external-content is out because the indexed text lives inside the payload
    JSON, not in addressable columns;
  - contentless is out because deleting rows requires SQLite >= 3.43 and the
    runtime floor is whatever the user's Python ships (3.42 observed locally).
Duplicating the indexed text (~600 bytes/event) is the lightweight choice at
local scale.

Availability is a property of the user's SQLite build, so everything here is
fail-open: the table and triggers are created application-level (never in a
numbered migration — a build without FTS5 must degrade, not brick every hook),
and callers treat an absent index as "lexical axis unavailable".

Liveness (archived / superseded) is filtered at query time by joining `events`
— index rows are never deleted, consistent with the lossless principle. Sync is
an AFTER INSERT trigger (+ a defensive AFTER UPDATE OF payload): event payloads
are immutable after insert and events are never deleted, so this is complete.
"""
from __future__ import annotations

import re
import sqlite3
from typing import Any

_META_KEY = "fts_enabled"

# tokenchars '-_' keeps identifier-shaped terms (`relay-default`, `_MAX_ATTEMPTS`)
# as single tokens — the measured recall-miss class. '.' is deliberately NOT a
# token char: it would glue sentence-final periods onto words.
_DDL_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
    description, subject,
    tokenize = "unicode61 remove_diacritics 2 tokenchars '-_'"
)
"""

_INDEXED_COLS = (
    "COALESCE(json_extract(payload,'$.description'),''), "
    "COALESCE(json_extract(payload,'$.subject'),"
    "         json_extract(payload,'$.triple.subject'),'')"
)

_DDL_TRG_INSERT = f"""
CREATE TRIGGER IF NOT EXISTS trg_events_fts_ai AFTER INSERT ON events BEGIN
    INSERT INTO events_fts(rowid, description, subject)
    SELECT new.id, {_INDEXED_COLS.replace('payload', 'new.payload')};
END
"""

# Defensive only: payload is immutable post-insert today. If that ever changes,
# the index follows instead of silently drifting stale.
_DDL_TRG_UPDATE = f"""
CREATE TRIGGER IF NOT EXISTS trg_events_fts_au AFTER UPDATE OF payload ON events BEGIN
    DELETE FROM events_fts WHERE rowid = old.id;
    INSERT INTO events_fts(rowid, description, subject)
    SELECT new.id, {_INDEXED_COLS.replace('payload', 'new.payload')};
END
"""

_SQL_BACKFILL = f"""
INSERT INTO events_fts(rowid, description, subject)
SELECT id, {_INDEXED_COLS}
FROM events WHERE id NOT IN (SELECT rowid FROM events_fts)
"""

# MATCH-query tokenizer. Mirrors the FTS tokenizer's identifier handling
# (hyphens/underscores stay inside tokens) — supersede.normalize_for_overlap is
# NOT reused here because it strips hyphens, which would split the identifiers
# this index exists to match.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-]*")
_MAX_QUERY_TOKENS = 12


def fts_available(conn: sqlite3.Connection) -> bool:
    """True if this SQLite build can create FTS5 tables."""
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.__fts_probe USING fts5(x)")
        conn.execute("DROP TABLE temp.__fts_probe")
        return True
    except sqlite3.OperationalError:
        return False


def fts_enabled(conn: sqlite3.Connection) -> bool:
    """Cheap read of the persisted availability flag (no probe, no writes)."""
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (_META_KEY,)
        ).fetchone()
        return row is not None and row[0] == "1"
    except Exception:
        return False


def ensure_fts(conn: sqlite3.Connection) -> bool:
    """Idempotently create the FTS index, triggers, and backfill. Fail-open.

    Returns True when the index is usable. A build without FTS5 persists '0'
    (re-probed on later calls, so a Python upgrade self-heals); a *transient*
    failure (e.g. lock contention with a merge) persists nothing and simply
    retries on the next run_migrations.
    """
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key = ?", (_META_KEY,)
        ).fetchone()
        if row is not None and row[0] == "1":
            return True  # fast path: one SELECT per call

        if not fts_available(conn):
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, '0')",
                (_META_KEY,),
            )
            conn.commit()
            return False

        conn.execute(_DDL_TABLE)
        conn.execute(_DDL_TRG_INSERT)
        conn.execute(_DDL_TRG_UPDATE)
        conn.execute(_SQL_BACKFILL)
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, '1')",
            (_META_KEY,),
        )
        conn.commit()
        return True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return False


def build_match_query(query_text: str) -> str:
    """Sanitize free text into a safe FTS5 MATCH expression.

    Raw user text is a syntax-error vector (quotes, parens, operators), so each
    token is double-quoted (a literal string in FTS5 syntax) and tokens are
    OR-joined. Stopword-only or empty input yields '' (caller returns no hits).
    """
    from memlora.delta.supersede import STOPWORDS

    seen: list[str] = []
    for tok in _TOKEN_RE.findall(query_text.lower()):
        if len(tok) <= 2 and not tok.isdigit():
            continue
        if tok in STOPWORDS or tok in seen:
            continue
        seen.append(tok)
        if len(seen) >= _MAX_QUERY_TOKENS:
            break
    return " OR ".join(f'"{t}"' for t in seen)


def bm25_search(
    conn: sqlite3.Connection,
    project_id: str,
    query_text: str,
    n: int = 20,
) -> list[dict[str, Any]]:
    """BM25-ranked active events for `query_text` (best first).

    Returns [] when the index is unavailable or the query sanitizes to nothing
    — callers treat that as "lexical axis absent", not an error. SQLite's
    bm25() is lower-is-better; results are returned best-first and carry the
    raw bm25 value for diagnostics (rank position is the load-bearing output).
    """
    import json

    if not fts_enabled(conn):
        return []
    match = build_match_query(query_text)
    if not match:
        return []
    try:
        rows = conn.execute(
            """
            SELECT e.id, e.event_type, e.payload, bm25(events_fts) AS b
            FROM events_fts
            JOIN events e ON e.id = events_fts.rowid
            WHERE events_fts MATCH ?
              AND e.project_id    = ?
              AND e.archived      = 0
              AND e.superseded_by IS NULL
            ORDER BY b
            LIMIT ?
            """,
            (match, project_id, n),
        ).fetchall()
    except sqlite3.OperationalError:
        return []  # malformed MATCH despite sanitization — degrade, don't raise
    out: list[dict[str, Any]] = []
    for r in rows:
        payload = json.loads(r["payload"])
        out.append(
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "description": payload.get("description", ""),
                "subject": payload.get("subject", ""),
                "bm25": round(float(r["b"]), 4),
            }
        )
    return out


def prohibition_search(
    conn: sqlite3.Connection,
    project_id: str,
    query_text: str,
    n: int = 8,
) -> list[dict[str, Any]]:
    """BM25-ranked active *prohibitions* for `query_text` (best first).

    The K1 retrieval primitive behind PreToolUse JIT surfacing. Identical to
    `bm25_search` but the candidate pool is type-restricted to the binding
    event types — graveyard ("do not retry") entries and hard constraints — so
    a prohibition is never crowded out of the top-n by ordinary decisions when
    the query is an edit diff. The query_text is typically the new code being
    written, not a prompt; `build_match_query` keeps identifier-shaped terms
    (`in-process`, `rate-limit`) intact.

    Returns [] when the index is unavailable or the query sanitizes to nothing
    — callers treat that as "no prohibition evidence", not an error. Carries
    `authority` (load-bearing: the surfacing gate weighs user_stated highest).
    """
    import json

    from memlora.storage.sections import GRAVEYARD_TYPES, HARD_TYPES

    if not fts_enabled(conn):
        return []
    match = build_match_query(query_text)
    if not match:
        return []
    types = tuple(sorted(GRAVEYARD_TYPES | HARD_TYPES))
    placeholders = ",".join("?" for _ in types)
    try:
        rows = conn.execute(
            f"""
            SELECT e.id, e.event_type, e.payload, bm25(events_fts) AS b
            FROM events_fts
            JOIN events e ON e.id = events_fts.rowid
            WHERE events_fts MATCH ?
              AND e.project_id    = ?
              AND e.archived      = 0
              AND e.superseded_by IS NULL
              AND e.event_type IN ({placeholders})
            ORDER BY b
            LIMIT ?
            """,
            (match, project_id, *types, n),
        ).fetchall()
    except sqlite3.OperationalError:
        return []  # malformed MATCH despite sanitization — degrade, don't raise
    out: list[dict[str, Any]] = []
    for r in rows:
        payload = json.loads(r["payload"])
        out.append(
            {
                "id": r["id"],
                "event_type": r["event_type"],
                "description": payload.get("description", ""),
                "subject": payload.get("subject", ""),
                "rationale": payload.get("rationale", payload.get("reason", "")),
                "authority": payload.get("authority", ""),
                "bm25": round(float(r["b"]), 4),
            }
        )
    return out
