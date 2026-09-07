from __future__ import annotations

import json
import logging
import sqlite3
import time

# Event + VALID_EVENT_TYPES live in the dependency-free base model module so the
# layered packages can type-hint Event without importing "up" into storage (the
# layering fix). Re-exported here for backward compatibility — existing
# `from cognikernel.storage.events import Event` call sites are unaffected.
from cognikernel.model import Event, VALID_EVENT_TYPES  # noqa: F401  (re-export)

_log = logging.getLogger("cognikernel.storage")

# Weight boost applied to an event that already exists (dedup hit).
WEIGHT_INCREMENT_ON_DEDUP: float = 0.15

# Maximum weight any event can accumulate.
MAX_EVENT_WEIGHT: float = 5.0

# Weight below which events are archived during the decay pass. Single source
# of truth — delta.decay imports this (it previously carried its own copy).
ARCHIVE_THRESHOLD: float = 0.05

# T-103 (#13): fixed vocabulary for events.supersede_reason -- for the
# debugger ("why was this superseded?"), never free text. "semantic" covers
# the pure-cosine axis in delta.supersede.find_superseded, which the original
# design note for this column didn't enumerate separately from lexical/
# cross_encoder; "decision_key" is reserved for the day the read-time
# decision_key consolidation (migration 016) becomes a write-time
# supersession instead of a projection-time reconciliation -- no call site
# emits it today. Soft vocabulary: an unrecognized value logs a WARNING but
# is still written, matching this module's fail-open posture (losing
# attribution precision is acceptable; losing a session is not).
SUPERSEDE_REASONS: frozenset[str] = frozenset({
    "lexical", "subject_key", "semantic", "cross_encoder",
    "cross_type_priority", "decision_key",
})


# ── writes ───────────────────────────────────────────────────────────────────

def insert_event(conn: sqlite3.Connection, event: Event) -> int:
    """Insert an event. On duplicate content_hash, increment mention_count and weight.

    Returns the row id of the inserted or existing event.
    """
    try:
        cursor = conn.execute(
            """
            INSERT INTO events
                (project_id, session_id, created_at, event_type, payload,
                 content_hash, weight, mention_count, evidence_id, decision_key,
                 captured_at_sha)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.project_id,
                event.session_id,
                event.created_at,
                event.event_type,
                json.dumps(event.payload, sort_keys=True, separators=(",", ":")),
                event.content_hash,
                event.weight,
                event.mention_count,
                event.evidence_id,
                event.decision_key,
                event.captured_at_sha,
            ),
        )
        row_id = cursor.lastrowid  # type: ignore[assignment]
        if event.evidence_id is not None:
            from cognikernel.storage.evidence import link_event_provenance

            link_event_provenance(
                conn,
                event_id=row_id,
                evidence_id=event.evidence_id,
                extractor_version="cognikernel.v2",
            )
        conn.commit()
        return row_id  # type: ignore[return-value]
    except sqlite3.IntegrityError:
        conn.execute(
            """
            UPDATE events
            SET mention_count = mention_count + 1,
                weight        = MIN(weight + ?, ?),
                evidence_id   = COALESCE(evidence_id, ?)
            WHERE project_id = ? AND content_hash = ?
            """,
            (
                WEIGHT_INCREMENT_ON_DEDUP,
                MAX_EVENT_WEIGHT,
                event.evidence_id,
                event.project_id,
                event.content_hash,
            ),
        )
        row = conn.execute(
            "SELECT id FROM events WHERE project_id = ? AND content_hash = ?",
            (event.project_id, event.content_hash),
        ).fetchone()
        if event.evidence_id is not None:
            from cognikernel.storage.evidence import link_event_provenance

            link_event_provenance(
                conn,
                event_id=row["id"],
                evidence_id=event.evidence_id,
                extractor_version="cognikernel.v2",
            )
        conn.commit()
        return row["id"]


# Hard cap on the chain walk in set_superseded_by's cycle check. superseded_by
# chains are expected to be short (a handful of restatements); this is a safety
# bound against a pathological pre-existing chain, not a realistic depth.
_MAX_CYCLE_WALK: int = 10_000


def set_superseded_by(
    conn: sqlite3.Connection,
    event_id: int,
    by_id: int,
    reason: str | None = None,
) -> bool:
    """Point event_id at by_id as its replacement, guarding against cycles.

    Refuses a self-link (event_id == by_id) and refuses if event_id is
    reachable by following by_id's superseded_by chain — a cycle of ANY
    length (a direct two-cycle A<->B, or a longer one A->B->C->A). Every live
    query filters `superseded_by IS NULL`, so a cycle silently drops every
    row in it out of memory even though none of them was ever actually
    replaced by something outside the cycle.

    ATOMICITY. The chain walk (reads) and the write must be one atomic unit,
    or two concurrent callers can each read "no cycle" before either commits,
    then both write — closing exactly the cycle the walk exists to prevent.
    Python's sqlite3 module does not take SQLite's write lock until the
    first DML statement, so a plain SELECT-then-UPDATE has a real gap: this
    was measured to form a two-cycle in >80% of iterations under genuine
    concurrent access (tests/reliability/test_supersession_race.py) before
    this fix. When no transaction is already open on `conn`, this function
    opens one with `BEGIN IMMEDIATE` before the walk, which takes SQLite's
    RESERVED lock up front — a second connection attempting the same then
    blocks (up to `busy_timeout`) rather than racing, so its own walk is
    guaranteed to see this call's write once it proceeds. When `conn`
    already has a transaction open (the mid-merge case: delta.merge has
    already written earlier in this same transaction), that transaction
    already holds the lock for its whole duration, so no extra BEGIN is
    needed or possible.

    Does not commit — callers inside a larger transaction (delta.merge,
    delta.supersede) rely on that; `mark_superseded` below commits for
    standalone callers. A `BEGIN IMMEDIATE` opened here is left open for
    that same caller-commits contract; it is never committed inside this
    function.

    A guarded-off write is not silent: it logs at WARNING so a refused
    supersession is diagnosable from the same log a hook failure would show
    up in, rather than reading identically to a successful one (DoD #4 —
    silence must not read as success).

    `reason` (T-103 / #13) names which gate fired -- see SUPERSEDE_REASONS
    for the fixed vocabulary the debugger expects. An unrecognized value is
    still written (fail-open) but logged, since a typo'd reason silently
    breaking debugger attribution is exactly the kind of degradation that
    must not read as success.

    Returns True if the link was written, False if guarded off.
    """
    if reason is not None and reason not in SUPERSEDE_REASONS:
        _log.warning(
            "set_superseded_by: reason %r is not in the fixed vocabulary %s "
            "-- writing it anyway (fail-open), but the debugger won't "
            "recognize it", reason, sorted(SUPERSEDE_REASONS),
        )
    if event_id == by_id:
        _log.warning(
            "set_superseded_by: refused self-link (event %s)", event_id
        )
        return False

    opened_transaction = False
    if not conn.in_transaction:
        conn.execute("BEGIN IMMEDIATE")
        opened_transaction = True

    try:
        cur = by_id
        seen: set[int] = set()
        for _ in range(_MAX_CYCLE_WALK):
            if cur == event_id:
                _log.warning(
                    "set_superseded_by: refused — event %s -> %s would close "
                    "a supersession cycle (event %s is reachable from %s)",
                    event_id, by_id, event_id, by_id,
                )
                if opened_transaction:
                    conn.rollback()  # nothing written; don't hold the lock
                return False
            if cur in seen:
                break  # walked into a pre-existing cycle that doesn't include event_id
            seen.add(cur)
            row = conn.execute(
                "SELECT superseded_by FROM events WHERE id = ?", (cur,)
            ).fetchone()
            cur = row["superseded_by"] if row is not None else None
            if cur is None:
                break
        conn.execute(
            """
            UPDATE events
            SET superseded_by = ?, superseded_at = ?, supersede_reason = ?
            WHERE id = ?
            """,
            (by_id, int(time.time() * 1000), reason, event_id),
        )
        return True
    except BaseException:
        if opened_transaction:
            conn.rollback()
        raise


def mark_superseded(
    conn: sqlite3.Connection,
    event_id: int,
    by_id: int,
    reason: str | None = None,
) -> None:
    """Record that event_id has been replaced by by_id. Both rows are kept."""
    set_superseded_by(conn, event_id, by_id, reason=reason)
    conn.commit()


def mark_archived(conn: sqlite3.Connection, event_id: int) -> None:
    conn.execute(
        "UPDATE events SET archived = 1, archived_at = ? WHERE id = ?",
        (int(time.time() * 1000), event_id),
    )
    conn.commit()


def update_weight(conn: sqlite3.Connection, event_id: int, weight: float) -> None:
    conn.execute(
        "UPDATE events SET weight = ? WHERE id = ?",
        (max(0.0, weight), event_id),
    )
    conn.commit()


# NOTE: a legacy `apply_weight_decay` used to live here. It was exported but
# never called in production, and — unlike the live decay paths
# (delta.decay.apply_decay_pass / delta.merge._apply_decay_inner) — it archived
# PROTECTED types (hard constraints, do-not-retry) too. Removed rather than
# fixed: one decay implementation, one archive policy.


def insert_extraction_failure(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    stage: str,
    error_message: str,
    raw_input_path: str,
) -> None:
    conn.execute(
        """
        INSERT INTO extraction_failures
            (project_id, session_id, failed_at, stage, error_message, raw_input_path)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            session_id,
            int(time.time() * 1000),
            stage,
            error_message,
            raw_input_path,
        ),
    )
    conn.commit()


# ── reads ────────────────────────────────────────────────────────────────────

def get_event_by_id(conn: sqlite3.Connection, event_id: int) -> Event | None:
    row = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
    return _row_to_event(row) if row else None


def get_events_for_projection(
    conn: sqlite3.Connection,
    project_id: str,
    after_id: int = 0,
) -> list[Event]:
    """Return active (non-archived, non-superseded) events for projection rebuild.

    Pass after_id = high_water_mark for an incremental delta query.
    """
    rows = conn.execute(
        """
        SELECT * FROM events
        WHERE project_id     = ?
          AND id             > ?
          AND archived       = 0
          AND superseded_by IS NULL
        ORDER BY id ASC
        """,
        (project_id, after_id),
    ).fetchall()
    return [_row_to_event(r) for r in rows]


def get_events_by_session(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
) -> list[Event]:
    rows = conn.execute(
        """
        SELECT * FROM events
        WHERE project_id = ? AND session_id = ?
        ORDER BY id ASC
        """,
        (project_id, session_id),
    ).fetchall()
    return [_row_to_event(r) for r in rows]


def get_events_by_type(
    conn: sqlite3.Connection,
    project_id: str,
    event_type: str,
    include_archived: bool = False,
) -> list[Event]:
    if include_archived:
        rows = conn.execute(
            "SELECT * FROM events WHERE project_id = ? AND event_type = ? ORDER BY id ASC",
            (project_id, event_type),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM events
            WHERE project_id = ? AND event_type = ? AND archived = 0
            ORDER BY id ASC
            """,
            (project_id, event_type),
        ).fetchall()
    return [_row_to_event(r) for r in rows]


def get_extraction_failures(
    conn: sqlite3.Connection,
    project_id: str,
    limit: int = 10,
) -> list[dict]:
    """Return the most recent extraction failures for a project."""
    rows = conn.execute(
        """
        SELECT session_id, failed_at, stage, error_message
        FROM extraction_failures
        WHERE project_id = ?
        ORDER BY failed_at DESC
        LIMIT ?
        """,
        (project_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_max_event_id(conn: sqlite3.Connection, project_id: str) -> int:
    row = conn.execute(
        "SELECT MAX(id) FROM events WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    return row[0] or 0


# ── internals ────────────────────────────────────────────────────────────────

def _row_to_event(row: sqlite3.Row) -> Event:
    keys = row.keys()
    return Event(
        id=row["id"],
        project_id=row["project_id"],
        session_id=row["session_id"],
        created_at=row["created_at"],
        event_type=row["event_type"],
        payload=json.loads(row["payload"]),
        content_hash=row["content_hash"],
        evidence_id=row["evidence_id"],
        weight=row["weight"],
        mention_count=row["mention_count"],
        superseded_by=row["superseded_by"],
        archived=bool(row["archived"]),
        decision_key=row["decision_key"] if "decision_key" in keys else None,
        captured_at_sha=row["captured_at_sha"] if "captured_at_sha" in keys else None,
    )
