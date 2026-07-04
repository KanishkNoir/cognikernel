from __future__ import annotations

import json
import sqlite3
import time

# Event + VALID_EVENT_TYPES live in the dependency-free base model module so the
# layered packages can type-hint Event without importing "up" into storage (the
# layering fix). Re-exported here for backward compatibility — existing
# `from memlora.storage.events import Event` call sites are unaffected.
from memlora.model import Event, VALID_EVENT_TYPES  # noqa: F401  (re-export)

# Weight boost applied to an event that already exists (dedup hit).
WEIGHT_INCREMENT_ON_DEDUP: float = 0.15

# Maximum weight any event can accumulate.
MAX_EVENT_WEIGHT: float = 5.0

# Weight below which events are archived during the decay pass. Single source
# of truth — delta.decay imports this (it previously carried its own copy).
ARCHIVE_THRESHOLD: float = 0.05


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
                 content_hash, weight, mention_count, evidence_id, decision_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        row_id = cursor.lastrowid  # type: ignore[assignment]
        if event.evidence_id is not None:
            from memlora.storage.evidence import link_event_provenance

            link_event_provenance(
                conn,
                event_id=row_id,
                evidence_id=event.evidence_id,
                extractor_version="memlora.v2",
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
            from memlora.storage.evidence import link_event_provenance

            link_event_provenance(
                conn,
                event_id=row["id"],
                evidence_id=event.evidence_id,
                extractor_version="memlora.v2",
            )
        conn.commit()
        return row["id"]


def mark_superseded(conn: sqlite3.Connection, event_id: int, by_id: int) -> None:
    """Record that event_id has been replaced by by_id. Both rows are kept."""
    conn.execute(
        "UPDATE events SET superseded_by = ? WHERE id = ?",
        (by_id, event_id),
    )
    conn.commit()


def mark_archived(conn: sqlite3.Connection, event_id: int) -> None:
    conn.execute("UPDATE events SET archived = 1 WHERE id = ?", (event_id,))
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
    )
