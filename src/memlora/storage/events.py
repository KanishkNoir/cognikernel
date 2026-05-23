from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

VALID_EVENT_TYPES: frozenset[str] = frozenset({
    "DECISION",
    "CONSTRAINT_HARD",
    "CONSTRAINT_SOFT",
    "COMPONENT_STATUS",
    "APPROACH_ABANDONED",
    "APPROACH_ABANDONED_DO_NOT_RETRY",
    "THREAD_OPEN",
    "THREAD_CLOSE",
})

# Weight boost applied to an event that already exists (dedup hit).
WEIGHT_INCREMENT_ON_DEDUP: float = 0.15

# Maximum weight any event can accumulate.
MAX_EVENT_WEIGHT: float = 5.0

# Weight below which events are archived during the decay pass.
ARCHIVE_THRESHOLD: float = 0.05


@dataclass
class Event:
    project_id: str
    session_id: str
    event_type: str
    payload: dict[str, Any]
    content_hash: str
    weight: float = 1.0
    mention_count: int = 1
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    id: int | None = None
    superseded_by: int | None = None
    archived: bool = False
    last_mentioned_session: int = 0

    def __post_init__(self) -> None:
        if self.event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type {self.event_type!r}. "
                f"Valid types: {sorted(VALID_EVENT_TYPES)}"
            )


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
                 content_hash, weight, mention_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
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
            ),
        )
        conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]
    except sqlite3.IntegrityError:
        conn.execute(
            """
            UPDATE events
            SET mention_count = mention_count + 1,
                weight        = MIN(weight + ?, ?)
            WHERE project_id = ? AND content_hash = ?
            """,
            (WEIGHT_INCREMENT_ON_DEDUP, MAX_EVENT_WEIGHT, event.project_id, event.content_hash),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id FROM events WHERE project_id = ? AND content_hash = ?",
            (event.project_id, event.content_hash),
        ).fetchone()
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


def apply_weight_decay(
    conn: sqlite3.Connection,
    project_id: str,
    factor: float,
    exclude_session_id: str,
    archive_threshold: float = ARCHIVE_THRESHOLD,
) -> None:
    """Multiply weight by factor for all non-archived events not in the current session.

    Events whose weight falls below archive_threshold are then archived.
    Called by Stage 5 (delta merge) at the end of each session.
    """
    conn.execute(
        """
        UPDATE events
        SET weight = MAX(0.0, weight * ?)
        WHERE project_id  = ?
          AND session_id != ?
          AND archived    = 0
        """,
        (factor, project_id, exclude_session_id),
    )
    conn.execute(
        """
        UPDATE events
        SET archived = 1
        WHERE project_id = ?
          AND archived   = 0
          AND weight     < ?
        """,
        (project_id, archive_threshold),
    )
    conn.commit()


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
    return Event(
        id=row["id"],
        project_id=row["project_id"],
        session_id=row["session_id"],
        created_at=row["created_at"],
        event_type=row["event_type"],
        payload=json.loads(row["payload"]),
        content_hash=row["content_hash"],
        weight=row["weight"],
        mention_count=row["mention_count"],
        superseded_by=row["superseded_by"],
        archived=bool(row["archived"]),
    )
