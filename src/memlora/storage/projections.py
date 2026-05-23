from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

_HARD_TYPES      = frozenset({"CONSTRAINT_HARD"})
_GRAVEYARD_TYPES = frozenset({"APPROACH_ABANDONED_DO_NOT_RETRY"})
_COMPONENT_TYPES = frozenset({"COMPONENT_STATUS"})
_DECISION_TYPES  = frozenset({"DECISION", "CONSTRAINT_SOFT", "APPROACH_ABANDONED"})
_THREAD_TYPES    = frozenset({"THREAD_OPEN"})


@dataclass
class Projection:
    project_id: str
    built_at: int
    event_id_high_water: int
    hard_constraints: list[dict[str, Any]] = field(default_factory=list)
    ranked_decisions: list[dict[str, Any]] = field(default_factory=list)
    component_map: dict[str, Any] = field(default_factory=dict)
    graveyard: list[dict[str, Any]] = field(default_factory=list)
    active_threads: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""


# ── reads ────────────────────────────────────────────────────────────────────

def load_projection(conn: sqlite3.Connection, project_id: str) -> Projection | None:
    row = conn.execute(
        "SELECT * FROM state_projections WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    return _row_to_projection(row) if row else None


def needs_rebuild(conn: sqlite3.Connection, project_id: str) -> bool:
    """Return True if new events exist past the projection's high-water mark."""
    row = conn.execute(
        "SELECT event_id_high_water FROM state_projections WHERE project_id = ?",
        (project_id,),
    ).fetchone()
    if row is None:
        return True  # no projection exists yet

    high_water = row["event_id_high_water"]
    max_event = conn.execute(
        "SELECT MAX(id) FROM events WHERE project_id = ?",
        (project_id,),
    ).fetchone()[0] or 0

    return max_event > high_water


# ── writes ───────────────────────────────────────────────────────────────────

def save_projection(conn: sqlite3.Connection, projection: Projection) -> None:
    conn.execute(
        """
        INSERT INTO state_projections
            (project_id, built_at, event_id_high_water,
             hard_constraints, ranked_decisions, component_map,
             graveyard, active_threads, summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(project_id) DO UPDATE SET
            built_at            = excluded.built_at,
            event_id_high_water = excluded.event_id_high_water,
            hard_constraints    = excluded.hard_constraints,
            ranked_decisions    = excluded.ranked_decisions,
            component_map       = excluded.component_map,
            graveyard           = excluded.graveyard,
            active_threads      = excluded.active_threads,
            summary             = excluded.summary
        """,
        (
            projection.project_id,
            projection.built_at,
            projection.event_id_high_water,
            json.dumps(projection.hard_constraints, sort_keys=True, separators=(",", ":")),
            json.dumps(projection.ranked_decisions, sort_keys=True, separators=(",", ":")),
            json.dumps(projection.component_map, sort_keys=True, separators=(",", ":")),
            json.dumps(projection.graveyard, sort_keys=True, separators=(",", ":")),
            json.dumps(projection.active_threads, sort_keys=True, separators=(",", ":")),
            projection.summary,
        ),
    )
    conn.commit()


def invalidate_projection(conn: sqlite3.Connection, project_id: str) -> None:
    """Force a full rebuild on the next load by resetting the high-water mark to 0."""
    conn.execute(
        "UPDATE state_projections SET event_id_high_water = 0 WHERE project_id = ?",
        (project_id,),
    )
    conn.commit()


def rebuild_projection(conn: sqlite3.Connection, project_id: str) -> Projection:
    """Build a fresh Projection from all active events and persist it."""
    from memlora.storage.events import get_events_for_projection

    events = get_events_for_projection(conn, project_id, after_id=0)

    hard_constraints: list[dict[str, Any]] = []
    ranked_decisions: list[dict[str, Any]] = []
    component_map: dict[str, dict[str, Any]] = {}
    graveyard: list[dict[str, Any]] = []
    active_threads: list[dict[str, Any]] = []

    for event in events:
        rec: dict[str, Any] = {
            "id": event.id,
            "event_type": event.event_type,
            "weight": event.weight,
            "mention_count": event.mention_count,
            "session_id": event.session_id,
            "content_hash": event.content_hash,
            "payload": event.payload,
        }
        if event.event_type in _HARD_TYPES:
            hard_constraints.append(rec)
        elif event.event_type in _GRAVEYARD_TYPES:
            graveyard.append(rec)
        elif event.event_type in _COMPONENT_TYPES:
            path = event.payload.get("path", "")
            component_map[path] = rec
        elif event.event_type in _DECISION_TYPES:
            ranked_decisions.append(rec)
        elif event.event_type in _THREAD_TYPES:
            active_threads.append(rec)

    ranked_decisions.sort(key=lambda r: r["weight"], reverse=True)

    high_water = max((e.id for e in events if e.id is not None), default=0)

    projection = Projection(
        project_id=project_id,
        built_at=int(time.time() * 1000),
        event_id_high_water=high_water,
        hard_constraints=hard_constraints,
        ranked_decisions=ranked_decisions,
        component_map=component_map,
        graveyard=graveyard,
        active_threads=active_threads,
        summary="",
    )
    save_projection(conn, projection)
    return projection


def load_or_rebuild(conn: sqlite3.Connection, project_id: str) -> Projection:
    """Return the current projection, rebuilding from events if stale or missing."""
    if needs_rebuild(conn, project_id):
        return rebuild_projection(conn, project_id)
    proj = load_projection(conn, project_id)
    assert proj is not None
    return proj


# ── internals ────────────────────────────────────────────────────────────────

def _row_to_projection(row: sqlite3.Row) -> Projection:
    return Projection(
        project_id=row["project_id"],
        built_at=row["built_at"],
        event_id_high_water=row["event_id_high_water"],
        hard_constraints=json.loads(row["hard_constraints"]),
        ranked_decisions=json.loads(row["ranked_decisions"]),
        component_map=json.loads(row["component_map"]),
        graveyard=json.loads(row["graveyard"]),
        active_threads=json.loads(row["active_threads"]),
        summary=row["summary"],
    )
