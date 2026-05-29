from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Any

from memlora.storage.sections import (
    COMPONENT_TYPES as _COMPONENT_TYPES,
    DECISION_TYPES as _DECISION_TYPES,
    GRAVEYARD_TYPES as _GRAVEYARD_TYPES,
    HARD_TYPES as _HARD_TYPES,
    THREAD_TYPES as _THREAD_TYPES,
)
from memlora.utils.paths import canonicalize_path, is_bare_basename


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
            # C2: canonicalize and drop bare-basename noise. Two events with
            # different separator styles or trailing slashes now collapse to
            # one key; an event whose path is just `env.py` (no directory)
            # is dropped — `alembic/env.py` is the authoritative version.
            raw_path = event.payload.get("path", "")
            canonical = canonicalize_path(raw_path)
            if not canonical or is_bare_basename(canonical):
                continue
            # Mirror the canonical path into the rec payload so callers
            # downstream see the normalized form.
            rec["payload"] = {**rec["payload"], "path": canonical}
            # Lossless collapse. The latest event (highest id — events arrive
            # id-ASC) is the representative status for display, but two
            # quantities accumulate across every event folding into this path:
            #   - mention_count sums, so hot-file tallies match the historical
            #     raw-event aggregation render performed;
            #   - weight takes the max, so greedy's per-path selection (which
            #     historically kept the highest-weight event) is unchanged and
            #     no event gets dropped from the budget by the collapse.
            prior = component_map.get(canonical)
            if prior is not None:
                rec["mention_count"] = prior["mention_count"] + rec["mention_count"]
                rec["weight"] = max(prior["weight"], rec["weight"])
            component_map[canonical] = rec
        elif event.event_type in _DECISION_TYPES:
            ranked_decisions.append(rec)
        elif event.event_type in _THREAD_TYPES:
            active_threads.append(rec)

    # Unit 3: recompute each rec's weight via the full composite model
    # (base × recency × repetition × centrality × activity × type) before
    # ranking. This is the single place the live ranking is computed.
    _apply_composite_weights(
        conn,
        project_id,
        events,
        [hard_constraints, ranked_decisions, graveyard,
         active_threads, list(component_map.values())],
    )

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


def projection_to_events(proj: Projection):
    """Flatten a Projection's buckets back into a list of active Event objects.

    The render path consumes the projection as its event source (so partition
    routing + component collapse live in exactly one place — rebuild_projection)
    but still runs greedy budget selection + render-time partitioning on a flat
    list. Events are returned id-ascending to match the historical
    `get_events_for_projection` ordering, preserving render byte-parity.

    Only active events are present in a Projection, so archived/superseded are
    left at their Event defaults (False / None).
    """
    from memlora.storage.events import Event

    out: list[Event] = []
    buckets = (
        proj.hard_constraints,
        proj.ranked_decisions,
        proj.graveyard,
        proj.active_threads,
        list(proj.component_map.values()),
    )
    for bucket in buckets:
        for r in bucket:
            out.append(
                Event(
                    project_id=proj.project_id,
                    session_id=r.get("session_id", ""),
                    event_type=r["event_type"],
                    payload=r["payload"],
                    content_hash=r["content_hash"],
                    weight=r["weight"],
                    mention_count=r.get("mention_count", 1),
                    id=r.get("id"),
                )
            )
    out.sort(key=lambda e: (e.id if e.id is not None else 0))
    return out


# ── internals ────────────────────────────────────────────────────────────────

def _apply_composite_weights(
    conn: sqlite3.Connection,
    project_id: str,
    events: list,
    bucket_lists: list[list[dict[str, Any]]],
) -> None:
    """Rewrite each rec's ``weight`` in place using the composite model.

    Revives ``compression.weights.compute_weight`` — the full
    base × recency × repetition × centrality × activity × type formula that
    until now was implemented and unit-tested but never called in production.

    Two signals are derived here rather than stored:
      - recency: a session ordinal (1..N by first-seen order in this rebuild).
        No schema change — `events` arrive id-ascending so first-seen order is
        a stable monotonic session index. `last_mentioned_session` is the
        ordinal of the event's session.
      - centrality: PageRank over the symbol import graph (local edges only).
    """
    from memlora.compression.centrality import compute_file_centrality
    from memlora.compression.weights import compute_weight
    from memlora.storage.events import Event

    # Session ordinals — 1..N by first appearance (events are id-ascending).
    session_ord: dict[str, int] = {}
    for e in events:
        if e.session_id not in session_ord:
            session_ord[e.session_id] = len(session_ord) + 1
    current_session = len(session_ord)

    # PageRank centrality over the local import graph (lazy import avoids a
    # storage→symbols module-load cycle).
    import_graph: dict[str, list[str]] = {}
    try:
        from memlora.symbols.store import load_symbol_edges
        for edge in load_symbol_edges(conn, project_id):
            import_graph.setdefault(edge.from_path, []).append(edge.to_path)
    except Exception:
        import_graph = {}
    centrality_map = compute_file_centrality(import_graph) if import_graph else {}

    # Activity status per canonical component path (from the component recs).
    activity_map: dict[str, dict[str, Any]] = {}
    for bucket in bucket_lists:
        for rec in bucket:
            if rec["event_type"] == "COMPONENT_STATUS":
                path = rec["payload"].get("path", "")
                if path:
                    activity_map[path] = {"status": rec["payload"].get("status", "unknown")}

    for bucket in bucket_lists:
        for rec in bucket:
            payload = rec["payload"]
            if rec["event_type"] == "COMPONENT_STATUS":
                affected = [payload["path"]] if payload.get("path") else []
            else:
                affected = list(payload.get("affected_files", []))
            ev = Event(
                project_id=project_id,
                session_id=rec.get("session_id", ""),
                event_type=rec["event_type"],
                payload={**payload, "affected_files": affected},
                content_hash=rec["content_hash"],
                weight=rec["weight"],
                mention_count=rec.get("mention_count", 1),
                last_mentioned_session=session_ord.get(rec.get("session_id", ""), 0),
            )
            rec["weight"] = compute_weight(ev, activity_map, centrality_map, current_session)


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
