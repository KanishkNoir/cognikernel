"""Tests for rebuild_projection and load_or_rebuild in memlora.storage.projections."""
from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from memlora.storage.events import Event, insert_event
from memlora.storage.projections import (
    Projection,
    load_or_rebuild,
    load_projection,
    needs_rebuild,
    projection_to_events,
    rebuild_projection,
    save_projection,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def insert(
    conn: sqlite3.Connection,
    event_type: str,
    content_hash: str,
    weight: float = 1.0,
    payload: dict[str, Any] | None = None,
    project_id: str = "proj1",
    session_id: str = "sess1",
) -> int:
    if payload is None:
        if event_type == "COMPONENT_STATUS":
            payload = {"path": f"src/{content_hash}.py", "status": "stable"}
        else:
            payload = {"description": f"event {content_hash}"}
    e = Event(
        project_id=project_id,
        session_id=session_id,
        event_type=event_type,
        payload=payload,
        content_hash=content_hash,
        weight=weight,
    )
    return insert_event(conn, e)


# ── rebuild_projection ────────────────────────────────────────────────────────

class TestRebuildProjection:
    def test_empty_db_returns_empty_projection(self, conn: sqlite3.Connection) -> None:
        proj = rebuild_projection(conn, "proj1")
        assert proj.project_id == "proj1"
        assert proj.event_id_high_water == 0
        assert proj.hard_constraints == []
        assert proj.ranked_decisions == []
        assert proj.component_map == {}
        assert proj.graveyard == []
        assert proj.active_threads == []

    def test_decision_goes_to_ranked_decisions(self, conn: sqlite3.Connection) -> None:
        insert(conn, "DECISION", "h1")
        proj = rebuild_projection(conn, "proj1")
        assert len(proj.ranked_decisions) == 1
        assert proj.ranked_decisions[0]["content_hash"] == "h1"

    def test_constraint_soft_goes_to_ranked_decisions(self, conn: sqlite3.Connection) -> None:
        insert(conn, "CONSTRAINT_SOFT", "h1")
        proj = rebuild_projection(conn, "proj1")
        assert len(proj.ranked_decisions) == 1

    def test_approach_abandoned_goes_to_ranked_decisions(self, conn: sqlite3.Connection) -> None:
        insert(conn, "APPROACH_ABANDONED", "h1")
        proj = rebuild_projection(conn, "proj1")
        assert len(proj.ranked_decisions) == 1

    def test_constraint_hard_goes_to_hard_constraints(self, conn: sqlite3.Connection) -> None:
        insert(conn, "CONSTRAINT_HARD", "h1")
        proj = rebuild_projection(conn, "proj1")
        assert len(proj.hard_constraints) == 1
        assert len(proj.ranked_decisions) == 0

    def test_approach_abandoned_do_not_retry_goes_to_graveyard(self, conn: sqlite3.Connection) -> None:
        insert(conn, "APPROACH_ABANDONED_DO_NOT_RETRY", "h1")
        proj = rebuild_projection(conn, "proj1")
        assert len(proj.graveyard) == 1
        assert len(proj.ranked_decisions) == 0

    def test_component_status_goes_to_component_map(self, conn: sqlite3.Connection) -> None:
        insert(conn, "COMPONENT_STATUS", "h1",
               payload={"path": "src/api.py", "status": "stable"})
        proj = rebuild_projection(conn, "proj1")
        assert "src/api.py" in proj.component_map

    def test_thread_open_goes_to_active_threads(self, conn: sqlite3.Connection) -> None:
        insert(conn, "THREAD_OPEN", "h1",
               payload={"description": "Add auth", "state": "in_progress"})
        proj = rebuild_projection(conn, "proj1")
        assert len(proj.active_threads) == 1

    def test_thread_close_not_included(self, conn: sqlite3.Connection) -> None:
        insert(conn, "THREAD_CLOSE", "h1",
               payload={"description": "Done with auth"})
        proj = rebuild_projection(conn, "proj1")
        assert proj.active_threads == []
        assert proj.ranked_decisions == []
        assert proj.hard_constraints == []
        assert proj.graveyard == []
        assert proj.component_map == {}

    def test_component_status_latest_per_path_wins(self, conn: sqlite3.Connection) -> None:
        insert(conn, "COMPONENT_STATUS", "h1",
               payload={"path": "src/api.py", "status": "stable"})
        insert(conn, "COMPONENT_STATUS", "h2",
               payload={"path": "src/api.py", "status": "blocked"})
        proj = rebuild_projection(conn, "proj1")
        # Latest event is the representative status...
        assert proj.component_map["src/api.py"]["content_hash"] == "h2"
        assert proj.component_map["src/api.py"]["payload"]["status"] == "blocked"
        # ...but mention_count accumulates across both collapsed events so that
        # hot-file tallies match the historical raw-event aggregation.
        assert proj.component_map["src/api.py"]["mention_count"] == 2

    def test_ranked_decisions_sorted_by_weight_desc(self, conn: sqlite3.Connection) -> None:
        insert(conn, "DECISION", "h1", weight=0.5)
        insert(conn, "DECISION", "h2", weight=2.0)
        insert(conn, "DECISION", "h3", weight=1.0)
        proj = rebuild_projection(conn, "proj1")
        weights = [r["weight"] for r in proj.ranked_decisions]
        assert weights == sorted(weights, reverse=True)

    def test_high_water_set_to_max_event_id(self, conn: sqlite3.Connection) -> None:
        id1 = insert(conn, "DECISION", "h1")
        id2 = insert(conn, "DECISION", "h2")
        proj = rebuild_projection(conn, "proj1")
        assert proj.event_id_high_water == max(id1, id2)

    def test_saves_projection_to_db(self, conn: sqlite3.Connection) -> None:
        insert(conn, "DECISION", "h1")
        rebuild_projection(conn, "proj1")
        stored = load_projection(conn, "proj1")
        assert stored is not None
        assert len(stored.ranked_decisions) == 1

    def test_archived_events_excluded(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """INSERT INTO events
               (project_id, session_id, created_at, event_type,
                payload, content_hash, weight, mention_count, archived)
               VALUES ('proj1', 's1', 0, 'DECISION',
                '{"description":"stale"}', 'h_old', 0.04, 1, 1)"""
        )
        conn.commit()
        proj = rebuild_projection(conn, "proj1")
        assert proj.ranked_decisions == []

    def test_superseded_events_excluded(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """INSERT INTO events
               (project_id, session_id, created_at, event_type,
                payload, content_hash, weight, mention_count, superseded_by)
               VALUES ('proj1', 's1', 0, 'DECISION',
                '{"description":"old"}', 'h_old', 1.0, 1, 99)"""
        )
        conn.commit()
        proj = rebuild_projection(conn, "proj1")
        assert proj.ranked_decisions == []

    def test_different_project_isolated(self, conn: sqlite3.Connection) -> None:
        insert(conn, "DECISION", "h1", project_id="other_proj")
        proj = rebuild_projection(conn, "proj1")
        assert proj.ranked_decisions == []

    def test_record_contains_expected_fields(self, conn: sqlite3.Connection) -> None:
        insert(conn, "DECISION", "h1", weight=1.5)
        proj = rebuild_projection(conn, "proj1")
        rec = proj.ranked_decisions[0]
        for field in ("id", "event_type", "weight", "mention_count",
                      "session_id", "content_hash", "payload"):
            assert field in rec

    def test_multiple_types_in_one_rebuild(self, conn: sqlite3.Connection) -> None:
        insert(conn, "CONSTRAINT_HARD", "h1")
        insert(conn, "DECISION", "h2")
        insert(conn, "APPROACH_ABANDONED_DO_NOT_RETRY", "h3")
        insert(conn, "COMPONENT_STATUS", "h4",
               payload={"path": "src/a.py", "status": "stable"})
        insert(conn, "THREAD_OPEN", "h5",
               payload={"description": "do work", "state": "in_progress"})
        proj = rebuild_projection(conn, "proj1")
        assert len(proj.hard_constraints) == 1
        assert len(proj.ranked_decisions) == 1
        assert len(proj.graveyard) == 1
        assert len(proj.component_map) == 1
        assert len(proj.active_threads) == 1


# ── C2 canonicalization invariants ────────────────────────────────────────────


class TestComponentMapCanonicalization:
    """C2 §1: drop bare basenames; dedup by canonical path; normalize separators."""

    def test_bare_basename_is_dropped(self, conn: sqlite3.Connection) -> None:
        insert(conn, "COMPONENT_STATUS", "h1",
               payload={"path": "env.py", "status": "stable"})
        proj = rebuild_projection(conn, "proj1")
        assert proj.component_map == {}

    def test_directory_prefixed_path_is_kept(self, conn: sqlite3.Connection) -> None:
        insert(conn, "COMPONENT_STATUS", "h1",
               payload={"path": "alembic/env.py", "status": "stable"})
        proj = rebuild_projection(conn, "proj1")
        assert "alembic/env.py" in proj.component_map

    def test_bare_and_full_path_for_same_file_only_keeps_full(
        self, conn: sqlite3.Connection,
    ) -> None:
        """The exact C4 bug: extraction sometimes produced both 'env.py' and
        'alembic/env.py' — projection should drop the bare and keep the full."""
        insert(conn, "COMPONENT_STATUS", "h1",
               payload={"path": "env.py", "status": "modified"})
        insert(conn, "COMPONENT_STATUS", "h2",
               payload={"path": "alembic/env.py", "status": "modified"})

        proj = rebuild_projection(conn, "proj1")
        assert list(proj.component_map.keys()) == ["alembic/env.py"]

    def test_backslash_separators_collapse_with_forward_slash(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Two events using different separator styles end up under one key."""
        insert(conn, "COMPONENT_STATUS", "h1",
               payload={"path": "app/main.py", "status": "stable"})
        insert(conn, "COMPONENT_STATUS", "h2",
               payload={"path": "app\\main.py", "status": "modified"})

        proj = rebuild_projection(conn, "proj1")
        assert list(proj.component_map.keys()) == ["app/main.py"]

    def test_payload_path_is_rewritten_to_canonical(
        self, conn: sqlite3.Connection,
    ) -> None:
        """component_map[key].payload.path matches the canonical key — callers
        downstream can trust the payload."""
        insert(conn, "COMPONENT_STATUS", "h1",
               payload={"path": "app\\models.py", "status": "modified"})
        proj = rebuild_projection(conn, "proj1")
        rec = proj.component_map["app/models.py"]
        assert rec["payload"]["path"] == "app/models.py"


# ── composite weighting (Unit 3) ──────────────────────────────────────────────

class TestCompositeWeighting:
    def test_composite_replaces_raw_weight(self, conn: sqlite3.Connection) -> None:
        """A single-mention DECISION's stored weight is replaced by the composite
        (base 0.7 × neutral recency/repetition/centrality/activity/type) — proof
        compute_weight is now a live, non-test caller."""
        insert(conn, "DECISION", "h1", weight=1.0, session_id="s1")
        proj = rebuild_projection(conn, "proj1")
        assert abs(proj.ranked_decisions[0]["weight"] - 0.7) < 1e-6

    def test_recent_decision_ranks_above_stale(self, conn: sqlite3.Connection) -> None:
        """Recency now influences ranking: two equally-mentioned decisions, the
        one from a later session ranks higher even with identical raw weight."""
        insert(conn, "DECISION", "old", weight=1.0, session_id="s1")
        insert(conn, "THREAD_OPEN", "t", session_id="s2",
               payload={"description": "advance the session clock", "state": "x"})
        insert(conn, "DECISION", "new", weight=1.0, session_id="s3")
        proj = rebuild_projection(conn, "proj1")
        hashes = [r["content_hash"] for r in proj.ranked_decisions]
        assert hashes.index("new") < hashes.index("old")

    def test_constraint_hard_outweighs_decision(self, conn: sqlite3.Connection) -> None:
        """Type multiplier is live: a hard constraint (base 1.0 × type 1.5)
        scores well above a decision (base 0.7 × type 1.0) at equal raw weight."""
        insert(conn, "CONSTRAINT_HARD", "c", weight=1.0, session_id="s1")
        insert(conn, "DECISION", "d", weight=1.0, session_id="s1")
        proj = rebuild_projection(conn, "proj1")
        assert proj.hard_constraints[0]["weight"] > proj.ranked_decisions[0]["weight"]


# ── projection_to_events (render source) ──────────────────────────────────────

class TestProjectionToEvents:
    def test_roundtrips_all_buckets_id_sorted(self, conn: sqlite3.Connection) -> None:
        insert(conn, "CONSTRAINT_HARD", "h1")
        insert(conn, "DECISION", "h2")
        insert(conn, "APPROACH_ABANDONED_DO_NOT_RETRY", "h3")
        insert(conn, "COMPONENT_STATUS", "h4",
               payload={"path": "src/a.py", "status": "stable"})
        insert(conn, "THREAD_OPEN", "h5",
               payload={"description": "do work", "state": "in_progress"})
        proj = rebuild_projection(conn, "proj1")
        events = projection_to_events(proj)
        assert {e.event_type for e in events} == {
            "CONSTRAINT_HARD", "DECISION", "APPROACH_ABANDONED_DO_NOT_RETRY",
            "COMPONENT_STATUS", "THREAD_OPEN",
        }
        ids = [e.id for e in events]
        assert ids == sorted(ids)

    def test_component_mention_sum_preserved(self, conn: sqlite3.Connection) -> None:
        insert(conn, "COMPONENT_STATUS", "h1",
               payload={"path": "src/api.py", "status": "stable"})
        insert(conn, "COMPONENT_STATUS", "h2",
               payload={"path": "src/api.py", "status": "blocked"})
        events = projection_to_events(rebuild_projection(conn, "proj1"))
        comp = [e for e in events if e.event_type == "COMPONENT_STATUS"]
        assert len(comp) == 1
        assert comp[0].mention_count == 2


# ── load_or_rebuild ───────────────────────────────────────────────────────────

class TestLoadOrRebuild:
    def test_returns_fresh_projection_when_none_exists(self, conn: sqlite3.Connection) -> None:
        insert(conn, "DECISION", "h1")
        proj = load_or_rebuild(conn, "proj1")
        assert proj is not None
        assert len(proj.ranked_decisions) == 1

    def test_returns_cached_projection_when_current(self, conn: sqlite3.Connection) -> None:
        event_id = insert(conn, "DECISION", "h1")
        save_projection(conn, Projection(
            project_id="proj1",
            built_at=0,
            event_id_high_water=event_id,
            ranked_decisions=[{"id": event_id, "content_hash": "cached"}],
        ))
        proj = load_or_rebuild(conn, "proj1")
        # should return cached projection, not rebuild
        assert proj.ranked_decisions[0]["content_hash"] == "cached"

    def test_rebuilds_when_new_events_past_high_water(self, conn: sqlite3.Connection) -> None:
        id1 = insert(conn, "DECISION", "h1")
        save_projection(conn, Projection(
            project_id="proj1",
            built_at=0,
            event_id_high_water=id1,
        ))
        insert(conn, "DECISION", "h2")  # new event
        proj = load_or_rebuild(conn, "proj1")
        assert len(proj.ranked_decisions) == 2

    def test_rebuilds_when_high_water_is_minus_one(self, conn: sqlite3.Connection) -> None:
        insert(conn, "DECISION", "h1")
        conn.execute(
            """INSERT INTO state_projections
               (project_id, built_at, event_id_high_water,
                hard_constraints, ranked_decisions, component_map,
                graveyard, active_threads, summary)
               VALUES ('proj1', 0, -1, '[]', '[]', '{}', '[]', '[]', '')"""
        )
        conn.commit()
        proj = load_or_rebuild(conn, "proj1")
        assert len(proj.ranked_decisions) == 1

    def test_empty_db_no_events_returns_empty_projection(self, conn: sqlite3.Connection) -> None:
        proj = load_or_rebuild(conn, "proj1")
        assert proj.event_id_high_water == 0
        assert proj.ranked_decisions == []
