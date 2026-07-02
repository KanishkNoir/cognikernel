import sqlite3
import time

import pytest

from memlora.storage.events import Event, insert_event
from memlora.storage.projections import (
    Projection,
    invalidate_projection,
    load_projection,
    needs_rebuild,
    save_projection,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def make_projection(**overrides) -> Projection:
    defaults = dict(
        project_id="proj1",
        built_at=int(time.time() * 1000),
        event_id_high_water=0,
        hard_constraints=[{"description": "No Redis", "rationale": "Ops complexity"}],
        ranked_decisions=[{"description": "Use SQLite"}],
        component_map={"auth/middleware.py": {"status": "stable"}},
        graveyard=[{"description": "Tried Redis", "reason": "Too complex"}],
        active_threads=[{"description": "Add login flow", "state": "in_progress"}],
        summary="Python API project. Currently adding login flow.",
    )
    defaults.update(overrides)
    return Projection(**defaults)


def make_and_insert_event(conn: sqlite3.Connection, content_hash: str = "h1") -> int:
    e = Event(
        project_id="proj1",
        session_id="sess1",
        event_type="DECISION",
        payload={"description": "test"},
        content_hash=content_hash,
    )
    return insert_event(conn, e)


# ── load_projection ───────────────────────────────────────────────────────────

class TestLoadProjection:
    def test_returns_none_for_missing_project(self, conn: sqlite3.Connection) -> None:
        assert load_projection(conn, "nonexistent") is None

    def test_returns_projection_after_save(self, conn: sqlite3.Connection) -> None:
        save_projection(conn, make_projection())
        loaded = load_projection(conn, "proj1")
        assert loaded is not None

    def test_fields_round_trip_correctly(self, conn: sqlite3.Connection) -> None:
        proj = make_projection(event_id_high_water=42, summary="Round-trip test")
        save_projection(conn, proj)
        loaded = load_projection(conn, "proj1")
        assert loaded is not None
        assert loaded.event_id_high_water == 42
        assert loaded.summary == "Round-trip test"
        assert loaded.hard_constraints == proj.hard_constraints
        assert loaded.ranked_decisions == proj.ranked_decisions
        assert loaded.component_map == proj.component_map
        assert loaded.graveyard == proj.graveyard
        assert loaded.active_threads == proj.active_threads

    def test_complex_nested_structures_preserved(self, conn: sqlite3.Connection) -> None:
        proj = make_projection(
            component_map={"a/b.py": {"status": "in_flux", "intent": "Rewriting auth"}},
            graveyard=[
                {"description": "Redis", "reason": "Ops cost"},
                {"description": "Kafka", "reason": "Too heavy"},
            ],
        )
        save_projection(conn, proj)
        loaded = load_projection(conn, "proj1")
        assert loaded is not None
        assert loaded.component_map["a/b.py"]["intent"] == "Rewriting auth"
        assert len(loaded.graveyard) == 2


# ── save_projection ───────────────────────────────────────────────────────────

class TestSaveProjection:
    def test_upsert_updates_existing_row(self, conn: sqlite3.Connection) -> None:
        save_projection(conn, make_projection(event_id_high_water=10, summary="v1"))
        save_projection(conn, make_projection(event_id_high_water=20, summary="v2"))
        loaded = load_projection(conn, "proj1")
        assert loaded is not None
        assert loaded.event_id_high_water == 20
        assert loaded.summary == "v2"

    def test_only_one_row_per_project(self, conn: sqlite3.Connection) -> None:
        save_projection(conn, make_projection())
        save_projection(conn, make_projection())
        count = conn.execute(
            "SELECT COUNT(*) FROM state_projections WHERE project_id='proj1'"
        ).fetchone()[0]
        assert count == 1

    def test_multiple_projects_independent(self, conn: sqlite3.Connection) -> None:
        save_projection(conn, make_projection(project_id="p1", summary="project 1"))
        save_projection(conn, make_projection(project_id="p2", summary="project 2"))
        p1 = load_projection(conn, "p1")
        p2 = load_projection(conn, "p2")
        assert p1 is not None and p1.summary == "project 1"
        assert p2 is not None and p2.summary == "project 2"


# ── invalidate_projection ─────────────────────────────────────────────────────

class TestInvalidateProjection:
    def test_sets_high_water_to_sentinel(self, conn: sqlite3.Connection) -> None:
        save_projection(conn, make_projection(event_id_high_water=99))
        invalidate_projection(conn, "proj1")
        loaded = load_projection(conn, "proj1")
        assert loaded is not None
        # -1 matches merge._invalidate_projection_inner and forces a rebuild
        # even for a store whose events were all deleted.
        assert loaded.event_id_high_water == -1

    def test_no_op_for_missing_project(self, conn: sqlite3.Connection) -> None:
        invalidate_projection(conn, "ghost")  # should not raise


# ── needs_rebuild ─────────────────────────────────────────────────────────────

class TestNeedsRebuild:
    def test_true_when_no_projection_exists(self, conn: sqlite3.Connection) -> None:
        make_and_insert_event(conn)
        assert needs_rebuild(conn, "proj1") is True

    def test_false_when_high_water_matches_max_event(self, conn: sqlite3.Connection) -> None:
        event_id = make_and_insert_event(conn)
        save_projection(conn, make_projection(event_id_high_water=event_id))
        assert needs_rebuild(conn, "proj1") is False

    def test_true_when_new_events_past_high_water(self, conn: sqlite3.Connection) -> None:
        id1 = make_and_insert_event(conn, "h1")
        save_projection(conn, make_projection(event_id_high_water=id1))
        make_and_insert_event(conn, "h2")  # new event past high-water
        assert needs_rebuild(conn, "proj1") is True

    def test_true_after_invalidation(self, conn: sqlite3.Connection) -> None:
        event_id = make_and_insert_event(conn)
        save_projection(conn, make_projection(event_id_high_water=event_id))
        invalidate_projection(conn, "proj1")
        assert needs_rebuild(conn, "proj1") is True

    def test_false_for_project_with_no_events(self, conn: sqlite3.Connection) -> None:
        save_projection(conn, make_projection(event_id_high_water=0))
        assert needs_rebuild(conn, "proj1") is False
