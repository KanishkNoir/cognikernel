import sqlite3
from typing import Any

import pytest

from memlora.storage.events import (
    VALID_EVENT_TYPES,
    WEIGHT_INCREMENT_ON_DEDUP,
    Event,
    get_event_by_id,
    get_events_by_session,
    get_events_by_type,
    get_events_for_projection,
    get_max_event_id,
    insert_event,
    insert_extraction_failure,
    mark_archived,
    mark_superseded,
    update_weight,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def make_event(**overrides: Any) -> Event:
    defaults: dict[str, Any] = {
        "project_id": "proj1",
        "session_id": "sess1",
        "event_type": "DECISION",
        "payload": {"description": "Use SQLite", "rationale": "Local-first"},
        "content_hash": "abc123",
        "weight": 1.0,
    }
    defaults.update(overrides)
    return Event(**defaults)


# ── Event dataclass ───────────────────────────────────────────────────────────

class TestEventValidation:
    def test_invalid_event_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown event_type"):
            make_event(event_type="BOGUS")

    @pytest.mark.parametrize("etype", sorted(VALID_EVENT_TYPES))
    def test_all_valid_event_types_accepted(self, etype: str) -> None:
        e = make_event(event_type=etype, content_hash=etype)
        assert e.event_type == etype

    def test_created_at_defaults_to_now(self) -> None:
        import time
        before = int(time.time() * 1000)
        e = make_event()
        after = int(time.time() * 1000)
        assert before <= e.created_at <= after


# ── insert_event ─────────────────────────────────────────────────────────────

class TestInsertEvent:
    def test_returns_positive_int(self, conn: sqlite3.Connection) -> None:
        row_id = insert_event(conn, make_event())
        assert isinstance(row_id, int) and row_id > 0

    def test_persists_all_fields(self, conn: sqlite3.Connection) -> None:
        e = make_event(weight=0.75)
        row_id = insert_event(conn, e)
        stored = get_event_by_id(conn, row_id)
        assert stored is not None
        assert stored.project_id == "proj1"
        assert stored.event_type == "DECISION"
        assert stored.weight == pytest.approx(0.75)
        assert stored.payload == e.payload
        assert stored.content_hash == "abc123"

    def test_duplicate_increments_mention_count(self, conn: sqlite3.Connection) -> None:
        e = make_event()
        insert_event(conn, e)
        insert_event(conn, e)
        stored = conn.execute(
            "SELECT mention_count FROM events WHERE project_id=? AND content_hash=?",
            ("proj1", "abc123"),
        ).fetchone()
        assert stored["mention_count"] == 2

    def test_duplicate_increments_weight(self, conn: sqlite3.Connection) -> None:
        e = make_event(weight=1.0)
        insert_event(conn, e)
        insert_event(conn, e)
        stored = conn.execute(
            "SELECT weight FROM events WHERE project_id=? AND content_hash=?",
            ("proj1", "abc123"),
        ).fetchone()
        assert stored["weight"] == pytest.approx(1.0 + WEIGHT_INCREMENT_ON_DEDUP)

    def test_duplicate_returns_same_id(self, conn: sqlite3.Connection) -> None:
        e = make_event()
        id1 = insert_event(conn, e)
        id2 = insert_event(conn, e)
        assert id1 == id2

    def test_different_projects_same_hash_both_stored(self, conn: sqlite3.Connection) -> None:
        e1 = make_event(project_id="proj1")
        e2 = make_event(project_id="proj2")
        id1 = insert_event(conn, e1)
        id2 = insert_event(conn, e2)
        assert id1 != id2

    def test_ids_are_monotonically_increasing(self, conn: sqlite3.Connection) -> None:
        ids = [insert_event(conn, make_event(content_hash=f"h{i}")) for i in range(5)]
        assert ids == sorted(ids)


# ── mark_superseded ───────────────────────────────────────────────────────────

class TestMarkSuperseded:
    def test_sets_superseded_by_field(self, conn: sqlite3.Connection) -> None:
        id1 = insert_event(conn, make_event(content_hash="old"))
        id2 = insert_event(conn, make_event(content_hash="new"))
        mark_superseded(conn, id1, by_id=id2)
        row = conn.execute("SELECT superseded_by FROM events WHERE id=?", (id1,)).fetchone()
        assert row["superseded_by"] == id2

    def test_superseded_event_excluded_from_projection(self, conn: sqlite3.Connection) -> None:
        id1 = insert_event(conn, make_event(content_hash="old"))
        id2 = insert_event(conn, make_event(content_hash="new"))
        mark_superseded(conn, id1, by_id=id2)
        events = get_events_for_projection(conn, "proj1")
        assert all(e.id != id1 for e in events)


# ── mark_archived ─────────────────────────────────────────────────────────────

class TestMarkArchived:
    def test_sets_archived_flag(self, conn: sqlite3.Connection) -> None:
        row_id = insert_event(conn, make_event())
        mark_archived(conn, row_id)
        row = conn.execute("SELECT archived FROM events WHERE id=?", (row_id,)).fetchone()
        assert row["archived"] == 1

    def test_archived_event_excluded_from_projection(self, conn: sqlite3.Connection) -> None:
        row_id = insert_event(conn, make_event())
        mark_archived(conn, row_id)
        events = get_events_for_projection(conn, "proj1")
        assert len(events) == 0


# ── update_weight ─────────────────────────────────────────────────────────────

class TestUpdateWeight:
    def test_sets_new_weight(self, conn: sqlite3.Connection) -> None:
        row_id = insert_event(conn, make_event())
        update_weight(conn, row_id, 0.42)
        stored = get_event_by_id(conn, row_id)
        assert stored is not None
        assert stored.weight == pytest.approx(0.42)

    def test_clamps_negative_to_zero(self, conn: sqlite3.Connection) -> None:
        row_id = insert_event(conn, make_event())
        update_weight(conn, row_id, -5.0)
        stored = get_event_by_id(conn, row_id)
        assert stored is not None
        assert stored.weight == pytest.approx(0.0)


# NOTE: the legacy `apply_weight_decay` (and its tests) were removed — it was
# uncalled in production and archived protected types. The live decay behavior
# is covered by tests/unit/delta/test_decay.py (apply_decay_pass) and
# tests/unit/delta/test_merge.py (_apply_decay_inner via execute_merge).


# ── get_events_for_projection ─────────────────────────────────────────────────

class TestGetEventsForProjection:
    def test_returns_active_events(self, conn: sqlite3.Connection) -> None:
        for i in range(3):
            insert_event(conn, make_event(content_hash=f"h{i}"))
        events = get_events_for_projection(conn, "proj1")
        assert len(events) == 3

    def test_excludes_archived(self, conn: sqlite3.Connection) -> None:
        row_id = insert_event(conn, make_event(content_hash="h1"))
        insert_event(conn, make_event(content_hash="h2"))
        mark_archived(conn, row_id)
        events = get_events_for_projection(conn, "proj1")
        assert len(events) == 1

    def test_excludes_superseded(self, conn: sqlite3.Connection) -> None:
        id1 = insert_event(conn, make_event(content_hash="old"))
        id2 = insert_event(conn, make_event(content_hash="new"))
        mark_superseded(conn, id1, by_id=id2)
        events = get_events_for_projection(conn, "proj1")
        assert all(e.id != id1 for e in events)
        assert any(e.id == id2 for e in events)

    def test_respects_after_id_for_delta(self, conn: sqlite3.Connection) -> None:
        for i in range(5):
            insert_event(conn, make_event(content_hash=f"h{i}"))
        all_events = get_events_for_projection(conn, "proj1")
        cutoff = all_events[1].id
        delta = get_events_for_projection(conn, "proj1", after_id=cutoff)
        assert len(delta) == 3
        assert all(e.id > cutoff for e in delta)

    def test_ordered_by_id_ascending(self, conn: sqlite3.Connection) -> None:
        for i in range(5):
            insert_event(conn, make_event(content_hash=f"h{i}"))
        events = get_events_for_projection(conn, "proj1")
        ids = [e.id for e in events]
        assert ids == sorted(ids)

    def test_empty_for_unknown_project(self, conn: sqlite3.Connection) -> None:
        insert_event(conn, make_event(project_id="proj1"))
        events = get_events_for_projection(conn, "other_project")
        assert events == []


# ── get_events_by_session ─────────────────────────────────────────────────────

class TestGetEventsBySession:
    def test_returns_only_matching_session(self, conn: sqlite3.Connection) -> None:
        insert_event(conn, make_event(session_id="s1", content_hash="h1"))
        insert_event(conn, make_event(session_id="s2", content_hash="h2"))
        events = get_events_by_session(conn, "proj1", "s1")
        assert len(events) == 1
        assert events[0].session_id == "s1"


# ── get_events_by_type ────────────────────────────────────────────────────────

class TestGetEventsByType:
    def test_returns_only_matching_type(self, conn: sqlite3.Connection) -> None:
        insert_event(conn, make_event(event_type="DECISION", content_hash="h1"))
        insert_event(conn, make_event(event_type="CONSTRAINT_HARD", content_hash="h2"))
        events = get_events_by_type(conn, "proj1", "DECISION")
        assert all(e.event_type == "DECISION" for e in events)
        assert len(events) == 1

    def test_excludes_archived_by_default(self, conn: sqlite3.Connection) -> None:
        row_id = insert_event(conn, make_event(event_type="DECISION", content_hash="h1"))
        mark_archived(conn, row_id)
        events = get_events_by_type(conn, "proj1", "DECISION")
        assert len(events) == 0

    def test_includes_archived_when_requested(self, conn: sqlite3.Connection) -> None:
        row_id = insert_event(conn, make_event(event_type="DECISION", content_hash="h1"))
        mark_archived(conn, row_id)
        events = get_events_by_type(conn, "proj1", "DECISION", include_archived=True)
        assert len(events) == 1


# ── get_max_event_id ──────────────────────────────────────────────────────────

class TestGetMaxEventId:
    def test_returns_zero_for_empty_project(self, conn: sqlite3.Connection) -> None:
        assert get_max_event_id(conn, "ghost") == 0

    def test_returns_highest_id(self, conn: sqlite3.Connection) -> None:
        ids = [insert_event(conn, make_event(content_hash=f"h{i}")) for i in range(4)]
        assert get_max_event_id(conn, "proj1") == max(ids)


# ── insert_extraction_failure ─────────────────────────────────────────────────

class TestInsertExtractionFailure:
    def test_creates_failure_record(self, conn: sqlite3.Connection) -> None:
        insert_extraction_failure(
            conn, "proj1", "sess1", "extraction",
            "Bad UTF-8 in transcript", "/tmp/transcript.md",
        )
        row = conn.execute("SELECT * FROM extraction_failures").fetchone()
        assert row["project_id"] == "proj1"
        assert row["stage"] == "extraction"
        assert row["error_message"] == "Bad UTF-8 in transcript"
        assert row["raw_input_path"] == "/tmp/transcript.md"
        assert row["retry_count"] == 0

    def test_multiple_failures_recorded_independently(self, conn: sqlite3.Connection) -> None:
        for i in range(3):
            insert_extraction_failure(
                conn, "proj1", "sess1", "extraction", f"error {i}", "/tmp/t.md"
            )
        count = conn.execute("SELECT COUNT(*) FROM extraction_failures").fetchone()[0]
        assert count == 3
