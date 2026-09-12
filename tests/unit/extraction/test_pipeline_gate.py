"""The gate is wired into the one function every extraction path reaches.

Uses the shared `conn` fixture from tests/conftest.py.
"""
import sqlite3

from cognikernel.extraction.pipeline import SessionMetadata, persist_events
from cognikernel.model import Event
from cognikernel.quality.gate import GroundingContext
from cognikernel.storage.quality_telemetry import get_rule_counts


def _event(event_type: str, description: str, **payload) -> Event:
    return Event(
        project_id="p",
        session_id="s",
        event_type=event_type,
        payload={"description": description, **payload},
        content_hash=(description[:32] or "h"),
        weight=1.0,
    )


_META = SessionMetadata(project_id="p", session_id="s", started_at=0, ended_at=0)


class TestGateWiring:
    def test_rejected_event_is_not_stored(self, conn: sqlite3.Connection) -> None:
        ids = persist_events(
            [_event("CONSTRAINT_HARD",
                    "Pick up the last task as if the break never happened.")],
            conn, _META,
        )
        assert ids == []
        assert conn.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0

    def test_clean_event_is_stored(self, conn: sqlite3.Connection) -> None:
        ids = persist_events(
            [_event("DECISION", "The dispatcher retries twice.")], conn, _META
        )
        assert len(ids) == 1

    def test_rejection_is_counted(self, conn: sqlite3.Connection) -> None:
        persist_events(
            [_event("CONSTRAINT_HARD",
                    "Pick up the last task as if the break never happened.")],
            conn, _META,
        )
        counts = get_rule_counts(conn, "p")
        assert any(c["rule_id"] == "D4" for c in counts)

    def test_subject_less_event_is_stored_but_demoted(self, conn: sqlite3.Connection) -> None:
        # D7 downgrades rather than rejects: still recallable. The stored weight
        # below is not what ranks it; the ranking's quality factor reads the
        # marker (tests/unit/storage/test_demotes_reach_ranking.py).
        ids = persist_events(
            [_event("DECISION", "It must not take down the pipeline.")], conn, _META
        )
        assert len(ids) == 1
        row = conn.execute("SELECT payload, weight FROM events").fetchone()
        assert "context_dependent" in row[0]
        assert row[1] < 1.0

    def test_unknown_path_is_downgraded_not_dropped(self, conn: sqlite3.Connection) -> None:
        ground = GroundingContext(frozenset({"src/known.py"}))
        ids = persist_events(
            [_event("COMPONENT_STATUS", "x", path="src/brand_new.py")],
            conn, _META, ground=ground,
        )
        assert len(ids) == 1
        row = conn.execute("SELECT payload, weight FROM events").fetchone()
        assert "unverified" in row[0]
        assert row[1] < 1.0

    def test_legacy_call_without_ground_still_works(self, conn: sqlite3.Connection) -> None:
        # Backwards compatibility: existing call sites pass no `ground`.
        ids = persist_events([_event("DECISION", "The queue is durable.")], conn, _META)
        assert len(ids) == 1
