"""The quality gate must run on the REAL write path.

execute_merge is what session_end, process_jobs and rebuild_from_raw all call.
persist_events has no production callers at all — the plan asserted it was "the
one function every extraction path reaches", which was wrong. These tests pin
the gate to the path that actually stores events.
"""
import sqlite3

from cognikernel.delta.merge import execute_merge
from cognikernel.model import Event
from cognikernel.storage.quality_telemetry import get_rule_counts


def _event(event_type: str, description: str, chash: str, **payload) -> Event:
    return Event(
        project_id="p",
        session_id="s",
        event_type=event_type,
        payload={"description": description, **payload},
        content_hash=chash,
        weight=1.0,
    )


def _stored(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]


class TestGateOnMergePath:
    def test_boilerplate_is_not_merged(self, conn: sqlite3.Connection) -> None:
        execute_merge(conn, "s", [
            _event("CONSTRAINT_HARD",
                   "Pick up the last task as if the break never happened.", "a"),
        ])
        assert _stored(conn) == 0

    def test_interrogative_constraint_is_not_merged(self, conn: sqlite3.Connection) -> None:
        execute_merge(conn, "s", [
            _event("CONSTRAINT_HARD", "Should we just bring in Celery?", "b"),
        ])
        assert _stored(conn) == 0

    def test_clean_event_is_merged(self, conn: sqlite3.Connection) -> None:
        stats = execute_merge(conn, "s", [
            _event("DECISION", "The dispatcher retries twice.", "c"),
        ])
        assert _stored(conn) == 1
        assert stats["inserted"] == 1

    def test_subject_less_is_merged_but_demoted(self, conn: sqlite3.Connection) -> None:
        execute_merge(conn, "s", [
            _event("DECISION", "It must not take down the pipeline.", "d"),
        ])
        row = conn.execute("SELECT payload, weight FROM events").fetchone()
        assert row is not None
        assert "context_dependent" in row[0]
        assert row[1] < 1.0

    def test_rejection_is_counted_in_telemetry(self, conn: sqlite3.Connection) -> None:
        execute_merge(conn, "s", [
            _event("CONSTRAINT_HARD",
                   "Pick up the last task as if the break never happened.", "e"),
        ])
        assert any(c["rule_id"] == "D4" for c in get_rule_counts(conn, "p"))

    def test_all_rejected_still_returns_valid_stats(self, conn: sqlite3.Connection) -> None:
        stats = execute_merge(conn, "s", [
            _event("CONSTRAINT_HARD", "Should we use Celery?", "f"),
        ])
        assert stats["inserted"] == 0
        assert set(stats) >= {"inserted", "updated", "superseded", "cascaded", "archived"}

    def test_mixed_batch_keeps_the_good_one(self, conn: sqlite3.Connection) -> None:
        execute_merge(conn, "s", [
            _event("CONSTRAINT_HARD", "Should we use Celery?", "g"),
            _event("DECISION", "Use Postgres for the queue.", "h"),
        ])
        assert _stored(conn) == 1
        row = conn.execute("SELECT payload FROM events").fetchone()
        assert "Postgres" in row[0]
