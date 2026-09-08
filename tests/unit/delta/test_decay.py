"""Tests for cognikernel.delta.decay."""
from __future__ import annotations

import sqlite3

import pytest

from cognikernel.delta.decay import ARCHIVE_THRESHOLD, DECAY_FACTOR, apply_decay_pass
from cognikernel.storage.events import insert_event, Event


# ── helpers ───────────────────────────────────────────────────────────────────

def make_event(
    conn: sqlite3.Connection,
    content_hash: str,
    weight: float = 1.0,
    session_id: str = "sess_old",
    event_type: str = "DECISION",
    project_id: str = "proj1",
) -> int:
    payload: dict = {"description": f"event {content_hash}"}
    if event_type == "COMPONENT_STATUS":
        payload = {"path": "x.py", "status": "stable", "description": content_hash}
    e = Event(
        project_id=project_id,
        session_id=session_id,
        event_type=event_type,
        payload=payload,
        content_hash=content_hash,
        weight=weight,
    )
    return insert_event(conn, e)


# ── constants ─────────────────────────────────────────────────────────────────

class TestDecayConstants:
    def test_decay_factor(self) -> None:
        assert DECAY_FACTOR == pytest.approx(0.92)

    def test_archive_threshold(self) -> None:
        assert ARCHIVE_THRESHOLD == pytest.approx(0.05)


# ── apply_decay_pass ──────────────────────────────────────────────────────────

class TestApplyDecayPass:
    def test_returns_zero_when_no_events(self, conn: sqlite3.Connection) -> None:
        result = apply_decay_pass(conn, "proj1", "sess_new")
        assert result == 0

    def test_applies_decay_to_older_sessions(self, conn: sqlite3.Connection) -> None:
        make_event(conn, "h1", weight=1.0, session_id="sess_old")
        apply_decay_pass(conn, "proj1", "sess_new")
        row = conn.execute("SELECT weight FROM events WHERE content_hash = 'h1'").fetchone()
        assert row["weight"] == pytest.approx(1.0 * DECAY_FACTOR)

    def test_does_not_decay_current_session(self, conn: sqlite3.Connection) -> None:
        make_event(conn, "h1", weight=1.0, session_id="sess_new")
        apply_decay_pass(conn, "proj1", "sess_new")
        row = conn.execute("SELECT weight FROM events WHERE content_hash = 'h1'").fetchone()
        assert row["weight"] == pytest.approx(1.0)

    def test_archives_below_threshold(self, conn: sqlite3.Connection) -> None:
        make_event(conn, "h1", weight=0.04, session_id="sess_old")
        apply_decay_pass(conn, "proj1", "sess_new")
        row = conn.execute("SELECT archived FROM events WHERE content_hash = 'h1'").fetchone()
        assert row["archived"] == 1

    def test_does_not_archive_above_threshold(self, conn: sqlite3.Connection) -> None:
        make_event(conn, "h1", weight=1.0, session_id="sess_old")
        apply_decay_pass(conn, "proj1", "sess_new")
        row = conn.execute("SELECT archived FROM events WHERE content_hash = 'h1'").fetchone()
        assert row["archived"] == 0

    def test_returns_archived_count(self, conn: sqlite3.Connection) -> None:
        make_event(conn, "h1", weight=0.03, session_id="sess_old")
        make_event(conn, "h2", weight=0.02, session_id="sess_old")
        make_event(conn, "h3", weight=1.0, session_id="sess_old")
        result = apply_decay_pass(conn, "proj1", "sess_new")
        assert result == 2

    def test_protects_constraint_hard_from_archive(self, conn: sqlite3.Connection) -> None:
        make_event(conn, "h1", weight=0.01, session_id="sess_old",
                   event_type="CONSTRAINT_HARD")
        apply_decay_pass(conn, "proj1", "sess_new")
        row = conn.execute("SELECT archived FROM events WHERE content_hash = 'h1'").fetchone()
        assert row["archived"] == 0

    def test_protects_approach_abandoned_do_not_retry_from_archive(
        self, conn: sqlite3.Connection
    ) -> None:
        make_event(conn, "h1", weight=0.01, session_id="sess_old",
                   event_type="APPROACH_ABANDONED_DO_NOT_RETRY")
        apply_decay_pass(conn, "proj1", "sess_new")
        row = conn.execute("SELECT archived FROM events WHERE content_hash = 'h1'").fetchone()
        assert row["archived"] == 0

    def test_idempotency_same_session_no_double_decay(self, conn: sqlite3.Connection) -> None:
        make_event(conn, "h1", weight=1.0, session_id="sess_old")
        apply_decay_pass(conn, "proj1", "sess_new")
        weight_after_first = conn.execute(
            "SELECT weight FROM events WHERE content_hash = 'h1'"
        ).fetchone()["weight"]
        apply_decay_pass(conn, "proj1", "sess_new")  # same session — should no-op
        weight_after_second = conn.execute(
            "SELECT weight FROM events WHERE content_hash = 'h1'"
        ).fetchone()["weight"]
        assert weight_after_first == pytest.approx(weight_after_second)

    def test_idempotency_returns_zero_on_repeat(self, conn: sqlite3.Connection) -> None:
        make_event(conn, "h1", weight=1.0, session_id="sess_old")
        apply_decay_pass(conn, "proj1", "sess_new")
        result = apply_decay_pass(conn, "proj1", "sess_new")
        assert result == 0

    def test_different_session_applies_fresh_decay(self, conn: sqlite3.Connection) -> None:
        make_event(conn, "h1", weight=1.0, session_id="sess_old")
        apply_decay_pass(conn, "proj1", "sess_a")
        w1 = conn.execute("SELECT weight FROM events WHERE content_hash='h1'").fetchone()["weight"]
        apply_decay_pass(conn, "proj1", "sess_b")
        w2 = conn.execute("SELECT weight FROM events WHERE content_hash='h1'").fetchone()["weight"]
        assert w2 == pytest.approx(w1 * DECAY_FACTOR)

    def test_already_archived_events_not_redecayed(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO events
                (project_id, session_id, created_at, event_type,
                 payload, content_hash, weight, mention_count, archived)
            VALUES ('proj1', 'sess_old', 0, 'DECISION',
                    '{"description":"old"}', 'h1', 1.0, 1, 1)
            """
        )
        conn.commit()
        apply_decay_pass(conn, "proj1", "sess_new")
        row = conn.execute("SELECT weight FROM events WHERE content_hash='h1'").fetchone()
        assert row["weight"] == pytest.approx(1.0)  # archived — weight unchanged

    def test_does_not_touch_other_projects(self, conn: sqlite3.Connection) -> None:
        make_event(conn, "h1", weight=1.0, session_id="sess_old", project_id="other_proj")
        apply_decay_pass(conn, "proj1", "sess_new")
        row = conn.execute("SELECT weight FROM events WHERE content_hash = 'h1'").fetchone()
        assert row["weight"] == pytest.approx(1.0)

    def test_weight_floor_at_zero(self, conn: sqlite3.Connection) -> None:
        make_event(conn, "h1", weight=0.001, session_id="sess_old")
        apply_decay_pass(conn, "proj1", "sess_new")
        row = conn.execute("SELECT weight FROM events WHERE content_hash = 'h1'").fetchone()
        assert row["weight"] >= 0.0

    def test_meta_row_written_for_idempotency(self, conn: sqlite3.Connection) -> None:
        apply_decay_pass(conn, "proj1", "sess_new")
        row = conn.execute(
            "SELECT value FROM meta WHERE key = 'last_decay_session:proj1'"
        ).fetchone()
        assert row is not None
        assert row["value"] == "sess_new"


class TestArchivedAt:
    """T-103 (#13): archived_at is set on exactly the rows this pass flips to
    archived=1 -- not on survivors, not on rows already archived earlier."""

    def test_set_on_exactly_the_rows_archived_this_pass(
        self, conn: sqlite3.Connection
    ) -> None:
        import time
        make_event(conn, "archived-h", weight=0.01, session_id="sess_old")
        make_event(conn, "survivor-h", weight=1.0, session_id="sess_old")
        before = int(time.time() * 1000)

        apply_decay_pass(conn, "proj1", "sess_new")

        after = int(time.time() * 1000)
        archived_row = conn.execute(
            "SELECT archived, archived_at FROM events WHERE content_hash = 'archived-h'"
        ).fetchone()
        survivor_row = conn.execute(
            "SELECT archived, archived_at FROM events WHERE content_hash = 'survivor-h'"
        ).fetchone()

        assert archived_row["archived"] == 1
        assert before <= archived_row["archived_at"] <= after
        assert survivor_row["archived"] == 0
        assert survivor_row["archived_at"] is None

    def test_already_archived_row_keeps_its_original_timestamp(
        self, conn: sqlite3.Connection
    ) -> None:
        """The bulk UPDATE filters archived = 0, so a row archived in an
        earlier pass must not have its archived_at overwritten by a later
        one -- that would make 'when was this archived' answer the wrong
        question for anything archived more than once-worth of history ago."""
        make_event(conn, "already-h", weight=0.01, session_id="sess_old")
        apply_decay_pass(conn, "proj1", "sess_new_1")
        first_stamp = conn.execute(
            "SELECT archived_at FROM events WHERE content_hash = 'already-h'"
        ).fetchone()["archived_at"]
        assert first_stamp is not None

        apply_decay_pass(conn, "proj1", "sess_new_2")
        second_stamp = conn.execute(
            "SELECT archived_at FROM events WHERE content_hash = 'already-h'"
        ).fetchone()["archived_at"]
        assert second_stamp == first_stamp
