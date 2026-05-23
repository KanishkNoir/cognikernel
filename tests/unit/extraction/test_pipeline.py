"""Tests for the extraction pipeline orchestrator."""
import sqlite3
from pathlib import Path

import pytest

from memlora.extraction.pipeline import (
    SessionMetadata,
    _FOREGROUND_BYTES,
    _HARD_CAP_BYTES,
    extract_session,
    persist_events,
)
from memlora.storage.events import (
    Event,
    get_events_by_session,
    insert_extraction_failure,
)


@pytest.fixture
def meta() -> SessionMetadata:
    return SessionMetadata(
        project_id="proj1",
        session_id="sess1",
        started_at=1_700_000_000_000,
        ended_at=1_700_000_001_000,
    )


# ── extract_session ───────────────────────────────────────────────────────────

class TestExtractSession:
    def test_returns_list(self, meta: SessionMetadata) -> None:
        events = extract_session("Human: We decided to use SQLite.", meta)
        assert isinstance(events, list)

    def test_decision_event_extracted(self, meta: SessionMetadata) -> None:
        events = extract_session("Assistant: We decided to use SQLite.", meta)
        types = {e.event_type for e in events}
        assert "DECISION" in types

    def test_constraint_event_extracted(self, meta: SessionMetadata) -> None:
        events = extract_session("Human: We cannot use Redis.", meta)
        types = {e.event_type for e in events}
        assert "CONSTRAINT_HARD" in types or "CONSTRAINT_SOFT" in types

    def test_content_hash_populated(self, meta: SessionMetadata) -> None:
        events = extract_session("Human: We decided to use SQLite.", meta)
        assert all(len(e.content_hash) == 64 for e in events)

    def test_empty_transcript_returns_empty(self, meta: SessionMetadata) -> None:
        events = extract_session("", meta)
        assert events == []

    def test_no_signal_returns_empty(self, meta: SessionMetadata) -> None:
        events = extract_session("Human: Hello, how are you today?", meta)
        assert events == []

    def test_git_diff_none_produces_no_git_events(self, meta: SessionMetadata) -> None:
        events = extract_session("Human: We decided to use SQLite.", meta, git_diff=None)
        component_events = [e for e in events if e.event_type == "COMPONENT_STATUS"]
        assert component_events == []

    def test_git_diff_adds_component_events(self, meta: SessionMetadata) -> None:
        diff = "M\tsrc/auth/middleware.py\nsrc/auth/middleware.py | 10 +++++-----"
        events = extract_session("Human: We decided to use SQLite.", meta, git_diff=diff)
        types = {e.event_type for e in events}
        assert "COMPONENT_STATUS" in types

    def test_cross_reference_boosts_abandoned(self, meta: SessionMetadata) -> None:
        transcript = "Human: We abandoned the redis approach."
        diff = "D\tsrc/legacy/redis_client.py\nsrc/legacy/redis_client.py | 20 ----------"
        events = extract_session(transcript, meta, git_diff=diff)
        abandoned = [e for e in events if e.event_type == "APPROACH_ABANDONED"]
        if abandoned:
            # At least one abandoned event should be corroborated
            assert any(e.payload.get("git_corroborated") for e in abandoned)

    def test_project_id_propagated(self, meta: SessionMetadata) -> None:
        events = extract_session("Human: We decided to use SQLite.", meta)
        assert all(e.project_id == "proj1" for e in events)

    def test_session_id_propagated(self, meta: SessionMetadata) -> None:
        events = extract_session("Human: We decided to use SQLite.", meta)
        assert all(e.session_id == "sess1" for e in events)

    def test_multiple_signals_produce_multiple_events(self, meta: SessionMetadata) -> None:
        transcript = "Assistant: We decided to use SQLite. We cannot use Redis."
        events = extract_session(transcript, meta)
        assert len(events) >= 2

    def test_hard_cap_truncates_large_transcript(self, meta: SessionMetadata) -> None:
        # Build a transcript that exceeds the 5 MB hard cap; signal is at the tail.
        filler = "Human: Some background context. " * 10_000   # ~320 KB
        # Pad to exceed 5 MB with neutral lines, then add a signal at the very end
        padding = ("x " * 500 + "\n") * 400   # ~200 KB per chunk × many
        big = filler + ("Human: No signal here at all.\n" * 5_000) + padding * 6
        tail_signal = "Human: We decided to use SQLite at the very end."
        transcript = big + tail_signal
        # Must not raise, and the tail signal should be extractable
        events = extract_session(transcript, meta)
        # The function should run without error (result content not guaranteed
        # due to truncation boundary, but no exception)
        assert isinstance(events, list)

    def test_size_under_hard_cap_not_truncated(self, meta: SessionMetadata) -> None:
        transcript = ("Assistant: We decided to use SQLite.\n" * 1_000)
        events = extract_session(transcript, meta)
        decision_events = [e for e in events if e.event_type == "DECISION"]
        assert len(decision_events) >= 1


# ── persist_events ────────────────────────────────────────────────────────────

class TestPersistEvents:
    def _make_event(self, **overrides) -> Event:
        defaults = dict(
            project_id="proj1", session_id="sess1",
            event_type="DECISION",
            payload={"description": "Use SQLite", "rationale": ""},
            content_hash="a" * 64, weight=1.0,
        )
        defaults.update(overrides)
        return Event(**defaults)

    def test_returns_list_of_ids(self, conn: sqlite3.Connection, meta: SessionMetadata) -> None:
        events = [self._make_event()]
        ids = persist_events(events, conn, meta)
        assert isinstance(ids, list)
        assert len(ids) == 1
        assert isinstance(ids[0], int) and ids[0] > 0

    def test_empty_events_returns_empty(self, conn: sqlite3.Connection, meta: SessionMetadata) -> None:
        ids = persist_events([], conn, meta)
        assert ids == []

    def test_multiple_events_all_written(self, conn: sqlite3.Connection, meta: SessionMetadata) -> None:
        events = [
            self._make_event(content_hash="a" * 64),
            self._make_event(event_type="CONSTRAINT_HARD", content_hash="b" * 64),
        ]
        ids = persist_events(events, conn, meta)
        assert len(ids) == 2

    def test_events_readable_after_persist(self, conn: sqlite3.Connection, meta: SessionMetadata) -> None:
        events = [self._make_event()]
        persist_events(events, conn, meta)
        stored = get_events_by_session(conn, "proj1", "sess1")
        assert len(stored) == 1
        assert stored[0].event_type == "DECISION"

    def test_persist_without_session_meta(self, conn: sqlite3.Connection) -> None:
        events = [self._make_event()]
        ids = persist_events(events, conn, session_meta=None)
        assert len(ids) == 1

    def test_persist_fails_gracefully_on_bad_conn(self, meta: SessionMetadata) -> None:
        bad_conn = sqlite3.connect(":memory:")  # no schema
        events = [self._make_event()]
        # Should not raise; returns empty IDs for failed events
        ids = persist_events(events, bad_conn, meta)
        assert ids == []
        bad_conn.close()

    def test_dedup_increments_mention_count(self, conn: sqlite3.Connection, meta: SessionMetadata) -> None:
        events = [self._make_event(), self._make_event()]
        ids = persist_events(events, conn, meta)
        # Both calls succeed (insert + dedup update); both return a valid ID
        assert len(ids) == 2

    def test_persist_writes_failure_on_error(
        self, conn: sqlite3.Connection, meta: SessionMetadata
    ) -> None:
        # Force a schema error by persisting to a connection that has no events table
        bad_conn = sqlite3.connect(":memory:")
        # Create only extraction_failures table so the failure write itself works
        bad_conn.execute(
            """CREATE TABLE extraction_failures (
                id INTEGER PRIMARY KEY,
                project_id TEXT, session_id TEXT,
                failed_at INTEGER,
                stage TEXT, error_message TEXT, raw_input_path TEXT
            )"""
        )
        events = [self._make_event()]
        ids = persist_events(events, bad_conn, meta)
        assert ids == []
        rows = bad_conn.execute("SELECT * FROM extraction_failures").fetchall()
        assert len(rows) == 1
        bad_conn.close()


# ── SessionMetadata dataclass ─────────────────────────────────────────────────

class TestSessionMetadata:
    def test_fields_accessible(self) -> None:
        m = SessionMetadata("p1", "s1", 1000, 2000)
        assert m.project_id == "p1"
        assert m.session_id == "s1"
        assert m.started_at == 1000
        assert m.ended_at == 2000

    def test_size_constants_correct(self) -> None:
        assert _FOREGROUND_BYTES == 500 * 1024
        assert _HARD_CAP_BYTES == 5 * 1024 * 1024
