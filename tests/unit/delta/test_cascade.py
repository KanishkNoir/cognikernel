"""Tests for cognikernel.delta.cascade."""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any

import pytest

from cognikernel.delta.cascade import cascade_component_status
from cognikernel.storage.events import Event


# ── helpers ───────────────────────────────────────────────────────────────────

def make_status_event(**overrides: Any) -> Event:
    defaults: dict[str, Any] = {
        "project_id": "proj1",
        "session_id": "sess1",
        "event_type": "COMPONENT_STATUS",
        "payload": {
            "path": "src/api.py",
            "status": "blocked",
            "reason": "waiting on auth",
        },
        "content_hash": "status_hash",
        "weight": 0.8,
    }
    defaults.update(overrides)
    e = Event(**defaults)
    return e


def insert_component_event(
    conn: sqlite3.Connection,
    path: str,
    status: str,
    dependencies: list[str],
    content_hash: str = "dep_hash",
    project_id: str = "proj1",
    extra_payload: dict | None = None,
) -> int:
    payload: dict[str, Any] = {
        "path": path,
        "status": status,
        "dependencies": dependencies,
    }
    if extra_payload:
        payload.update(extra_payload)
    cursor = conn.execute(
        """
        INSERT INTO events
            (project_id, session_id, created_at, event_type,
             payload, content_hash, weight, mention_count)
        VALUES (?, 'sess1', ?, 'COMPONENT_STATUS', ?, ?, 0.8, 1)
        """,
        (project_id, int(time.time() * 1000),
         json.dumps(payload, sort_keys=True, separators=(",", ":")),
         content_hash),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def set_event_id(event: Event, row_id: int) -> Event:
    event.id = row_id
    return event


# ── cascade_component_status ──────────────────────────────────────────────────

class TestCascadeComponentStatus:
    def test_non_cascade_status_returns_zero(self, conn: sqlite3.Connection) -> None:
        e = make_status_event(payload={"path": "a.py", "status": "stable"})
        e.id = 1
        assert cascade_component_status(conn, e) == 0

    def test_in_progress_status_returns_zero(self, conn: sqlite3.Connection) -> None:
        e = make_status_event(payload={"path": "a.py", "status": "in_progress"})
        e.id = 1
        assert cascade_component_status(conn, e) == 0

    def test_cascaded_from_guard_returns_zero(self, conn: sqlite3.Connection) -> None:
        e = make_status_event(payload={
            "path": "a.py",
            "status": "blocked",
            "cascaded_from": 99,
        })
        e.id = 1
        assert cascade_component_status(conn, e) == 0

    def test_no_target_path_returns_zero(self, conn: sqlite3.Connection) -> None:
        e = make_status_event(payload={"status": "blocked"})
        e.id = 1
        assert cascade_component_status(conn, e) == 0

    def test_no_dependents_returns_zero(self, conn: sqlite3.Connection) -> None:
        e = make_status_event(payload={"path": "src/api.py", "status": "blocked"})
        e.id = 1
        assert cascade_component_status(conn, e) == 0

    def test_blocked_cascades_to_dependents(self, conn: sqlite3.Connection) -> None:
        insert_component_event(
            conn, "src/router.py", "stable", ["src/api.py"], "dep1"
        )
        e = make_status_event(
            payload={"path": "src/api.py", "status": "blocked"},
            content_hash="trig1",
        )
        e.id = 99
        count = cascade_component_status(conn, e)
        assert count == 1

    def test_abandoned_cascades_to_dependents(self, conn: sqlite3.Connection) -> None:
        insert_component_event(
            conn, "src/router.py", "stable", ["src/api.py"], "dep1"
        )
        e = make_status_event(
            payload={"path": "src/api.py", "status": "abandoned"},
            content_hash="trig1",
        )
        e.id = 99
        count = cascade_component_status(conn, e)
        assert count == 1

    def test_cascade_inserts_needs_review_event(self, conn: sqlite3.Connection) -> None:
        insert_component_event(
            conn, "src/router.py", "stable", ["src/api.py"], "dep1"
        )
        e = make_status_event(
            payload={"path": "src/api.py", "status": "blocked"},
            content_hash="trig1",
        )
        e.id = 99
        cascade_component_status(conn, e)
        row = conn.execute(
            "SELECT payload FROM events WHERE event_type = 'COMPONENT_STATUS' AND id != 1"
        ).fetchone()
        assert row is not None
        payload = json.loads(row["payload"])
        assert payload["status"] == "needs_review"
        assert payload["path"] == "src/router.py"
        assert "cascaded_from" in payload

    def test_cascade_sets_cascaded_from(self, conn: sqlite3.Connection) -> None:
        insert_component_event(
            conn, "src/router.py", "stable", ["src/api.py"], "dep1"
        )
        e = make_status_event(
            payload={"path": "src/api.py", "status": "blocked"},
            content_hash="trig1",
        )
        e.id = 99
        cascade_component_status(conn, e)
        rows = conn.execute(
            "SELECT payload FROM events WHERE event_type = 'COMPONENT_STATUS' AND id > 1"
        ).fetchall()
        assert len(rows) == 1
        payload = json.loads(rows[0]["payload"])
        assert payload["cascaded_from"] == 99

    def test_cascade_does_not_recurse(self, conn: sqlite3.Connection) -> None:
        """A cascaded event (has cascaded_from) must not trigger further cascade."""
        insert_component_event(
            conn, "src/router.py", "stable", ["src/middleware.py"], "dep1"
        )
        # This event already has cascaded_from — simulates a cascade event
        e = make_status_event(
            payload={
                "path": "src/middleware.py",
                "status": "blocked",
                "cascaded_from": 42,
            },
            content_hash="trig1",
        )
        e.id = 55
        count = cascade_component_status(conn, e)
        assert count == 0

    def test_cascade_skips_self_dependency(self, conn: sqlite3.Connection) -> None:
        """A file listing itself as a dependency should not cascade to itself."""
        insert_component_event(
            conn, "src/api.py", "stable", ["src/api.py"], "dep1"
        )
        e = make_status_event(
            payload={"path": "src/api.py", "status": "blocked"},
            content_hash="trig1",
        )
        e.id = 99
        count = cascade_component_status(conn, e)
        assert count == 0

    def test_cascade_ignores_archived_dependents(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO events
                (project_id, session_id, created_at, event_type,
                 payload, content_hash, weight, mention_count, archived)
            VALUES ('proj1', 'sess1', 0, 'COMPONENT_STATUS',
                    '{"path":"src/router.py","status":"stable","dependencies":["src/api.py"]}',
                    'dep1', 0.8, 1, 1)
            """
        )
        conn.commit()
        e = make_status_event(
            payload={"path": "src/api.py", "status": "blocked"},
            content_hash="trig1",
        )
        e.id = 99
        count = cascade_component_status(conn, e)
        assert count == 0

    def test_cascade_ignores_superseded_dependents(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO events
                (project_id, session_id, created_at, event_type,
                 payload, content_hash, weight, mention_count, superseded_by)
            VALUES ('proj1', 'sess1', 0, 'COMPONENT_STATUS',
                    '{"path":"src/router.py","status":"stable","dependencies":["src/api.py"]}',
                    'dep1', 0.8, 1, 77)
            """
        )
        conn.commit()
        e = make_status_event(
            payload={"path": "src/api.py", "status": "blocked"},
            content_hash="trig1",
        )
        e.id = 99
        count = cascade_component_status(conn, e)
        assert count == 0

    def test_dedup_increments_mention_count(self, conn: sqlite3.Connection) -> None:
        """Calling cascade twice for the same trigger should update mention_count."""
        insert_component_event(
            conn, "src/router.py", "stable", ["src/api.py"], "dep1"
        )
        e = make_status_event(
            payload={"path": "src/api.py", "status": "blocked"},
            content_hash="trig1",
        )
        e.id = 99
        cascade_component_status(conn, e)
        # Second call — same hash already inserted; should increment mention_count
        cascade_component_status(conn, e)
        row = conn.execute(
            "SELECT mention_count FROM events WHERE event_type = 'COMPONENT_STATUS' AND id > 1"
        ).fetchone()
        assert row["mention_count"] == 2

    def test_cascade_multiple_dependents(self, conn: sqlite3.Connection) -> None:
        insert_component_event(conn, "src/router.py", "stable", ["src/api.py"], "dep1")
        insert_component_event(conn, "src/auth.py", "stable", ["src/api.py"], "dep2")
        e = make_status_event(
            payload={"path": "src/api.py", "status": "blocked"},
            content_hash="trig1",
        )
        e.id = 99
        count = cascade_component_status(conn, e)
        assert count == 2

    def test_cascade_ignores_different_project(self, conn: sqlite3.Connection) -> None:
        insert_component_event(
            conn, "src/router.py", "stable", ["src/api.py"], "dep1",
            project_id="other_proj"
        )
        e = make_status_event(
            project_id="proj1",
            payload={"path": "src/api.py", "status": "blocked"},
            content_hash="trig1",
        )
        e.id = 99
        count = cascade_component_status(conn, e)
        assert count == 0
