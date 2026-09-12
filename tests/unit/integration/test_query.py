"""CK-2 — agent-facing memory queries (recall / find_related backing)."""
from __future__ import annotations

import json
import sqlite3

import pytest

from cognikernel.integration.query import _lexical_recall, find_related_memory, recall_memory
from cognikernel.storage.migrations import run_migrations


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    run_migrations(c)
    yield c
    c.close()


def _insert(conn, eid, desc, etype="DECISION", archived=0, superseded_by=None):
    conn.execute(
        """INSERT INTO events (id, project_id, session_id, created_at, event_type,
                               payload, content_hash, weight, mention_count, archived, superseded_by)
           VALUES (?, 'p1', 's', ?, ?, ?, ?, 1.0, 1, ?, ?)""",
        (eid, eid, etype, json.dumps({"description": desc}), f"h{eid}", archived, superseded_by),
    )
    conn.commit()


class TestLexicalRecall:
    def test_ranks_most_overlapping_first(self, conn: sqlite3.Connection) -> None:
        _insert(conn, 1, "Use PostgreSQL for the primary database")
        _insert(conn, 2, "Frontend uses React and TypeScript")
        hits = _lexical_recall(conn, "p1", "which database do we use", 8)
        assert hits and hits[0]["id"] == 1

    def test_stopword_only_query_returns_empty(self, conn: sqlite3.Connection) -> None:
        _insert(conn, 1, "anything at all")
        assert _lexical_recall(conn, "p1", "the a an of", 8) == []

    def test_excludes_archived_and_superseded(self, conn: sqlite3.Connection) -> None:
        _insert(conn, 1, "database postgresql primary", archived=1)
        _insert(conn, 2, "database postgresql replica", superseded_by=99)
        assert _lexical_recall(conn, "p1", "database postgresql", 8) == []

    def test_respects_limit(self, conn: sqlite3.Connection) -> None:
        for i in range(1, 6):
            _insert(conn, i, f"database decision number {i}")
        assert len(_lexical_recall(conn, "p1", "database decision", 2)) == 2


def test_recall_memory_missing_project_returns_message(tmp_path) -> None:
    msg = recall_memory(str(tmp_path / "no_such_project"), "anything")
    assert "No CogniKernel memory" in msg


def test_recall_memory_end_to_end_lexical(tmp_path, monkeypatch) -> None:
    """Full wrapper, forced down the deterministic lexical path (model-independent)."""
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path))
    monkeypatch.setattr("cognikernel.embedding.model.is_ready", lambda: False)

    from cognikernel.config import Config
    from cognikernel.integration.session import init_project
    from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path

    proj = str(tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    init_project(proj)
    pid = hash_project_path(proj)
    db = get_db_path(Config.load(project_path=proj), pid)
    with get_connection(db) as c:
        c.execute(
            """INSERT INTO events (project_id, session_id, created_at, event_type,
                                   payload, content_hash, weight, mention_count)
               VALUES (?, 's', 1, 'DECISION', ?, 'h1', 1.0, 1)""",
            (pid, json.dumps({"description": "Use PostgreSQL for the primary database"})),
        )
        c.commit()

    out = recall_memory(proj, "which database")
    assert "PostgreSQL" in out


def _two_session_project(tmp_path, monkeypatch):
    """A project with one claim from each of two sessions, captured a day apart."""
    from datetime import datetime

    from cognikernel.config import Config
    from cognikernel.integration.session import init_project
    from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
    from cognikernel.storage.evidence import store_evidence

    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path))
    monkeypatch.setattr("cognikernel.embedding.model.is_ready", lambda: False)
    proj = str(tmp_path / "proj")
    (tmp_path / "proj").mkdir()
    init_project(proj)
    pid = hash_project_path(proj)
    first, second = "3f2a9c1e-first-session", "8b7d4e20-second-session"
    ids = {}
    with get_connection(get_db_path(Config.load(project_path=proj), pid)) as c:
        for sid, day, key, desc in (
            (first, 11, "primary", "Use PostgreSQL for the primary database"),
            (first, 11, "weekly", "Back up the PostgreSQL database weekly"),
            (second, 12, "nightly", "Back up the PostgreSQL database nightly"),
        ):
            at = int(datetime(2026, 9, day, 10, 0).timestamp() * 1000)
            if key != "weekly":
                store_evidence(c, pid, sid, "transcript", desc.encode(), captured_at=at)
            cur = c.execute(
                """INSERT INTO events (project_id, session_id, created_at, event_type,
                                       payload, content_hash, weight, mention_count)
                   VALUES (?, ?, ?, 'DECISION', ?, ?, 1.0, 1)""",
                (pid, sid, at, json.dumps({"description": desc}), f"h-{key}"),
            )
            ids[key] = cur.lastrowid
        # The nightly backup replaced the weekly one: its value changed over time.
        c.execute("UPDATE events SET superseded_by = ? WHERE id = ?", (ids["nightly"], ids["weekly"]))
        c.commit()
    return proj, first, second, ids


def test_recall_labels_a_claim_whose_value_changed_and_nothing_else(tmp_path, monkeypatch) -> None:
    """A label goes where the store holds an earlier value of the same thing."""
    proj, first, second, _ = _two_session_project(tmp_path, monkeypatch)

    out = recall_memory(proj, "postgresql database")

    primary = next(line for line in out.splitlines() if "primary database" in line)
    nightly = next(line for line in out.splitlines() if "nightly" in line)
    assert "S2 · 09-12" in nightly
    assert "S1" not in primary
    assert first not in out and second not in out


def test_find_related_labels_a_changed_claim_too(tmp_path, monkeypatch) -> None:
    proj, first, second, ids = _two_session_project(tmp_path, monkeypatch)
    monkeypatch.setattr(
        "cognikernel.embedding.retrieval.find_related",
        lambda conn, project_id, event_id, k=8: [{"id": ids["nightly"], "why": "semantic", "score": 0.8},
                                                  {"id": ids["primary"], "why": "semantic", "score": 0.6}],
    )

    out = find_related_memory(proj, "primary database")

    related = [line for line in out.splitlines() if line.startswith("- ")]
    assert any("nightly" in line and "S2 · 09-12" in line for line in related)
    assert all("S1" not in line for line in related if "primary database" in line)
    assert second not in out


def test_query_functions_never_raise(tmp_path, monkeypatch) -> None:
    """Both entrypoints return a string even when the underlying call blows up."""
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path))
    # Missing project → safe message, not an exception.
    assert isinstance(recall_memory(str(tmp_path / "x"), "q"), str)
    assert isinstance(find_related_memory(str(tmp_path / "x"), "q"), str)
