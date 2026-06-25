"""Tests for semantic recall + symbol-graph fusion (E3/E5). Model-guarded."""
from __future__ import annotations

import json
import sqlite3

import pytest

from memlora.embedding.backfill import backfill_embeddings
from memlora.embedding.model import is_available
from memlora.embedding.retrieval import find_related, recall
from memlora.storage.migrations import run_migrations

pytestmark = pytest.mark.skipif(not is_available(), reason="embedding model not installed")


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    run_migrations(c)
    yield c
    c.close()


def _insert(conn, eid, etype, payload, project_id="p1"):
    conn.execute(
        """INSERT INTO events (id, project_id, session_id, created_at, event_type,
                               payload, content_hash, weight, mention_count)
           VALUES (?, ?, 's', ?, ?, ?, ?, 1.0, 1)""",
        (eid, project_id, eid, etype, json.dumps(payload), f"h{eid}"),
    )
    conn.commit()


class TestRecall:
    def test_recall_ranks_relevant_first(self, conn: sqlite3.Connection) -> None:
        _insert(conn, 1, "DECISION", {"description": "We will use argon2id for password hashing."})
        _insert(conn, 2, "DECISION", {"description": "All API routes use the /api/v1/ prefix."})
        _insert(conn, 3, "DECISION", {"description": "Use shadcn/ui for the component library."})
        backfill_embeddings(conn, "p1")
        hits = recall(conn, "p1", "which password hashing algorithm do we use", k=3)
        assert hits, "expected at least one hit"
        assert hits[0]["id"] == 1  # the password-hashing decision ranks first

    def test_recall_empty_when_no_events(self, conn: sqlite3.Connection) -> None:
        assert recall(conn, "p1", "anything", k=3) == []


class TestFindRelated:
    def test_structural_fusion_links_graph_neighbors(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            "INSERT INTO symbol_edges (project_id, from_path, to_path, edge_type, is_external) "
            "VALUES ('p1','api/auth.py','core/security.py','imports',0)"
        )
        _insert(conn, 1, "COMPONENT_STATUS", {"description": "auth route work", "path": "api/auth.py"})
        _insert(conn, 2, "COMPONENT_STATUS", {"description": "security helper change", "path": "core/security.py"})
        backfill_embeddings(conn, "p1")
        rel = find_related(conn, "p1", 1, k=5)
        ids = {r["id"] for r in rel}
        assert 2 in ids  # linked via the import edge (and/or semantics)
