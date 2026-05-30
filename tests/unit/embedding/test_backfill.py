"""Tests for embedding backfill (E4). Model-guarded."""
from __future__ import annotations

import json
import sqlite3

import pytest

from memlora.embedding.backfill import backfill_embeddings
from memlora.embedding.model import EMBEDDING_MODEL_VERSION, is_available
from memlora.storage.migrations import run_migrations

pytestmark = pytest.mark.skipif(not is_available(), reason="embedding model not installed")


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    run_migrations(c)
    yield c
    c.close()


def _insert(conn, eid, payload, archived=0, superseded_by=None):
    conn.execute(
        """INSERT INTO events (id, project_id, session_id, created_at, event_type,
                               payload, content_hash, weight, mention_count, archived, superseded_by)
           VALUES (?, 'p1', 's', ?, 'DECISION', ?, ?, 1.0, 1, ?, ?)""",
        (eid, eid, json.dumps(payload), f"h{eid}", archived, superseded_by),
    )
    conn.commit()


class TestBackfill:
    def test_backfills_active_only(self, conn: sqlite3.Connection) -> None:
        _insert(conn, 1, {"description": "Use argon2id for hashing"})
        _insert(conn, 2, {"description": "archived decision"}, archived=1)
        _insert(conn, 3, {"description": "superseded decision"}, superseded_by=1)
        n = backfill_embeddings(conn, "p1")
        assert n == 1  # only the active, non-superseded event
        stored = conn.execute("SELECT event_id FROM event_embeddings").fetchall()
        assert {r["event_id"] for r in stored} == {1}

    def test_idempotent(self, conn: sqlite3.Connection) -> None:
        _insert(conn, 1, {"description": "Use argon2id for hashing"})
        assert backfill_embeddings(conn, "p1") == 1
        assert backfill_embeddings(conn, "p1") == 0  # already embedded for this model_version

    def test_input_version_bump_triggers_reembed_in_place(self, conn: sqlite3.Connection) -> None:
        """#3: a vector from a prior composition (stale version) is re-embedded
        under the current version, replacing the row (event_id PK — no orphan)."""
        import numpy as np

        from memlora.embedding.store import upsert_embedding

        _insert(conn, 1, {"description": "Use argon2id for hashing"})
        upsert_embedding(conn, 1, np.zeros(384, dtype="float32"), "bge-small-en-v1.5+in0")
        assert backfill_embeddings(conn, "p1") == 1  # current-version row missing → re-embed
        rows = conn.execute("SELECT model_version FROM event_embeddings WHERE event_id=1").fetchall()
        assert len(rows) == 1  # replaced in place, not duplicated
        assert rows[0]["model_version"] == EMBEDDING_MODEL_VERSION
