"""Backfill embeddings for events that lack one (E4).

Used for historic events created before the feature was enabled, and after a
model swap (model_version changes invalidate prior vectors). Idempotent: only
embeds active events with no vector for the current model_version.
"""
from __future__ import annotations

import json
import sqlite3


def backfill_embeddings(conn: sqlite3.Connection, project_id: str) -> int:
    """Embed + store any active events missing a current-model vector. Returns count."""
    from memlora.embedding.input import embedding_input
    from memlora.embedding.model import EMBEDDING_MODEL_VERSION, embed_text, is_available
    from memlora.embedding.store import upsert_embedding

    if not is_available():
        return 0

    rows = conn.execute(
        """
        SELECT e.id, e.event_type, e.payload
        FROM events e
        LEFT JOIN event_embeddings ee
          ON ee.event_id = e.id AND ee.model_version = ?
        WHERE e.project_id     = ?
          AND e.archived       = 0
          AND e.superseded_by IS NULL
          AND ee.event_id IS NULL
        """,
        (EMBEDDING_MODEL_VERSION, project_id),
    ).fetchall()

    count = 0
    for r in rows:
        payload = json.loads(r["payload"])
        vec = embed_text(embedding_input(payload, r["event_type"]))
        if vec is not None:
            upsert_embedding(conn, r["id"], vec, EMBEDDING_MODEL_VERSION)
            count += 1
    conn.commit()
    return count
