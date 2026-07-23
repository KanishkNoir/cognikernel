"""Embedding persistence + cosine retrieval over SQLite.

Vectors are stored as raw float32 bytes (L2-normalized at write), so cosine
similarity is a dot product. At local project scale (hundreds–low thousands of
events) a brute-force NumPy dot over the candidate set is well under a
millisecond, so no vector index is required; sqlite-vec could slot in later
behind this same interface if scale demands it.
"""
from __future__ import annotations

import sqlite3
import time
from typing import Any


def upsert_embedding(
    conn: sqlite3.Connection,
    event_id: int,
    vector,
    model_version: str,
    now_ms: int | None = None,
) -> None:
    """Store (or replace) the embedding for an event. No-op if vector is None."""
    if vector is None:
        return
    buf = vector.astype("float32").tobytes()
    conn.execute(
        """
        INSERT OR REPLACE INTO event_embeddings
            (event_id, model_version, dim, vector, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (event_id, model_version, int(len(vector)), buf,
         now_ms if now_ms is not None else int(time.time() * 1000)),
    )


def load_embeddings(
    conn: sqlite3.Connection,
    event_ids: list[int],
    model_version: str | None = None,
) -> dict[int, Any]:
    """Load {event_id: float32 ndarray} for the given ids that have a vector.

    Filters to `model_version` when provided so a model swap doesn't compare
    vectors across embedding spaces. Returns {} if numpy is unavailable.
    """
    if not event_ids:
        return {}
    try:
        import numpy as np
    except Exception:
        return {}

    placeholders = ",".join("?" * len(event_ids))
    params: list[Any] = list(event_ids)
    sql = f"SELECT event_id, vector FROM event_embeddings WHERE event_id IN ({placeholders})"
    if model_version is not None:
        sql += " AND model_version = ?"
        params.append(model_version)

    out: dict[int, Any] = {}
    for row in conn.execute(sql, params).fetchall():
        out[row["event_id"]] = np.frombuffer(row["vector"], dtype="float32")
    return out


def cosine_matches(
    query_vec,
    candidates: dict[int, Any],
    threshold: float,
) -> dict[int, float]:
    """Return {event_id: cosine} for candidates whose cosine >= threshold.

    Assumes both query and stored vectors are L2-normalized, so cosine == dot.
    """
    if query_vec is None or not candidates:
        return {}
    matches: dict[int, float] = {}
    for eid, vec in candidates.items():
        score = float(query_vec @ vec)
        if score >= threshold:
            matches[eid] = score
    return matches
