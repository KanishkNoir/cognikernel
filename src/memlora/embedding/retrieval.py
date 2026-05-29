"""Semantic retrieval over the event embedding store (E3 + E5).

Two reusable primitives that other layers consume:

  recall(query)        — pure semantic top-K over active events. Powers semantic
                         dedup-at-insert and the (deferred) pull-model
                         query-directory ("decisions about X").

  find_related(event)  — semantic neighbours UNION symbol-graph-adjacent events.
                         The code-aware fusion: a decision about auth.py is
                         related to other auth.py decisions AND to decisions
                         touching files that import / are imported by auth.py
                         (via symbol_edges). This is the angle a text-only memory
                         system cannot offer.

Brute-force cosine over the project's active vectors — fine at local scale; the
same interface admits sqlite-vec later if needed.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

_STRUCTURAL_BASE_SCORE = 0.5   # baseline rank for graph-only neighbours
_STRUCTURAL_BOOST = 0.1        # added when a candidate is both semantic + structural


def _active_vectors(conn: sqlite3.Connection, project_id: str, model_version: str,
                    exclude_id: int | None = None):
    """Yield (id, event_type, payload, weight, np.ndarray) for active events with vectors."""
    import numpy as np
    rows = conn.execute(
        """
        SELECT e.id, e.event_type, e.payload, e.weight, ee.vector
        FROM events e JOIN event_embeddings ee ON ee.event_id = e.id
        WHERE e.project_id    = ?
          AND e.archived      = 0
          AND e.superseded_by IS NULL
          AND ee.model_version = ?
        """,
        (project_id, model_version),
    ).fetchall()
    for r in rows:
        if exclude_id is not None and r["id"] == exclude_id:
            continue
        yield r["id"], r["event_type"], r["payload"], r["weight"], np.frombuffer(r["vector"], dtype="float32")


def recall(
    conn: sqlite3.Connection,
    project_id: str,
    query_text: str,
    k: int = 8,
    threshold: float = 0.0,
    event_types: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return up to `k` active events most semantically similar to `query_text`.

    Empty list if the model is unavailable. Results are dicts with id, event_type,
    score, description, subject — ranked by cosine descending.
    """
    from memlora.embedding.model import EMBEDDING_MODEL_VERSION, embed_text

    qv = embed_text(query_text)
    if qv is None:
        return []

    scored: list[tuple[float, int, str, dict[str, Any]]] = []
    for eid, etype, payload_json, _weight, vec in _active_vectors(conn, project_id, EMBEDDING_MODEL_VERSION):
        if event_types and etype not in event_types:
            continue
        score = float(qv @ vec)
        if score >= threshold:
            scored.append((score, eid, etype, json.loads(payload_json)))

    scored.sort(key=lambda t: -t[0])
    return [
        {
            "id": eid,
            "event_type": etype,
            "score": round(score, 4),
            "description": payload.get("description", ""),
            "subject": payload.get("subject", ""),
        }
        for score, eid, etype, payload in scored[:k]
    ]


# ── symbol-graph fusion (E3) ──────────────────────────────────────────────────


def _event_files(payload: dict[str, Any]) -> set[str]:
    files = set(payload.get("affected_files") or [])
    path = payload.get("path")
    if path:
        files.add(path)
    return {f for f in files if f}


def _graph_neighbor_files(conn: sqlite3.Connection, project_id: str, files: set[str]) -> set[str]:
    """Files one import-edge away from `files` (both directions, local only)."""
    if not files:
        return set()
    placeholders = ",".join("?" * len(files))
    neighbors: set[str] = set()
    for row in conn.execute(
        f"SELECT to_path FROM symbol_edges WHERE project_id=? AND is_external=0 AND from_path IN ({placeholders})",
        [project_id, *files],
    ).fetchall():
        neighbors.add(row["to_path"])
    for row in conn.execute(
        f"SELECT from_path FROM symbol_edges WHERE project_id=? AND is_external=0 AND to_path IN ({placeholders})",
        [project_id, *files],
    ).fetchall():
        neighbors.add(row["from_path"])
    return neighbors


def find_related(
    conn: sqlite3.Connection,
    project_id: str,
    event_id: int,
    k: int = 8,
    threshold: float = 0.6,
) -> list[dict[str, Any]]:
    """Return events related to `event_id` by semantics and/or code structure.

    semantic: cosine of the event's embedding vs the others.
    structural: events touching files adjacent to this event's files in the
                import graph (symbol_edges).
    Candidates found by both rank highest.
    """
    from memlora.embedding.input import embedding_input
    from memlora.embedding.model import EMBEDDING_MODEL_VERSION, embed_text

    row = conn.execute(
        "SELECT event_type, payload FROM events WHERE id = ? AND project_id = ?",
        (event_id, project_id),
    ).fetchone()
    if row is None:
        return []
    payload = json.loads(row["payload"])

    related: dict[int, dict[str, Any]] = {}

    # Semantic neighbours.
    qv = embed_text(embedding_input(payload, row["event_type"]))
    if qv is not None:
        for eid, _etype, _pj, _w, vec in _active_vectors(
            conn, project_id, EMBEDDING_MODEL_VERSION, exclude_id=event_id
        ):
            score = float(qv @ vec)
            if score >= threshold:
                related[eid] = {"score": score, "why": "semantic"}

    # Structural neighbours via the symbol import graph.
    files = _event_files(payload)
    neighbor_files = _graph_neighbor_files(conn, project_id, files) | files
    if neighbor_files:
        for r in conn.execute(
            """
            SELECT id, payload FROM events
            WHERE project_id = ? AND archived = 0 AND superseded_by IS NULL AND id != ?
            """,
            (project_id, event_id),
        ).fetchall():
            if _event_files(json.loads(r["payload"])) & neighbor_files:
                if r["id"] in related:
                    related[r["id"]]["score"] += _STRUCTURAL_BOOST
                    related[r["id"]]["why"] = "semantic+structural"
                else:
                    related[r["id"]] = {"score": _STRUCTURAL_BASE_SCORE, "why": "structural"}

    ranked = sorted(related.items(), key=lambda kv: -kv[1]["score"])[:k]
    return [{"id": eid, "score": round(meta["score"], 4), "why": meta["why"]} for eid, meta in ranked]
