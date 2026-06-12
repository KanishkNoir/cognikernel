"""Hybrid retrieval core (J1.2): BM25 ∪ dense cosine → Reciprocal Rank Fusion.

One engine behind all three memory surfaces (`recall` MCP tool, `find_related`
seeding, CK-1 per-prompt push). Lexical and semantic evidence fuse BY RANK —
no raw score is ever compared across axes. That is the structural fix for the
calibration category error that silenced CK-1: cosine and Jaccard/BM25 live on
incomparable scales, but rank 3 is rank 3 everywhere.

Degradation ladder (every step additive, nothing load-bearing):
  both axes available  -> RRF fusion
  one axis available   -> that axis's ranking alone
  zero axes            -> [] (caller falls back to the legacy Jaccard scan)
"""
from __future__ import annotations

import sqlite3
from typing import Any

# Canonical RRF constant (Cormack & Clarke). Deliberately not a config knob —
# fusion is insensitive to K at this corpus size (~hundreds of events).
_RRF_K = 60

# Normalizer so the displayed score is human-meaningful: 1.0 = rank 1 on both
# axes, 0.5 = rank 1 on a single axis. Ranks, not this score, drive gating.
_RRF_MAX = 2.0 / (_RRF_K + 1)


def hybrid_recall(
    conn: sqlite3.Connection,
    project_id: str,
    query_text: str,
    k: int = 8,
    n_per_axis: int = 20,
) -> list[dict[str, Any]]:
    """Top-k active events for `query_text`, fused across lexical + dense axes.

    Result dicts: {id, event_type, description, subject, score, dense_rank,
    bm25_rank, cosine} — `score` is normalized RRF in [0, 1]; the per-axis
    ranks (None when that axis didn't surface the event) are load-bearing for
    the CK-1 dual-evidence gate. Returns [] when no axis is available.
    """
    from memlora.embedding.model import is_ready, warm

    # Kick the single background model load (no-op if loading/loaded); never
    # block on it — a cold model just means the dense axis is absent this call.
    warm()

    dense_hits: list[dict[str, Any]] = []
    if is_ready():
        from memlora.embedding.retrieval import recall as dense_recall

        dense_hits = dense_recall(conn, project_id, query_text, k=n_per_axis)

    from memlora.storage.fts import bm25_search

    lex_hits = bm25_search(conn, project_id, query_text, n=n_per_axis)

    if not dense_hits and not lex_hits:
        return []

    fused: dict[int, dict[str, Any]] = {}

    def _entry(h: dict[str, Any]) -> dict[str, Any]:
        return fused.setdefault(
            h["id"],
            {
                "id": h["id"],
                "event_type": h["event_type"],
                "description": h.get("description", ""),
                "subject": h.get("subject", ""),
                "rrf": 0.0,
                "dense_rank": None,
                "bm25_rank": None,
                "cosine": None,
            },
        )

    for rank, h in enumerate(dense_hits, 1):
        e = _entry(h)
        e["rrf"] += 1.0 / (_RRF_K + rank)
        e["dense_rank"] = rank
        e["cosine"] = h.get("score")

    for rank, h in enumerate(lex_hits, 1):
        e = _entry(h)
        e["rrf"] += 1.0 / (_RRF_K + rank)
        e["bm25_rank"] = rank

    # Deterministic: ties broken by id so identical stores render identically.
    ranked = sorted(fused.values(), key=lambda e: (-e["rrf"], e["id"]))[:k]
    for e in ranked:
        e["score"] = round(e.pop("rrf") / _RRF_MAX, 4)
    return ranked
