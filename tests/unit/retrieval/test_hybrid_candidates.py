"""S5 T-501: every candidate recall considered, not just the ones it returned.

`hybrid_recall` fuses the lexical and dense axes and keeps the top k; everything
below the cut, and which axes ran at all, was discarded. `explain-recall` needs
both to answer "why was this retrieved" and "why wasn't it", so the fusion is
exposed as `hybrid_candidates` and recall is defined as its top k — one
implementation, so the explanation cannot drift from what recall does.
"""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from cognikernel.retrieval.hybrid import hybrid_candidates, hybrid_recall
from cognikernel.storage.events import Event, insert_event
from cognikernel.storage.migrations import run_migrations

PID = "b" * 16


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    run_migrations(c)
    yield c
    c.close()


def _add(conn, desc: str) -> int:
    return insert_event(conn, Event(
        project_id=PID, session_id="s1", event_type="DECISION",
        payload={"description": desc, "subject": ""},
        content_hash=hashlib.sha256(desc.encode()).hexdigest(),
    ))


def _seed(conn) -> list[int]:
    return [
        _add(conn, "Retry policy: 6 attempts with full jitter"),
        _add(conn, "Retry backoff base is 1.5 seconds"),
        _add(conn, "Retry cap is 45 seconds"),
        _add(conn, "Retry only on network errors"),
        _add(conn, "Retry budget is per deployment"),
        _add(conn, "Streaming uses SSE, not WebSockets"),
    ]


@pytest.fixture()
def cold_model(monkeypatch):
    monkeypatch.setattr("cognikernel.embedding.model.is_ready", lambda: False)
    monkeypatch.setattr("cognikernel.embedding.model.warm", lambda: None)


class TestHybridCandidates:
    def test_recall_is_exactly_the_top_k_of_the_candidates(self, conn, cold_model) -> None:
        _seed(conn)

        candidates = hybrid_candidates(conn, PID, "retry", n_per_axis=20)
        hits = hybrid_recall(conn, PID, "retry", k=2, n_per_axis=20)

        assert hits == candidates["fused"][:2]

    def test_candidates_below_the_cut_are_kept(self, conn, cold_model) -> None:
        _seed(conn)

        candidates = hybrid_candidates(conn, PID, "retry", n_per_axis=20)

        assert len(candidates["fused"]) == 5
        assert len(hybrid_recall(conn, PID, "retry", k=2)) == 2

    def test_reports_which_axes_ran(self, conn, cold_model) -> None:
        _seed(conn)

        candidates = hybrid_candidates(conn, PID, "retry")

        assert candidates["dense_available"] is False
        assert candidates["lexical_available"] is True

    def test_no_fts_and_a_cold_model_means_no_axis(self, conn, cold_model, monkeypatch) -> None:
        monkeypatch.setattr("cognikernel.storage.fts.fts_enabled", lambda c: False)
        _seed(conn)

        candidates = hybrid_candidates(conn, PID, "retry")

        assert candidates["lexical_available"] is False
        assert candidates["fused"] == []
        assert hybrid_recall(conn, PID, "retry") == []

    def test_both_axes_keep_their_own_ranks(self, conn, monkeypatch) -> None:
        ids = _seed(conn)
        monkeypatch.setattr("cognikernel.embedding.model.is_ready", lambda: True)
        monkeypatch.setattr("cognikernel.embedding.model.warm", lambda: None)
        dense = [{"id": ids[5], "event_type": "DECISION", "description": "Streaming uses SSE", "subject": "",
                  "score": 0.71},
                 {"id": ids[0], "event_type": "DECISION", "description": "Retry policy", "subject": "",
                  "score": 0.64}]
        monkeypatch.setattr("cognikernel.embedding.retrieval.recall", lambda *a, **kw: dense)

        candidates = hybrid_candidates(conn, PID, "retry")
        by_id = {c["id"]: c for c in candidates["fused"]}

        assert candidates["dense_available"] is True
        assert (by_id[ids[5]]["dense_rank"], by_id[ids[5]]["bm25_rank"]) == (1, None)
        assert by_id[ids[0]]["dense_rank"] == 2 and by_id[ids[0]]["bm25_rank"] is not None
        assert by_id[ids[5]]["cosine"] == 0.71
        assert candidates["fused"][0]["id"] == ids[0]  # on both axes, so it fuses highest
