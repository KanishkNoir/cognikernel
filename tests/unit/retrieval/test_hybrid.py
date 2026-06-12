"""J1.2 — RRF fusion math, degradation ladder, and the measured F-B miss class."""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

import memlora.retrieval.hybrid as hybrid_mod
from memlora.retrieval.hybrid import _RRF_K, _RRF_MAX, hybrid_recall
from memlora.storage.events import Event, insert_event
from memlora.storage.migrations import run_migrations

PID = "a" * 16


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    run_migrations(c)
    yield c
    c.close()


def _mk(desc: str, etype: str = "DECISION") -> Event:
    return Event(
        project_id=PID,
        session_id="s1",
        event_type=etype,
        payload={"description": desc, "subject": ""},
        content_hash=hashlib.sha256(desc.encode()).hexdigest(),
    )


def _seed(conn) -> dict[str, int]:
    """The two measured F-B misses + distractors."""
    ids = {}
    ids["timeout"] = insert_event(conn, _mk(
        "Retry: 2 attempts per deployment, base=100 ms, multiplier=2x, max=500 ms; "
        "request_total_timeout 300 s per model group"))
    ids["selection"] = insert_event(conn, _mk(
        "Do not use least-latency selection or weighted random. Operator-defined "
        "priority order picks between two healthy deployments of the same model."))
    ids["noise1"] = insert_event(conn, _mk("Streaming uses SSE, not WebSockets."))
    ids["noise2"] = insert_event(conn, _mk("Money is stored as integer nano-dollars."))
    return ids


class TestDegradationLadder:
    def test_bm25_only_when_model_cold(self, conn, monkeypatch) -> None:
        """Embedding model cold -> pure BM25 ranking, still returns hits."""
        monkeypatch.setattr("memlora.embedding.model.is_ready", lambda: False)
        monkeypatch.setattr("memlora.embedding.model.warm", lambda: None)
        ids = _seed(conn)
        hits = hybrid_recall(conn, PID, "how many retry attempts and what timeout")
        assert hits, "BM25 axis alone must produce hits"
        assert hits[0]["id"] == ids["timeout"]
        assert hits[0]["bm25_rank"] == 1
        assert hits[0]["dense_rank"] is None
        assert hits[0]["score"] == pytest.approx((1 / (_RRF_K + 1)) / _RRF_MAX, abs=1e-4)

    def test_zero_axes_returns_empty(self, conn, monkeypatch) -> None:
        """No FTS + cold model -> [] so the caller can use the legacy scan."""
        monkeypatch.setattr("memlora.embedding.model.is_ready", lambda: False)
        monkeypatch.setattr("memlora.embedding.model.warm", lambda: None)
        monkeypatch.setattr("memlora.storage.fts.fts_enabled", lambda c: False)
        _seed(conn)
        assert hybrid_recall(conn, PID, "retry attempts") == []

    def test_identifier_query_hits(self, conn, monkeypatch) -> None:
        """The measured miss class: identifier-shaped query terms."""
        monkeypatch.setattr("memlora.embedding.model.is_ready", lambda: False)
        monkeypatch.setattr("memlora.embedding.model.warm", lambda: None)
        target = insert_event(conn, _mk("the relay-default alias resolves to claude-opus-4-8"))
        _seed(conn)
        hits = hybrid_recall(conn, PID, "what does relay-default resolve to")
        assert hits[0]["id"] == target


class TestRRFMath:
    def test_fusion_of_known_ranks(self, conn, monkeypatch) -> None:
        """Doc on both axes outranks a doc that is rank-1 on one axis only."""
        ids = _seed(conn)
        # Fake dense axis: 'selection' rank 1, 'timeout' rank 2.
        fake_dense = [
            {"id": ids["selection"], "event_type": "DECISION", "score": 0.9,
             "description": "d", "subject": ""},
            {"id": ids["timeout"], "event_type": "DECISION", "score": 0.8,
             "description": "d", "subject": ""},
        ]
        monkeypatch.setattr("memlora.embedding.model.is_ready", lambda: True)
        monkeypatch.setattr("memlora.embedding.model.warm", lambda: None)
        monkeypatch.setattr(
            "memlora.embedding.retrieval.recall",
            lambda conn, pid, q, k: fake_dense,
        )
        # BM25 axis on this query ranks 'timeout' first (attempts/timeout terms).
        hits = hybrid_recall(conn, PID, "retry attempts timeout deployment")
        by_id = {h["id"]: h for h in hits}
        t = by_id[ids["timeout"]]
        assert t["dense_rank"] == 2 and t["bm25_rank"] == 1
        expected_rrf = 1 / (_RRF_K + 2) + 1 / (_RRF_K + 1)
        assert t["score"] == pytest.approx(expected_rrf / _RRF_MAX, abs=1e-4)
        # Dual-axis presence beats single-axis rank 1.
        s = by_id[ids["selection"]]
        assert t["score"] > s["score"] or s["bm25_rank"] is not None

    def test_deterministic_tiebreak_by_id(self, conn, monkeypatch) -> None:
        monkeypatch.setattr("memlora.embedding.model.is_ready", lambda: False)
        monkeypatch.setattr("memlora.embedding.model.warm", lambda: None)
        a = insert_event(conn, _mk("identical twin fact alpha variant"))
        b = insert_event(conn, _mk("identical twin fact alpha variant."))
        hits = hybrid_recall(conn, PID, "identical twin fact alpha")
        assert [h["id"] for h in hits[:2]] == sorted([a, b]) or hits[0]["id"] < hits[1]["id"]


class TestQueryRewire:
    def test_recall_hits_uses_hybrid(self, conn, monkeypatch) -> None:
        from memlora.integration.query import _recall_hits

        monkeypatch.setattr("memlora.embedding.model.is_ready", lambda: False)
        monkeypatch.setattr("memlora.embedding.model.warm", lambda: None)
        ids = _seed(conn)
        hits = _recall_hits(conn, PID, "retry attempts timeout", 5)
        assert hits[0]["id"] == ids["timeout"]
        assert "score" in hits[0]

    def test_recall_hits_falls_back_to_lexical(self, conn, monkeypatch) -> None:
        from memlora.integration.query import _recall_hits

        monkeypatch.setattr("memlora.embedding.model.is_ready", lambda: False)
        monkeypatch.setattr("memlora.embedding.model.warm", lambda: None)
        monkeypatch.setattr("memlora.storage.fts.fts_enabled", lambda c: False)
        _seed(conn)
        hits = _recall_hits(conn, PID, "retry attempts timeout", 5)
        assert hits, "legacy Jaccard scan must keep recall alive with zero axes"
