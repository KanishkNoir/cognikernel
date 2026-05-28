"""CRUD tests for storage.enrichment_jobs (Phase A-5)."""
from __future__ import annotations

import sqlite3

import pytest

from memlora.storage import enrichment_jobs as ej
from memlora.storage.evidence import store_evidence


def _evidence_id(conn: sqlite3.Connection) -> int:
    return store_evidence(conn, "p1", "s1", "transcript", b"content")


# ── enqueue ──────────────────────────────────────────────────────────────────


class TestEnqueue:
    def test_returns_new_id(self, conn: sqlite3.Connection) -> None:
        eid = _evidence_id(conn)
        job_id = ej.enqueue(conn, "p1", eid, extractor_version="llm-v1")
        assert job_id > 0

    def test_idempotent_for_same_natural_key(self, conn: sqlite3.Connection) -> None:
        eid = _evidence_id(conn)
        a = ej.enqueue(conn, "p1", eid, extractor_version="llm-v1")
        b = ej.enqueue(conn, "p1", eid, extractor_version="llm-v1")
        assert a == b
        assert conn.execute("SELECT COUNT(*) FROM enrichment_jobs").fetchone()[0] == 1

    def test_different_versions_create_separate_jobs(
        self, conn: sqlite3.Connection,
    ) -> None:
        eid = _evidence_id(conn)
        v1 = ej.enqueue(conn, "p1", eid, extractor_version="llm-v1")
        v2 = ej.enqueue(conn, "p1", eid, extractor_version="llm-v2")
        assert v1 != v2

    def test_rejects_unknown_kind(self, conn: sqlite3.Connection) -> None:
        with pytest.raises(ValueError, match="invalid enrichment_kind"):
            ej.enqueue(
                conn, "p1", _evidence_id(conn),
                enrichment_kind="bogus", extractor_version="llm-v1",
            )

    def test_default_kind_is_decision_extraction(
        self, conn: sqlite3.Connection,
    ) -> None:
        job_id = ej.enqueue(conn, "p1", _evidence_id(conn), extractor_version="llm-v1")
        job = ej.get(conn, job_id)
        assert job.enrichment_kind == "llm_decision_extraction"


# ── lifecycle transitions ────────────────────────────────────────────────────


class TestLifecycleTransitions:
    def test_mark_completed_advances_state(self, conn: sqlite3.Connection) -> None:
        job_id = ej.enqueue(conn, "p1", _evidence_id(conn), extractor_version="llm-v1")
        ej.mark_completed(conn, job_id, now_ms=42)

        job = ej.get(conn, job_id)
        assert job.state == "completed"
        assert job.completed_at == 42
        assert job.error == ""

    def test_mark_partial_keeps_error(self, conn: sqlite3.Connection) -> None:
        job_id = ej.enqueue(conn, "p1", _evidence_id(conn), extractor_version="llm-v1")
        ej.mark_partial(conn, job_id, "3 of 5 events failed")

        job = ej.get(conn, job_id)
        assert job.state == "partial"
        assert "3 of 5" in job.error

    def test_mark_failed_records_error(self, conn: sqlite3.Connection) -> None:
        job_id = ej.enqueue(conn, "p1", _evidence_id(conn), extractor_version="llm-v1")
        ej.mark_failed(conn, job_id, "all events rejected")

        job = ej.get(conn, job_id)
        assert job.state == "failed"
        assert job.error == "all events rejected"

    def test_long_errors_are_truncated(self, conn: sqlite3.Connection) -> None:
        job_id = ej.enqueue(conn, "p1", _evidence_id(conn), extractor_version="llm-v1")
        long_err = "x" * 1000
        ej.mark_partial(conn, job_id, long_err)

        job = ej.get(conn, job_id)
        assert len(job.error) == 500


# ── list_pending_for_version ─────────────────────────────────────────────────


class TestListPending:
    def test_returns_queued_jobs_for_version(self, conn: sqlite3.Connection) -> None:
        eid = _evidence_id(conn)
        ej.enqueue(conn, "p1", eid, extractor_version="llm-v1")

        pending = ej.list_pending_for_version(conn, "p1", "llm-v1")
        assert len(pending) == 1
        assert pending[0].evidence_id == eid
        assert pending[0].state == "queued"

    def test_returns_partial_jobs_for_retry(self, conn: sqlite3.Connection) -> None:
        eid = _evidence_id(conn)
        job_id = ej.enqueue(conn, "p1", eid, extractor_version="llm-v1")
        ej.mark_partial(conn, job_id, "some failed")

        pending = ej.list_pending_for_version(conn, "p1", "llm-v1")
        # Partial jobs are retry candidates — included.
        assert len(pending) == 1
        assert pending[0].state == "partial"

    def test_excludes_completed(self, conn: sqlite3.Connection) -> None:
        eid = _evidence_id(conn)
        job_id = ej.enqueue(conn, "p1", eid, extractor_version="llm-v1")
        ej.mark_completed(conn, job_id)

        pending = ej.list_pending_for_version(conn, "p1", "llm-v1")
        assert pending == []

    def test_excludes_failed(self, conn: sqlite3.Connection) -> None:
        eid = _evidence_id(conn)
        job_id = ej.enqueue(conn, "p1", eid, extractor_version="llm-v1")
        ej.mark_failed(conn, job_id, "x")

        pending = ej.list_pending_for_version(conn, "p1", "llm-v1")
        assert pending == []

    def test_other_version_isolated(self, conn: sqlite3.Connection) -> None:
        eid = _evidence_id(conn)
        ej.enqueue(conn, "p1", eid, extractor_version="llm-v1")
        ej.enqueue(conn, "p1", eid, extractor_version="llm-v2")

        v1 = ej.list_pending_for_version(conn, "p1", "llm-v1")
        v2 = ej.list_pending_for_version(conn, "p1", "llm-v2")
        assert len(v1) == 1
        assert len(v2) == 1
        assert v1[0].extractor_version == "llm-v1"
        assert v2[0].extractor_version == "llm-v2"

    def test_limit_respected(self, conn: sqlite3.Connection) -> None:
        # Seed 7 evidence rows so we can have 7 distinct jobs.
        for i in range(7):
            eid = store_evidence(conn, "p1", f"s{i}", "transcript", f"c{i}".encode())
            ej.enqueue(conn, "p1", eid, extractor_version="llm-v1")

        pending = ej.list_pending_for_version(conn, "p1", "llm-v1", limit=5)
        assert len(pending) == 5

    def test_ordered_by_queue_time(self, conn: sqlite3.Connection) -> None:
        e1 = store_evidence(conn, "p1", "s1", "transcript", b"c1")
        e2 = store_evidence(conn, "p1", "s2", "transcript", b"c2")

        ej.enqueue(conn, "p1", e1, extractor_version="llm-v1", now_ms=200)
        ej.enqueue(conn, "p1", e2, extractor_version="llm-v1", now_ms=100)

        pending = ej.list_pending_for_version(conn, "p1", "llm-v1")
        # e2 queued earlier → comes first.
        assert pending[0].evidence_id == e2
        assert pending[1].evidence_id == e1
