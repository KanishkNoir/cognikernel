"""Malformed evidence must degrade gracefully, never crash the worker or corrupt
state (audit, integration level).

A real ingest can hand the pipeline binary garbage or a transcript truncated
mid-line (a killed capture). The worker must survive: it either extracts nothing
or dead-letters the job, but it must not raise out of process_jobs and must not
leave the events table corrupt or half-written.
"""
from __future__ import annotations

from cognikernel.storage.connection import get_connection
from cognikernel.storage.evidence import store_evidence
from cognikernel.storage.jobs import enqueue_extraction


def _enqueue_raw(project, session_id: str, blob: bytes) -> int:
    with get_connection(project.db) as conn:
        ev = store_evidence(conn, project.pid, session_id, "jsonl_transcript", blob)
        return enqueue_extraction(conn, project.pid, session_id, ev, "extract.transcript")


class TestCorruptEvidence:
    def test_binary_garbage_does_not_crash_worker(self, project) -> None:
        from cognikernel.integration.session import process_jobs

        _enqueue_raw(project, "sess-garbage", b"\x00\x01\x02 not json \xff\xfe rubbish")

        # Must return a summary, not raise.
        summary = process_jobs(project.path)
        assert isinstance(summary, dict)
        # The job is resolved one way or another; the queue does not wedge.
        with get_connection(project.db) as conn:
            stuck = conn.execute(
                "SELECT COUNT(*) FROM extraction_jobs WHERE state IN ('queued','claimed','running')"
            ).fetchone()[0]
            # DB stays readable / uncorrupted.
            conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert stuck == 0, "corrupt-evidence job left the queue wedged"

    def test_truncated_jsonl_is_handled(self, project) -> None:
        from cognikernel.integration.session import process_jobs

        # Valid first line, second line cut off mid-JSON (a killed capture).
        truncated = (
            '{"type": "user", "message": {"content": "We decided to use Redis"}}\n'
            '{"type": "user", "message": {"content": "We dec'
        )
        _enqueue_raw(project, "sess-trunc", truncated.encode("utf-8"))

        summary = process_jobs(project.path)
        assert isinstance(summary, dict)
        # Worker survived and the DB is intact + queryable.
        with get_connection(project.db) as conn:
            conn.execute("SELECT COUNT(*) FROM events").fetchone()
            stuck = conn.execute(
                "SELECT COUNT(*) FROM extraction_jobs WHERE state IN ('queued','claimed','running')"
            ).fetchone()[0]
        assert stuck == 0
