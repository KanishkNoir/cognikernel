"""Concurrent workers must not double-process or corrupt state (audit, integration).

The single-flight lock (one worker per project) is the defense against a worker
storm piling onto the same SQLite file. This drives it under real concurrency:
two workers racing on a project with queued jobs must process every job exactly
once, with no exceptions and no duplicate events.
"""
from __future__ import annotations

import threading

from memlora.storage.connection import get_connection


class TestWorkerContention:
    def test_two_racing_workers_process_each_job_once(self, project, jsonl) -> None:
        from memlora.integration.session import process_jobs, session_capture

        # Three distinct sessions => three independent queued jobs.
        for i in range(3):
            session_capture(project.path, f"sess-{i}", jsonl(5))

        results: list = []
        errors: list = []

        def _run() -> None:
            try:
                results.append(process_jobs(project.path))
            except Exception as exc:  # a worker must never crash the process
                errors.append(exc)

        threads = [threading.Thread(target=_run) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert not errors, f"worker(s) raised: {errors}"
        assert all(not t.is_alive() for t in threads), "a worker hung"

        # Exactly-once across both workers: every job is processed exactly once,
        # never twice. (We deliberately do NOT assert the second worker observed a
        # held lock — on a fast runner worker A can drain all three and release
        # before B even attempts, so B legitimately finds an empty queue. The real
        # invariant is the outcome below, not the interleaving.)
        total_processed = sum(r.get("processed", 0) for r in results)
        assert total_processed == 3, f"expected 3 jobs processed once, got {total_processed}"

        # The queue is fully drained and no duplicate events were minted.
        with get_connection(project.db) as conn:
            remaining = conn.execute(
                "SELECT COUNT(*) FROM extraction_jobs WHERE state IN ('queued','claimed','running')"
            ).fetchone()[0]
            dup = conn.execute(
                "SELECT COUNT(*) FROM ("
                "  SELECT content_hash FROM events GROUP BY project_id, content_hash "
                "  HAVING COUNT(*) > 1)"
            ).fetchone()[0]
        assert remaining == 0, f"{remaining} jobs left unprocessed"
        assert dup == 0, "duplicate (project_id, content_hash) events were minted"
