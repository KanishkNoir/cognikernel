"""Tests for the I4 decoupled capture+process pipeline.

Key invariants:
1. Exactly-once claim: two workers racing on the same job — only one processes it.
2. Crash mid-merge → dead_lettered → replay reproduces identical (content_hash) events.
3. session_capture returns before any extraction, well under the 500ms target.
4. process_jobs advances the ingest cursor after each successful merge.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
from cognikernel.storage.cursors import get_cursor
from cognikernel.storage.evidence import store_evidence
from cognikernel.storage.jobs import (
    claim_next_job, enqueue_extraction, fail_job,
    get_job, list_jobs, recover_stuck_running_jobs, replay_dead_letter,
)
from cognikernel.storage.migrations import run_migrations


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_project(tmp_path: Path):
    project_path = tmp_path / "proj"
    project_path.mkdir()
    from cognikernel.integration.session import init_project
    import os
    os.environ.setdefault("COGNIKERNEL_DISABLE_AUTO_WARM", "1")
    init_project(str(project_path))
    return project_path


def _jsonl(n: int) -> str:
    lines = [
        json.dumps({"type": "user", "message": {"content": f"User: We decided D{i} = value{i}"}})
        for i in range(n)
    ]
    return "\n".join(lines) + "\n"


# ── exactly-once claim ────────────────────────────────────────────────────────

class TestExactlyOnceClaim:
    def test_two_workers_claim_same_job_only_one_wins(self, tmp_path):
        db = tmp_path / "test.db"
        conn1 = sqlite3.connect(str(db)); conn1.row_factory = sqlite3.Row
        conn2 = sqlite3.connect(str(db)); conn2.row_factory = sqlite3.Row
        run_migrations(conn1)

        ev_id = store_evidence(conn1, "p", "s", "transcript", b"hello")
        enqueue_extraction(conn1, "p", "s", ev_id, "extract.transcript")

        # Both workers attempt to claim concurrently (simulated via separate connections).
        job1 = claim_next_job(conn1, "extract.transcript", "worker-1")
        job2 = claim_next_job(conn2, "extract.transcript", "worker-2")

        # Exactly one should succeed; the other gets None (job already claimed).
        results = [job1, job2]
        successes = [j for j in results if j is not None]
        assert len(successes) == 1
        winner = successes[0]
        assert winner.claimed_by in ("worker-1", "worker-2")

    def test_second_claim_returns_none_when_queue_empty(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db)); conn.row_factory = sqlite3.Row
        run_migrations(conn)

        ev_id = store_evidence(conn, "p", "s", "transcript", b"data")
        enqueue_extraction(conn, "p", "s", ev_id, "extract.transcript")

        job = claim_next_job(conn, "extract.transcript", "worker-A")
        assert job is not None

        none_job = claim_next_job(conn, "extract.transcript", "worker-B")
        assert none_job is None


# ── crash → dead_lettered → replay ───────────────────────────────────────────

class TestCrashAndReplay:
    def test_failed_job_becomes_dead_lettered(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db)); conn.row_factory = sqlite3.Row
        run_migrations(conn)

        ev_id = store_evidence(conn, "p", "s", "transcript", b"content")
        job_id = enqueue_extraction(conn, "p", "s", ev_id, "extract.transcript")

        # Simulate crash: fail immediately (non-retryable after max_attempts).
        for _ in range(3):  # max_attempts = 3
            fail_job(conn, job_id, "EXTRACTOR_BUG", "simulated crash")

        job = get_job(conn, job_id)
        assert job.state == "dead_lettered"

    def test_replay_restores_to_queued(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db)); conn.row_factory = sqlite3.Row
        run_migrations(conn)

        ev_id = store_evidence(conn, "p", "s", "transcript", b"content")
        job_id = enqueue_extraction(conn, "p", "s", ev_id, "extract.transcript")
        for _ in range(3):
            fail_job(conn, job_id, "EXTRACTOR_BUG", "crash")

        replay_dead_letter(conn, job_id)
        job = get_job(conn, job_id)
        assert job.state == "queued"
        assert job.failure_class is None
        assert job.attempts == 0

    def test_process_jobs_replays_timeout_dead_letters(self, tmp_path):
        """process_jobs promotes TIMEOUT dead-letters (process-killed) → processed."""
        project_path = _make_project(tmp_path)
        from cognikernel.integration.session import session_capture, process_jobs
        from cognikernel.config import Config

        raw = _jsonl(10)
        session_id = "test-session-replay"

        # Capture (enqueue only).
        result = session_capture(str(project_path), session_id, raw)
        job_id = result["job_id"]

        # Dead-letter via TIMEOUT (the process-killed class — auto-replayable).
        config = Config.load(project_path=str(project_path))
        pid = hash_project_path(str(project_path))
        db_path = get_db_path(config, pid)
        with get_connection(db_path) as conn:
            for _ in range(3):
                fail_job(conn, job_id, "TIMEOUT", "simulated kill")

        with get_connection(db_path) as conn:
            assert get_job(conn, job_id).state == "dead_lettered"

        summary = process_jobs(str(project_path))
        assert summary["replayed"] >= 1
        assert summary["processed"] >= 1
        assert summary["failed"] == 0

        with get_connection(db_path) as conn:
            assert get_job(conn, job_id).state == "completed"

    def test_process_jobs_does_not_resurrect_poison(self, tmp_path):
        """Gamma post-mortem F3: EXTRACTOR_BUG dead-letters must stay dead —
        auto-replaying them forever masks real bugs and burns CPU."""
        project_path = _make_project(tmp_path)
        from cognikernel.integration.session import session_capture, process_jobs
        from cognikernel.config import Config

        raw = _jsonl(10)
        result = session_capture(str(project_path), "test-poison", raw)
        job_id = result["job_id"]

        config = Config.load(project_path=str(project_path))
        pid = hash_project_path(str(project_path))
        db_path = get_db_path(config, pid)
        with get_connection(db_path) as conn:
            for _ in range(3):
                fail_job(conn, job_id, "EXTRACTOR_BUG", "poison input")

        summary = process_jobs(str(project_path))
        assert summary["replayed"] == 0

        with get_connection(db_path) as conn:
            assert get_job(conn, job_id).state == "dead_lettered"  # stays dead

    def test_single_flight_lock_blocks_second_worker(self, tmp_path):
        """Gamma post-mortem F2: a second worker must exit immediately when the
        project lock is held (worker storm prevention)."""
        project_path = _make_project(tmp_path)
        from cognikernel.config import Config
        from cognikernel.integration.session import process_jobs, _acquire_worker_lock

        # Hold the lock under the SAME root process_jobs resolves from config, so
        # the second worker actually sees it (audit P2: lock dir follows config).
        cfg = Config.load(project_path=str(project_path))
        pid = hash_project_path(str(project_path))
        lock = _acquire_worker_lock(pid, cognikernel_dir=cfg.cognikernel_dir)
        assert lock is not None
        try:
            summary = process_jobs(str(project_path))
            assert summary["skipped"] is True
            assert summary["processed"] == 0
        finally:
            lock.unlink(missing_ok=True)

    def test_worker_lock_follows_configured_cognikernel_dir(self, tmp_path, monkeypatch):
        """Audit P2: the lock lives under the configured cognikernel_dir, never a
        hardcoded ~/.cognikernel — needed for read-only homes and relocated data."""
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
        from cognikernel.config import Config
        from cognikernel.integration.session import _acquire_worker_lock

        cfg = Config.load()
        lock = _acquire_worker_lock("pidX", cognikernel_dir=cfg.cognikernel_dir)
        try:
            assert lock is not None
            assert lock.parent == cfg.cognikernel_dir / "locks"
            assert str(tmp_path / "data") in str(lock)
        finally:
            lock.unlink(missing_ok=True)

    def test_capture_skips_when_no_new_content(self, tmp_path):
        """Gamma post-mortem F4: identical content after cursor advance must not
        re-store the full transcript (sha collision, burned ids)."""
        project_path = _make_project(tmp_path)
        from cognikernel.integration.session import session_capture, process_jobs

        raw = _jsonl(10)
        r1 = session_capture(str(project_path), "test-nonew", raw)
        assert r1["job_id"] is not None
        process_jobs(str(project_path))  # advances cursor

        r2 = session_capture(str(project_path), "test-nonew", raw)  # same content
        assert r2["job_id"] is None  # skipped — nothing new


# ── session_capture speed ─────────────────────────────────────────────────────

class TestCaptureSpeed:
    def test_session_capture_is_fast(self, tmp_path, monkeypatch):
        """session_capture must complete well under 500ms — it's the hook fast path.

        COGNIKERNEL_DIR is isolated so the timing measures capture itself, not the
        machine's real store population (resolve_project_id's alias scan on a
        new project walks projects_dir; against a populated real ~/.cognikernel
        that is machine-state-dependent noise, not capture cost).
        """
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
        project_path = _make_project(tmp_path)
        from cognikernel.integration.session import session_capture

        raw = _jsonl(50)  # simulate a mid-session JSONL
        start = time.monotonic()
        result = session_capture(str(project_path), "test-speed", raw)
        elapsed_ms = (time.monotonic() - start) * 1000

        assert result["job_id"] > 0
        assert elapsed_ms < 500, f"session_capture took {elapsed_ms:.0f}ms (target <500ms)"

    def test_session_capture_does_not_write_events(self, tmp_path):
        """Capture must not write any events — processing is deferred to worker."""
        project_path = _make_project(tmp_path)
        from cognikernel.integration.session import session_capture
        from cognikernel.config import Config

        raw = _jsonl(20)
        session_capture(str(project_path), "test-no-events", raw)

        config = Config.load(project_path=str(project_path))
        pid = hash_project_path(str(project_path))
        db_path = get_db_path(config, pid)
        with get_connection(db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM events WHERE session_id='test-no-events'"
            ).fetchone()[0]
        assert count == 0


# ── cursor advance ────────────────────────────────────────────────────────────

class TestCursorAdvance:
    def test_process_jobs_advances_cursor(self, tmp_path):
        """After process_jobs, the ingest cursor must reflect the full line count."""
        project_path = _make_project(tmp_path)
        from cognikernel.integration.session import session_capture, process_jobs
        from cognikernel.config import Config

        raw = _jsonl(30)
        line_count = len([ln for ln in raw.splitlines() if ln.strip()])
        session_id = "test-cursor"

        session_capture(str(project_path), session_id, raw)
        process_jobs(str(project_path))

        config = Config.load(project_path=str(project_path))
        pid = hash_project_path(str(project_path))
        db_path = get_db_path(config, pid)
        with get_connection(db_path) as conn:
            cursor = get_cursor(conn, pid, session_id)

        assert cursor is not None
        assert cursor.last_line_count == line_count


class TestOrphanRecovery:
    def test_dead_claimant_job_requeued_immediately(self, tmp_path):
        """I7d: a claimed/running job whose claimant pid is dead must requeue
        with NO grace period — time-graced recovery left the queue invisible
        to the MCP drainer for 10 minutes per killed hook-drain."""
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db)); conn.row_factory = sqlite3.Row
        run_migrations(conn)
        from cognikernel.storage.jobs import recover_orphaned_jobs, claim_next_job

        ev_id = store_evidence(conn, "p", "s", "transcript", b"data")
        enqueue_extraction(conn, "p", "s", ev_id, "extract.transcript")
        job = claim_next_job(conn, "extract.transcript", "worker-999999")  # dead pid
        assert job is not None and job.state == "claimed"

        n = recover_orphaned_jobs(conn, pid_alive=lambda pid: False)
        assert n == 1
        assert get_job(conn, job.id).state == "queued"

    def test_live_claimant_job_untouched(self, tmp_path):
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db)); conn.row_factory = sqlite3.Row
        run_migrations(conn)
        from cognikernel.storage.jobs import recover_orphaned_jobs, claim_next_job

        ev_id = store_evidence(conn, "p", "s", "transcript", b"data")
        enqueue_extraction(conn, "p", "s", ev_id, "extract.transcript")
        job = claim_next_job(conn, "extract.transcript", "worker-12345")
        n = recover_orphaned_jobs(conn, pid_alive=lambda pid: True)
        assert n == 0
        assert get_job(conn, job.id).state == "claimed"
