"""Worker crash → replay must not corrupt state (audit P1 / #58, integration level).

The unit test (tests/unit/delta/test_merge.py) proves execute_merge's guard in
isolation. This drives the REAL worker: the same evidence is processed a second
time through claim -> slice -> extract -> merge. Without the evidence-provenance
guard a re-merge would inflate mention_count (the re-run lands under a different
session, a cross-session restatement) and re-apply decay; the guard must make it
a no-op while the job still completes.

A fresh queued job for the same evidence is the faithful trigger: enqueue is
INSERT-OR-IGNORE keyed on (project, session, evidence), so a new session id
yields a genuinely re-processable job pointing at already-merged evidence — which
is exactly what the recovery paths (recover_orphaned_jobs / dead-letter replay)
hand the worker after a crash in the merge->cursor window.
"""
from __future__ import annotations

from cognikernel.storage.connection import get_connection
from cognikernel.storage.jobs import enqueue_extraction, fail_job, get_job


def _event_stats(db) -> tuple[int, int, float]:
    with get_connection(db) as conn:
        row = conn.execute(
            "SELECT COUNT(*) c, COALESCE(SUM(mention_count),0) m, "
            "COALESCE(SUM(weight),0.0) w FROM events"
        ).fetchone()
    return row["c"], row["m"], round(row["w"], 6)


class TestCrashReplayNoDrift:
    def test_replayed_evidence_does_not_drift(self, project, jsonl) -> None:
        from cognikernel.integration.session import process_jobs, session_capture

        raw = jsonl(8)
        job_id = session_capture(project.path, "sess-orig", raw)["job_id"]
        assert job_id is not None
        assert process_jobs(project.path)["processed"] >= 1

        baseline = _event_stats(project.db)
        assert baseline[0] > 0, "first pass should have created events"

        # Re-queue the SAME evidence as a fresh job (new session) — the state a
        # recovered worker replays after a crash in the merge->cursor window.
        with get_connection(project.db) as conn:
            evidence_id = get_job(conn, job_id).evidence_id
            replay_job = enqueue_extraction(
                conn, project.pid, "sess-replay", evidence_id, "extract.transcript"
            )
        assert replay_job != job_id, "expected a genuinely new queued job"

        replay_summary = process_jobs(project.path)

        # The merge re-ran but the provenance guard made it a no-op: no extra
        # events, no inflated mention_count, no second decay tick.
        assert _event_stats(project.db) == baseline, "crash-replay drifted"
        # And the worker still made forward progress (job completes, no failures).
        assert replay_summary["processed"] >= 1
        assert replay_summary["failed"] == 0
        with get_connection(project.db) as conn:
            assert get_job(conn, replay_job).state == "completed"

    def test_timeout_dead_letter_replay_does_not_drift(self, project, jsonl) -> None:
        """The auto-replay route: a TIMEOUT dead-letter (process-killed) replayed by
        process_jobs must also not double-apply its merge."""
        from cognikernel.integration.session import process_jobs, session_capture

        job_id = session_capture(project.path, "sess-orig", jsonl(6))["job_id"]
        assert process_jobs(project.path)["processed"] >= 1
        baseline = _event_stats(project.db)

        with get_connection(project.db) as conn:
            evidence_id = get_job(conn, job_id).evidence_id
            dl_job = enqueue_extraction(
                conn, project.pid, "sess-dl", evidence_id, "extract.transcript"
            )
        with get_connection(project.db) as conn:
            for _ in range(3):  # max_attempts -> dead_lettered
                fail_job(conn, dl_job, "TIMEOUT", "simulated kill")
            assert get_job(conn, dl_job).state == "dead_lettered"

        summary = process_jobs(project.path)
        assert summary["replayed"] >= 1
        assert _event_stats(project.db) == baseline
        with get_connection(project.db) as conn:
            assert get_job(conn, dl_job).state == "completed"


class TestCrashReplayPreservesHeadSha:
    """T-103 (#13): captured_at_sha must survive a crash+replay cycle exactly
    like every other event field the provenance guard already protects — a
    replay that silently dropped or altered it would make future drift
    detection trust a commit anchor that was never actually re-verified.
    """

    def test_replayed_evidence_keeps_the_original_sha(self, project, jsonl) -> None:
        from cognikernel.integration.session import process_jobs, session_capture

        sha = "d" * 40
        raw = jsonl(6)
        job_id = session_capture(project.path, "sess-orig", raw, head_sha=sha)["job_id"]
        assert job_id is not None
        assert process_jobs(project.path)["processed"] >= 1

        with get_connection(project.db) as conn:
            shas_before = [
                r["captured_at_sha"]
                for r in conn.execute("SELECT captured_at_sha FROM events").fetchall()
            ]
        assert shas_before, "first pass should have created events"
        assert all(s == sha for s in shas_before)

        # Re-queue the SAME evidence as a fresh job — the crash-recovery path.
        # Its raw_evidence row (and therefore its metadata.head_sha) is
        # untouched; only a NEW job is queued against it.
        with get_connection(project.db) as conn:
            evidence_id = get_job(conn, job_id).evidence_id
            replay_job = enqueue_extraction(
                conn, project.pid, "sess-replay", evidence_id, "extract.transcript"
            )
        replay_summary = process_jobs(project.path)
        assert replay_summary["failed"] == 0

        with get_connection(project.db) as conn:
            shas_after = [
                r["captured_at_sha"]
                for r in conn.execute("SELECT captured_at_sha FROM events").fetchall()
            ]
        # The provenance guard makes the replay a no-op (proven by
        # TestCrashReplayNoDrift above) — same events, same shas, not lost,
        # not overwritten by a NULL, not duplicated onto a second row.
        assert shas_after == shas_before
