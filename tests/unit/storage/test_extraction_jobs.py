from __future__ import annotations

import sqlite3

import pytest

from memlora.storage.evidence import store_evidence
from memlora.storage.jobs import (
    TERMINAL_STATES,
    ack_stage,
    claim_next_job,
    enqueue_extraction,
    fail_job,
    get_job,
    list_jobs,
    reclaim_stale_jobs,
    replay_dead_letter,
)


def _evidence_id(conn: sqlite3.Connection) -> int:
    return store_evidence(conn, "proj1", "sess1", "transcript", b"content")


def test_enqueue_extraction_is_idempotent_for_evidence_and_category(
    conn: sqlite3.Connection,
) -> None:
    evidence_id = _evidence_id(conn)

    first = enqueue_extraction(conn, "proj1", "sess1", evidence_id, "extract.transcript")
    second = enqueue_extraction(conn, "proj1", "sess1", evidence_id, "extract.transcript")

    assert first == second
    assert conn.execute("SELECT COUNT(*) FROM extraction_jobs").fetchone()[0] == 1


def test_enqueue_extraction_creates_separate_jobs_per_session(
    conn: sqlite3.Connection,
) -> None:
    """Same content across sessions = same evidence (content-deduped) but
    distinct jobs — each session has its own workflow run."""
    evidence_id = _evidence_id(conn)

    j1 = enqueue_extraction(conn, "proj1", "sess1", evidence_id, "extract.transcript")
    j2 = enqueue_extraction(conn, "proj1", "sess2", evidence_id, "extract.transcript")

    assert j1 != j2
    assert conn.execute("SELECT COUNT(*) FROM extraction_jobs").fetchone()[0] == 2


def test_claim_next_job_marks_job_claimed(conn: sqlite3.Connection) -> None:
    job_id = enqueue_extraction(conn, "proj1", "sess1", _evidence_id(conn), "extract.transcript")

    job = claim_next_job(conn, "extract.transcript", claimant="worker-a")

    assert job is not None
    assert job.id == job_id
    assert job.state == "claimed"
    assert job.claimed_by == "worker-a"


def test_ack_stage_records_output_and_advances_state(conn: sqlite3.Connection) -> None:
    job_id = enqueue_extraction(conn, "proj1", "sess1", _evidence_id(conn), "extract.transcript")

    ack_stage(conn, job_id, "OBSERVED", output_ref="raw_evidence:1")
    ack_stage(conn, job_id, "PARSED", output_ref="events:2")
    ack_stage(conn, job_id, "COMPLETED", output_ref="stats")

    rows = conn.execute(
        "SELECT stage, output_ref FROM extraction_job_acks WHERE job_id=? ORDER BY completed_at",
        (job_id,),
    ).fetchall()
    assert [r["stage"] for r in rows] == ["OBSERVED", "PARSED", "COMPLETED"]
    assert rows[1]["output_ref"] == "events:2"
    assert get_job(conn, job_id).state == "completed"


def test_fail_job_dead_letters_poison_input_immediately(conn: sqlite3.Connection) -> None:
    job_id = enqueue_extraction(conn, "proj1", "sess1", _evidence_id(conn), "extract.transcript")

    fail_job(conn, job_id, "POISON_INPUT", "bad transcript")

    job = get_job(conn, job_id)
    assert job.state == "dead_lettered"
    assert job.failure_class == "POISON_INPUT"
    assert job.last_error == "bad transcript"


def test_fail_job_retries_transient_then_dead_letters(conn: sqlite3.Connection) -> None:
    job_id = enqueue_extraction(
        conn,
        "proj1",
        "sess1",
        _evidence_id(conn),
        "extract.transcript",
        max_attempts=2,
    )

    fail_job(conn, job_id, "TRANSIENT", "locked")
    assert get_job(conn, job_id).state == "retryable_failure"

    fail_job(conn, job_id, "TRANSIENT", "locked again")
    job = get_job(conn, job_id)
    assert job.state == "dead_lettered"
    assert job.attempts == 2


def test_reclaim_stale_jobs_returns_old_claim_to_queue(conn: sqlite3.Connection) -> None:
    job_id = enqueue_extraction(conn, "proj1", "sess1", _evidence_id(conn), "extract.transcript")
    claim_next_job(conn, "extract.transcript", claimant="worker-a")
    conn.execute("UPDATE extraction_jobs SET claimed_at=1 WHERE id=?", (job_id,))
    conn.commit()

    reclaimed = reclaim_stale_jobs(conn, stale_after_ms=60_000, now_ms=120_000)

    assert reclaimed == 1
    job = get_job(conn, job_id)
    assert job.state == "queued"
    assert job.claimed_by is None


def test_replay_dead_letter_resets_job_to_queued(conn: sqlite3.Connection) -> None:
    job_id = enqueue_extraction(conn, "proj1", "sess1", _evidence_id(conn), "extract.transcript")
    fail_job(conn, job_id, "POISON_INPUT", "bad transcript")

    replayed = replay_dead_letter(conn, job_id)

    assert replayed == job_id
    job = get_job(conn, job_id)
    assert job.state == "queued"
    assert job.failure_class is None
    assert job.last_error is None
    assert job.attempts == 0


def test_list_jobs_filters_by_state(conn: sqlite3.Connection) -> None:
    job_id = enqueue_extraction(conn, "proj1", "sess1", _evidence_id(conn), "extract.transcript")
    fail_job(conn, job_id, "POISON_INPUT", "bad transcript")

    jobs = list_jobs(conn, "proj1", state="dead_lettered")

    assert [j.id for j in jobs] == [job_id]


# ── monotonicity guard (plan §6.2 invariant) ──────────────────────────────────

def test_terminal_states_contains_expected_members() -> None:
    assert TERMINAL_STATES == {
        "completed",
        "dead_lettered",
        "skipped_policy",
        "superseded_job",
    }


def test_ack_stage_rejects_dead_lettered_job(conn: sqlite3.Connection) -> None:
    job_id = enqueue_extraction(conn, "proj1", "sess1", _evidence_id(conn), "extract.transcript")
    fail_job(conn, job_id, "POISON_INPUT", "bad transcript")

    with pytest.raises(ValueError, match="terminal state"):
        ack_stage(conn, job_id, "COMPLETED")

    job = get_job(conn, job_id)
    assert job.state == "dead_lettered"
    assert job.stage == "OBSERVED"


def test_ack_stage_rejects_completed_job(conn: sqlite3.Connection) -> None:
    job_id = enqueue_extraction(conn, "proj1", "sess1", _evidence_id(conn), "extract.transcript")
    ack_stage(conn, job_id, "COMPLETED")

    with pytest.raises(ValueError, match="terminal state"):
        ack_stage(conn, job_id, "PARSED")


def test_fail_job_rejects_completed_job(conn: sqlite3.Connection) -> None:
    job_id = enqueue_extraction(conn, "proj1", "sess1", _evidence_id(conn), "extract.transcript")
    ack_stage(conn, job_id, "COMPLETED")

    with pytest.raises(ValueError, match="terminal state"):
        fail_job(conn, job_id, "TRANSIENT", "should not apply")

    job = get_job(conn, job_id)
    assert job.state == "completed"
    assert job.failure_class is None
    assert job.last_error is None


def test_fail_job_rejects_dead_lettered_job(conn: sqlite3.Connection) -> None:
    job_id = enqueue_extraction(conn, "proj1", "sess1", _evidence_id(conn), "extract.transcript")
    fail_job(conn, job_id, "POISON_INPUT", "bad transcript")

    with pytest.raises(ValueError, match="terminal state"):
        fail_job(conn, job_id, "TRANSIENT", "second failure")

    job = get_job(conn, job_id)
    assert job.attempts == 1
    assert job.failure_class == "POISON_INPUT"


def test_replay_is_the_only_escape_from_dead_lettered(conn: sqlite3.Connection) -> None:
    job_id = enqueue_extraction(conn, "proj1", "sess1", _evidence_id(conn), "extract.transcript")
    fail_job(conn, job_id, "POISON_INPUT", "bad transcript")

    replay_dead_letter(conn, job_id)
    ack_stage(conn, job_id, "COMPLETED")  # must not raise

    assert get_job(conn, job_id).state == "completed"
