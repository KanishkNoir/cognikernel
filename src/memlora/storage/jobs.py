from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass


VALID_FAILURE_CLASSES = frozenset({
    "TRANSIENT",
    "POISON_INPUT",
    "SCHEMA_MISMATCH",
    "EXTRACTOR_BUG",
    "IO_MISSING",
    "TIMEOUT",
})

NON_RETRYABLE_FAILURE_CLASSES = frozenset({
    "POISON_INPUT",
    "SCHEMA_MISMATCH",
})

# States from which no further transition is legal except via the explicit
# replay_dead_letter() escape hatch. Plan §6.2 invariant: "State transitions
# are monotone (no `dead_lettered → running`)."
TERMINAL_STATES = frozenset({
    "completed",
    "dead_lettered",
    "skipped_policy",
    "superseded_job",
})


@dataclass(frozen=True)
class ExtractionJob:
    id: int
    project_id: str
    session_id: str
    evidence_id: int
    trace_id: str
    job_category: str
    stage: str
    state: str
    failure_class: str | None
    last_error: str | None
    claimed_by: str | None
    claimed_at: int | None
    attempts: int
    max_attempts: int
    soft_timeout_ms: int
    hard_timeout_ms: int
    created_at: int
    updated_at: int


def enqueue_extraction(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    evidence_id: int,
    job_category: str,
    trace_id: str | None = None,
    max_attempts: int = 3,
    soft_timeout_ms: int = 60_000,
    hard_timeout_ms: int = 120_000,
) -> int:
    now = _now_ms()
    trace = trace_id or str(uuid.uuid7() if hasattr(uuid, "uuid7") else uuid.uuid4())
    conn.execute(
        """
        INSERT OR IGNORE INTO extraction_jobs
            (project_id, session_id, evidence_id, trace_id, job_category,
             stage, state, attempts, max_attempts, soft_timeout_ms,
             hard_timeout_ms, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, 'OBSERVED', 'queued', 0, ?, ?, ?, ?, ?)
        """,
        (
            project_id,
            session_id,
            evidence_id,
            trace,
            job_category,
            max_attempts,
            soft_timeout_ms,
            hard_timeout_ms,
            now,
            now,
        ),
    )
    conn.commit()
    row = conn.execute(
        """
        SELECT id FROM extraction_jobs
        WHERE project_id=? AND session_id=? AND evidence_id=? AND job_category=?
        """,
        (project_id, session_id, evidence_id, job_category),
    ).fetchone()
    return row["id"]


def claim_next_job(
    conn: sqlite3.Connection,
    job_category: str,
    claimant: str,
    now_ms: int | None = None,
) -> ExtractionJob | None:
    now = now_ms if now_ms is not None else _now_ms()
    with conn:
        row = conn.execute(
            """
            SELECT * FROM extraction_jobs
            WHERE job_category = ?
              AND state IN ('queued', 'retryable_failure')
            ORDER BY updated_at ASC, id ASC
            LIMIT 1
            """,
            (job_category,),
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            """
            UPDATE extraction_jobs
            SET state='claimed', claimed_by=?, claimed_at=?, updated_at=?
            WHERE id=?
            """,
            (claimant, now, now, row["id"]),
        )
    return get_job(conn, row["id"])


def ack_stage(
    conn: sqlite3.Connection,
    job_id: int,
    stage: str,
    output_ref: str | None = None,
    now_ms: int | None = None,
) -> None:
    now = now_ms if now_ms is not None else _now_ms()
    state = "completed" if stage == "COMPLETED" else "running"
    with conn:
        row = conn.execute(
            "SELECT state FROM extraction_jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Unknown extraction job {job_id}")
        if row["state"] in TERMINAL_STATES:
            raise ValueError(
                f"Cannot ack stage {stage!r} on extraction job {job_id} "
                f"in terminal state {row['state']!r} — use replay_dead_letter "
                f"to revive a dead-lettered job."
            )
        conn.execute(
            """
            INSERT OR IGNORE INTO extraction_job_acks
                (job_id, stage, completed_at, output_ref)
            VALUES (?, ?, ?, ?)
            """,
            (job_id, stage, now, output_ref),
        )
        conn.execute(
            """
            UPDATE extraction_jobs
            SET stage=?, state=?, updated_at=?
            WHERE id=?
            """,
            (stage, state, now, job_id),
        )


def fail_job(
    conn: sqlite3.Connection,
    job_id: int,
    failure_class: str,
    error_message: str,
    now_ms: int | None = None,
) -> None:
    if failure_class not in VALID_FAILURE_CLASSES:
        raise ValueError(f"Unknown failure_class {failure_class!r}")
    now = now_ms if now_ms is not None else _now_ms()
    row = conn.execute(
        "SELECT attempts, max_attempts, state FROM extraction_jobs WHERE id=?",
        (job_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Unknown extraction job {job_id}")
    if row["state"] in TERMINAL_STATES:
        raise ValueError(
            f"Cannot fail extraction job {job_id} in terminal state "
            f"{row['state']!r}."
        )
    attempts = int(row["attempts"]) + 1
    state = (
        "dead_lettered"
        if failure_class in NON_RETRYABLE_FAILURE_CLASSES or attempts >= row["max_attempts"]
        else "retryable_failure"
    )
    conn.execute(
        """
        UPDATE extraction_jobs
        SET state=?, failure_class=?, last_error=?, attempts=?,
            claimed_by=NULL, claimed_at=NULL, updated_at=?
        WHERE id=?
        """,
        (state, failure_class, error_message, attempts, now, job_id),
    )
    conn.commit()


def reclaim_stale_jobs(
    conn: sqlite3.Connection,
    stale_after_ms: int,
    now_ms: int | None = None,
) -> int:
    now = now_ms if now_ms is not None else _now_ms()
    cutoff = now - stale_after_ms
    result = conn.execute(
        """
        UPDATE extraction_jobs
        SET state='queued', claimed_by=NULL, claimed_at=NULL, updated_at=?
        WHERE state='claimed'
          AND claimed_at IS NOT NULL
          AND claimed_at < ?
        """,
        (now, cutoff),
    )
    conn.commit()
    return result.rowcount


def replay_dead_letter(conn: sqlite3.Connection, job_id: int) -> int:
    conn.execute(
        """
        UPDATE extraction_jobs
        SET state='queued', failure_class=NULL, last_error=NULL, attempts=0,
            claimed_by=NULL, claimed_at=NULL, updated_at=?
        WHERE id=? AND state='dead_lettered'
        """,
        (_now_ms(), job_id),
    )
    conn.commit()
    return job_id


def get_job(conn: sqlite3.Connection, job_id: int) -> ExtractionJob:
    row = conn.execute("SELECT * FROM extraction_jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        raise ValueError(f"Unknown extraction job {job_id}")
    return _row_to_job(row)


def list_jobs(
    conn: sqlite3.Connection,
    project_id: str,
    state: str | None = None,
    limit: int = 20,
) -> list[ExtractionJob]:
    if state is None:
        rows = conn.execute(
            """
            SELECT * FROM extraction_jobs
            WHERE project_id=?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM extraction_jobs
            WHERE project_id=? AND state=?
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (project_id, state, limit),
        ).fetchall()
    return [_row_to_job(row) for row in rows]


def _row_to_job(row: sqlite3.Row) -> ExtractionJob:
    return ExtractionJob(
        id=row["id"],
        project_id=row["project_id"],
        session_id=row["session_id"],
        evidence_id=row["evidence_id"],
        trace_id=row["trace_id"],
        job_category=row["job_category"],
        stage=row["stage"],
        state=row["state"],
        failure_class=row["failure_class"],
        last_error=row["last_error"],
        claimed_by=row["claimed_by"],
        claimed_at=row["claimed_at"],
        attempts=row["attempts"],
        max_attempts=row["max_attempts"],
        soft_timeout_ms=row["soft_timeout_ms"],
        hard_timeout_ms=row["hard_timeout_ms"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _now_ms() -> int:
    return int(time.time() * 1000)
