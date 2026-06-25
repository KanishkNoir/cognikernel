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
    """Claim the oldest queued job for `job_category`. Returns None if none available.

    Uses optimistic concurrency: the UPDATE includes `AND state IN (...)` so that
    if two workers race on the same row, only the first commit succeeds (rowcount=1)
    and the second sees rowcount=0 and retries. Safe under concurrent SQLite writers.
    """
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
        result = conn.execute(
            """
            UPDATE extraction_jobs
            SET state='claimed', claimed_by=?, claimed_at=?, updated_at=?
            WHERE id=? AND state IN ('queued', 'retryable_failure')
            """,
            (claimant, now, now, row["id"]),
        )
        if result.rowcount == 0:
            # Another worker claimed it between our SELECT and UPDATE — skip.
            return None
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


def recover_orphaned_jobs(
    conn: sqlite3.Connection,
    pid_alive,
) -> int:
    """Immediately recover claimed/running jobs whose claimant process is dead.

    Hook-spawned drains are killed at the hook ceiling (subprocess timeout or
    Claude Code's hook timeout), leaving their job claimed/running with a dead
    claimant pid. Time-graced recovery (recover_stuck_running_jobs, 10 min)
    makes the queue invisible to the MCP drainer for that whole window — the
    GAMMA_CK_TEST run showed 30+ minute per-job lag from exactly this. The
    claimant string is "worker-{pid}", so pid liveness gives definitive,
    grace-free orphan detection. Jobs with no parseable claimant fall back to
    the time-graced paths.

    `pid_alive` is injected (callable pid->bool) to keep this module free of
    platform-specific process probing. Returns number of jobs recovered.
    """
    recovered = 0
    rows = conn.execute(
        "SELECT id, state, claimed_by FROM extraction_jobs "
        "WHERE state IN ('claimed','running') AND claimed_by IS NOT NULL"
    ).fetchall()
    now = _now_ms()
    for row in rows:
        claimant = row["claimed_by"] or ""
        if not claimant.startswith("worker-"):
            continue
        try:
            pid = int(claimant.split("-", 1)[1])
        except (ValueError, IndexError):
            continue
        if pid_alive(pid):
            continue
        conn.execute(
            """
            UPDATE extraction_jobs
            SET state='queued', claimed_by=NULL, claimed_at=NULL, updated_at=?
            WHERE id=? AND state IN ('claimed','running')
            """,
            (now, row["id"]),
        )
        recovered += 1
    if recovered:
        conn.commit()
    return recovered


def recover_stuck_running_jobs(
    conn: sqlite3.Connection,
    stale_after_ms: int = 5 * 60 * 1000,
    now_ms: int | None = None,
) -> int:
    """Transition stuck `running` jobs to `dead_lettered` so replay can recover them.

    A `running` job that hasn't been updated in `stale_after_ms` milliseconds was
    almost certainly killed mid-execution (Stop hook subprocess torn down when the
    session ended). Its raw_evidence is intact; it just needs a merge replay.
    Moving it to `dead_lettered` makes it eligible for `replay_dead_letter`.

    Default grace period: 5 minutes. Call at session_end / SessionStart to surface
    stuck jobs from prior runs before the next extraction starts.
    """
    now = now_ms if now_ms is not None else _now_ms()
    cutoff = now - stale_after_ms
    result = conn.execute(
        """
        UPDATE extraction_jobs
        SET state='dead_lettered',
            failure_class='TIMEOUT',
            last_error='subprocess killed mid-execution; recovered by recover_stuck_running_jobs',
            updated_at=?
        WHERE state='running'
          AND updated_at < ?
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
