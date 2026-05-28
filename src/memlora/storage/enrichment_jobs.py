"""CRUD for the `enrichment_jobs` table — Stage A-5.

Tracks the lifecycle of LLM enrichment work: queued → claimed → completed |
partial | failed | skipped. Separate from `extraction_jobs` (the trie pipeline)
because the LLM enrichment has a different state machine and the schema
already differs (versioned extractor IDs).

This module is used by the two new MCP tools:
  - `get_unprocessed_evidence` queries via `list_pending_for_version()`
  - `store_extracted_events` marks completion via `mark_completed()` /
    `mark_partial()` and bumps `raw_evidence.llm_extractor_version` on full
    success.

The all-or-nothing version bump semantics live in store_extracted_events; this
module is purely state-machine CRUD with no business policy.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass


VALID_KINDS = frozenset({
    "llm_decision_extraction",
    "llm_constraint_inference",
})

VALID_STATES = frozenset({
    "queued",
    "claimed",
    "completed",
    "partial",
    "failed",
    "skipped",
})


@dataclass(frozen=True)
class EnrichmentJob:
    id: int
    project_id: str
    evidence_id: int
    enrichment_kind: str
    extractor_version: str
    state: str
    error: str
    queued_at: int
    completed_at: int


def enqueue(
    conn: sqlite3.Connection,
    project_id: str,
    evidence_id: int,
    *,
    enrichment_kind: str = "llm_decision_extraction",
    extractor_version: str,
    now_ms: int | None = None,
) -> int:
    """Insert a queued job (idempotent on the natural unique key).

    The UNIQUE (project_id, evidence_id, enrichment_kind, extractor_version)
    constraint means re-calling with identical args is a no-op — useful when
    the MCP tool retries after a partial-failure.
    """
    if enrichment_kind not in VALID_KINDS:
        raise ValueError(f"invalid enrichment_kind: {enrichment_kind!r}")
    now = _now_ms() if now_ms is None else now_ms

    conn.execute(
        """
        INSERT OR IGNORE INTO enrichment_jobs
            (project_id, evidence_id, enrichment_kind, extractor_version,
             state, error, queued_at, completed_at)
        VALUES (?, ?, ?, ?, 'queued', '', ?, 0)
        """,
        (project_id, evidence_id, enrichment_kind, extractor_version, now),
    )
    conn.commit()

    row = conn.execute(
        """
        SELECT id FROM enrichment_jobs
        WHERE project_id=? AND evidence_id=? AND enrichment_kind=? AND extractor_version=?
        """,
        (project_id, evidence_id, enrichment_kind, extractor_version),
    ).fetchone()
    return row["id"]


def mark_completed(
    conn: sqlite3.Connection,
    job_id: int,
    *,
    now_ms: int | None = None,
) -> None:
    """Transition a job to `completed`. Final state."""
    now = _now_ms() if now_ms is None else now_ms
    conn.execute(
        "UPDATE enrichment_jobs SET state='completed', error='', completed_at=? WHERE id=?",
        (now, job_id),
    )
    conn.commit()


def mark_partial(
    conn: sqlite3.Connection,
    job_id: int,
    error: str,
    *,
    now_ms: int | None = None,
) -> None:
    """Transition a job to `partial` — some events inserted, some errored.

    The job stays at `partial` until the caller retries with the same
    (evidence_id, version) pair; only on full success does it advance.
    """
    now = _now_ms() if now_ms is None else now_ms
    conn.execute(
        "UPDATE enrichment_jobs SET state='partial', error=?, completed_at=? WHERE id=?",
        (error[:500], now, job_id),
    )
    conn.commit()


def mark_failed(
    conn: sqlite3.Connection,
    job_id: int,
    error: str,
    *,
    now_ms: int | None = None,
) -> None:
    """Transition a job to `failed` — nothing useful was extracted."""
    now = _now_ms() if now_ms is None else now_ms
    conn.execute(
        "UPDATE enrichment_jobs SET state='failed', error=?, completed_at=? WHERE id=?",
        (error[:500], now, job_id),
    )
    conn.commit()


def get(conn: sqlite3.Connection, job_id: int) -> EnrichmentJob | None:
    row = conn.execute(
        "SELECT * FROM enrichment_jobs WHERE id=?",
        (job_id,),
    ).fetchone()
    return _row_to_job(row) if row else None


def list_pending_for_version(
    conn: sqlite3.Connection,
    project_id: str,
    extractor_version: str,
    *,
    limit: int = 5,
) -> list[EnrichmentJob]:
    """Return up to `limit` queued/partial jobs for this version.

    Includes 'partial' so retries pick them up automatically; excludes
    completed/failed/skipped (those are terminal for this version)."""
    rows = conn.execute(
        """
        SELECT * FROM enrichment_jobs
        WHERE project_id=? AND extractor_version=?
          AND state IN ('queued', 'partial')
        ORDER BY queued_at ASC
        LIMIT ?
        """,
        (project_id, extractor_version, limit),
    ).fetchall()
    return [_row_to_job(r) for r in rows]


# ── internals ────────────────────────────────────────────────────────────────


def _row_to_job(row: sqlite3.Row) -> EnrichmentJob:
    return EnrichmentJob(
        id=row["id"],
        project_id=row["project_id"],
        evidence_id=row["evidence_id"],
        enrichment_kind=row["enrichment_kind"],
        extractor_version=row["extractor_version"],
        state=row["state"],
        error=row["error"],
        queued_at=row["queued_at"],
        completed_at=row["completed_at"],
    )


def _now_ms() -> int:
    return int(time.time() * 1000)
