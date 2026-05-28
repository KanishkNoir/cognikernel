"""Deny-timer table for the 60-second retry escape hatch — Stage C0.

A skeleton-fresh Read is denied on first attempt; if Claude retries the same
Read within `window_ms`, the second attempt is allowed and tagged
`body_needed_retry` in `read_session_cache`. This module owns the timer rows.

The timer is per (project, session, file). A re-denial overwrites the prior
row's `denied_at` so the window restarts. After the second allowance, callers
should `clear()` the row so a future denial cycle starts cleanly.

Cleanup runs cheaply on each PreToolUse:Read call (`cleanup_old`) — rows older
than ~5 minutes are safely abandoned.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

# 60s default retry window — matches the user-facing semantics described in v2 plan §2.
DEFAULT_RETRY_WINDOW_MS: int = 60 * 1000

# 5min cleanup TTL — rows past this are stale; longer than the retry window so
# we never accidentally evict an active deny.
DEFAULT_CLEANUP_TTL_MS: int = 5 * 60 * 1000

VALID_REASONS = frozenset({"skeleton_fresh"})


@dataclass(frozen=True)
class DeniedRead:
    project_id: str
    session_id: str
    file_path: str
    denied_at: int
    reason: str


def record(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    file_path: str,
    *,
    reason: str = "skeleton_fresh",
    now_ms: int | None = None,
) -> None:
    """Insert or refresh a denial row. Re-denial bumps denied_at."""
    if reason not in VALID_REASONS:
        raise ValueError(f"invalid reason: {reason!r}")
    now = _now_ms() if now_ms is None else now_ms
    conn.execute(
        """
        INSERT INTO denied_reads (project_id, session_id, file_path, denied_at, reason)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (project_id, session_id, file_path) DO UPDATE SET
            denied_at = excluded.denied_at,
            reason = excluded.reason
        """,
        (project_id, session_id, file_path, now, reason),
    )
    conn.commit()


def was_denied_within(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    file_path: str,
    *,
    window_ms: int = DEFAULT_RETRY_WINDOW_MS,
    now_ms: int | None = None,
) -> bool:
    """Return True if the file was denied within the retry window.

    Used by PreToolUse:Read to decide whether a re-attempt should be allowed
    as a 'body_needed_retry'. Strict less-than-or-equal-to on the window edge.
    """
    now = _now_ms() if now_ms is None else now_ms
    row = conn.execute(
        """
        SELECT denied_at FROM denied_reads
        WHERE project_id=? AND session_id=? AND file_path=?
        """,
        (project_id, session_id, file_path),
    ).fetchone()
    if row is None:
        return False
    return (now - row["denied_at"]) <= window_ms


def get(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    file_path: str,
) -> DeniedRead | None:
    """Return the denial row, or None if not present."""
    row = conn.execute(
        """
        SELECT project_id, session_id, file_path, denied_at, reason
        FROM denied_reads
        WHERE project_id=? AND session_id=? AND file_path=?
        """,
        (project_id, session_id, file_path),
    ).fetchone()
    if row is None:
        return None
    return DeniedRead(
        project_id=row["project_id"],
        session_id=row["session_id"],
        file_path=row["file_path"],
        denied_at=row["denied_at"],
        reason=row["reason"],
    )


def clear(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    file_path: str,
) -> bool:
    """Remove a specific denial row. Returns True if a row was deleted.

    Called after a successful body_needed_retry so the next denial cycle for
    the same file (in a hypothetical re-edit) starts clean.
    """
    cur = conn.execute(
        "DELETE FROM denied_reads WHERE project_id=? AND session_id=? AND file_path=?",
        (project_id, session_id, file_path),
    )
    conn.commit()
    return cur.rowcount > 0


def cleanup_old(
    conn: sqlite3.Connection,
    *,
    ttl_ms: int = DEFAULT_CLEANUP_TTL_MS,
    now_ms: int | None = None,
) -> int:
    """Delete denial rows older than ttl_ms. Returns the number of rows removed."""
    now = _now_ms() if now_ms is None else now_ms
    cutoff = now - ttl_ms
    cur = conn.execute(
        "DELETE FROM denied_reads WHERE denied_at < ?",
        (cutoff,),
    )
    conn.commit()
    return cur.rowcount


# ── internals ────────────────────────────────────────────────────────────────


def _now_ms() -> int:
    return int(time.time() * 1000)
