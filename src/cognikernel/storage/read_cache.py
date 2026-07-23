"""Per-session read cache — Stage C0.

Populated by the PostToolUse:Read hook on every successful Read. Queried by the
PreToolUse:Read hook to deny re-reads of files already cited in the current
session's context. The hook layer's STEP 1 (re-read check) runs against this
table before any skeleton-based decision logic.

Idempotency: record_read() is safe to call repeatedly for the same
(project_id, session_id, file_path) — the row's read_count increments and
last_read_at advances; first_read_at stays pinned.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

VALID_OUTCOMES = frozenset({"ok", "body_needed_retry"})

# 24h default for read-cache row eviction. Configurable via Config in the hook.
DEFAULT_TTL_MS: int = 24 * 60 * 60 * 1000


@dataclass(frozen=True)
class ReadCacheEntry:
    project_id: str
    session_id: str
    file_path: str
    first_read_at: int
    last_read_at: int
    read_count: int
    last_read_outcome: str


def record_read(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    file_path: str,
    *,
    outcome: str = "ok",
    now_ms: int | None = None,
) -> None:
    """Record a successful Read. Upsert: first call inserts, later calls bump counters.

    Callers MUST pass canonical relative paths. The hook layer normalizes before
    invoking this function.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(f"invalid outcome: {outcome!r}")
    now = _now_ms() if now_ms is None else now_ms
    conn.execute(
        """
        INSERT INTO read_session_cache
            (project_id, session_id, file_path, first_read_at, last_read_at,
             read_count, last_read_outcome)
        VALUES (?, ?, ?, ?, ?, 1, ?)
        ON CONFLICT (project_id, session_id, file_path) DO UPDATE SET
            last_read_at = excluded.last_read_at,
            read_count = read_count + 1,
            last_read_outcome = excluded.last_read_outcome
        """,
        (project_id, session_id, file_path, now, now, outcome),
    )
    conn.commit()


def get_read(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    file_path: str,
) -> ReadCacheEntry | None:
    """Return the cache row for this (project, session, file), or None if absent."""
    row = conn.execute(
        """
        SELECT project_id, session_id, file_path, first_read_at, last_read_at,
               read_count, last_read_outcome
        FROM read_session_cache
        WHERE project_id=? AND session_id=? AND file_path=?
        """,
        (project_id, session_id, file_path),
    ).fetchone()
    if row is None:
        return None
    return ReadCacheEntry(
        project_id=row["project_id"],
        session_id=row["session_id"],
        file_path=row["file_path"],
        first_read_at=row["first_read_at"],
        last_read_at=row["last_read_at"],
        read_count=row["read_count"],
        last_read_outcome=row["last_read_outcome"],
    )


def was_read_in_session(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    file_path: str,
) -> tuple[bool, str | None]:
    """Hook-facing convenience: returns (was_read, last_outcome).

    `was_read=False` ⇒ outcome is None.
    `was_read=True` ⇒ outcome is 'ok' or 'body_needed_retry'.
    """
    entry = get_read(conn, project_id, session_id, file_path)
    if entry is None:
        return False, None
    return True, entry.last_read_outcome


def cleanup_old(
    conn: sqlite3.Connection,
    *,
    ttl_ms: int = DEFAULT_TTL_MS,
    now_ms: int | None = None,
) -> int:
    """Delete cache rows older than ttl_ms. Returns the number of rows removed."""
    now = _now_ms() if now_ms is None else now_ms
    cutoff = now - ttl_ms
    cur = conn.execute(
        "DELETE FROM read_session_cache WHERE first_read_at < ?",
        (cutoff,),
    )
    conn.commit()
    return cur.rowcount


def clear_session(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
) -> int:
    """Drop all cache entries for a session. Returns the number of rows removed."""
    cur = conn.execute(
        "DELETE FROM read_session_cache WHERE project_id=? AND session_id=?",
        (project_id, session_id),
    )
    conn.commit()
    return cur.rowcount


# ── internals ────────────────────────────────────────────────────────────────


def _now_ms() -> int:
    return int(time.time() * 1000)
