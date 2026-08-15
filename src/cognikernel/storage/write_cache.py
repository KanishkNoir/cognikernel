"""Per-session write cache — real Write/Edit/MultiEdit touches (Commit A of #31).

Populated by the PostToolUse hook on every successful Write/Edit/MultiEdit.
Collection only: nothing in `symbols/projection.py` reads this table yet. It
exists to accumulate real per-session write data before any change to
`_compute_hot_files` or the `_HOT_WEIGHT` scoring term — see design doc §9.6.

Idempotency: record_write() is safe to call repeatedly for the same
(project_id, session_id, file_path) — the row's write_count increments and
last_write_at advances; first_write_at stays pinned. Mirrors read_cache.py.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

VALID_ACTIONS = frozenset({"Write", "Edit", "MultiEdit"})


@dataclass(frozen=True)
class WriteCacheEntry:
    project_id: str
    session_id: str
    file_path: str
    first_write_at: int
    last_write_at: int
    write_count: int
    last_write_action: str


def record_write(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    file_path: str,
    *,
    action: str = "Write",
    now_ms: int | None = None,
) -> None:
    """Record a successful Write/Edit/MultiEdit. Upsert: first call inserts,
    later calls bump counters.

    Callers MUST pass canonical relative paths. The hook layer normalizes
    before invoking this function.
    """
    if action not in VALID_ACTIONS:
        raise ValueError(f"invalid action: {action!r}")
    now = _now_ms() if now_ms is None else now_ms
    conn.execute(
        """
        INSERT INTO write_session_cache
            (project_id, session_id, file_path, first_write_at, last_write_at,
             write_count, last_write_action)
        VALUES (?, ?, ?, ?, ?, 1, ?)
        ON CONFLICT (project_id, session_id, file_path) DO UPDATE SET
            last_write_at = excluded.last_write_at,
            write_count = write_count + 1,
            last_write_action = excluded.last_write_action
        """,
        (project_id, session_id, file_path, now, now, action),
    )
    conn.commit()


def get_write(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    file_path: str,
) -> WriteCacheEntry | None:
    """Return the cache row for this (project, session, file), or None if absent."""
    row = conn.execute(
        """
        SELECT project_id, session_id, file_path, first_write_at, last_write_at,
               write_count, last_write_action
        FROM write_session_cache
        WHERE project_id=? AND session_id=? AND file_path=?
        """,
        (project_id, session_id, file_path),
    ).fetchone()
    if row is None:
        return None
    return WriteCacheEntry(
        project_id=row["project_id"],
        session_id=row["session_id"],
        file_path=row["file_path"],
        first_write_at=row["first_write_at"],
        last_write_at=row["last_write_at"],
        write_count=row["write_count"],
        last_write_action=row["last_write_action"],
    )


def was_written_in_session(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    file_path: str,
) -> tuple[bool, str | None]:
    """Convenience: returns (was_written, last_action).

    `was_written=False` ⇒ action is None.
    `was_written=True` ⇒ action is 'Write', 'Edit', or 'MultiEdit'.
    """
    entry = get_write(conn, project_id, session_id, file_path)
    if entry is None:
        return False, None
    return True, entry.last_write_action


def clear_session(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
) -> int:
    """Drop all cache entries for a session. Returns the number of rows removed."""
    cur = conn.execute(
        "DELETE FROM write_session_cache WHERE project_id=? AND session_id=?",
        (project_id, session_id),
    )
    conn.commit()
    return cur.rowcount


# ── internals ────────────────────────────────────────────────────────────────


def _now_ms() -> int:
    return int(time.time() * 1000)
