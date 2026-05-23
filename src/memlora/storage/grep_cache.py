"""Grep result cache — deduplicate identical Grep calls within a session."""
from __future__ import annotations

import fnmatch
import hashlib
import sqlite3
import time


def cache_key(pattern: str, path: str, glob: str) -> str:
    """Return a deterministic 24-char hex key for a grep (pattern, path, glob) triple."""
    raw = f"{pattern}\x00{path}\x00{glob}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def lookup_grep_result(
    conn: sqlite3.Connection,
    project_id: str,
    pattern: str,
    path: str,
    glob: str,
) -> str | None:
    """Return the cached grep result text, or None on cache miss."""
    key = cache_key(pattern, path, glob)
    row = conn.execute(
        "SELECT result_text FROM grep_cache WHERE project_id = ? AND cache_key = ?",
        (project_id, key),
    ).fetchone()
    if row is None:
        return None
    conn.execute(
        "UPDATE grep_cache SET hit_count = hit_count + 1 WHERE project_id = ? AND cache_key = ?",
        (project_id, key),
    )
    conn.commit()
    return row[0]


def store_grep_result(
    conn: sqlite3.Connection,
    project_id: str,
    pattern: str,
    path: str,
    glob: str,
    result_text: str,
) -> None:
    """Upsert a grep result into the cache. Re-storing resets hit_count to 0."""
    key = cache_key(pattern, path, glob)
    now = int(time.time() * 1000)
    conn.execute(
        """
        INSERT INTO grep_cache
            (project_id, cache_key, pattern, path_filter, glob_filter, result_text, cached_at, hit_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT (project_id, cache_key) DO UPDATE SET
            result_text = excluded.result_text,
            cached_at   = excluded.cached_at,
            hit_count   = 0
        """,
        (project_id, key, pattern, path, glob, result_text, now),
    )
    conn.commit()


def invalidate_project_cache(
    conn: sqlite3.Connection,
    project_id: str,
    changed_path: str | None = None,
) -> int:
    """Remove grep cache entries invalidated by a file change.

    When changed_path is given, only deletes rows whose path_filter or glob_filter
    matches the changed file. Rows with no filter (whole-project greps) are always
    invalidated. When changed_path is None, deletes all rows for the project.
    """
    if changed_path is None:
        cursor = conn.execute(
            "DELETE FROM grep_cache WHERE project_id = ?",
            (project_id,),
        )
        conn.commit()
        return cursor.rowcount

    # Load all rows to check path/glob filters in Python (table is small per project)
    rows = conn.execute(
        "SELECT cache_key, path_filter, glob_filter FROM grep_cache WHERE project_id = ?",
        (project_id,),
    ).fetchall()

    to_delete = []
    for row in rows:
        path_filter = row["path_filter"]
        glob_filter = row["glob_filter"]
        if not path_filter and not glob_filter:
            # No filter = whole-project grep; always invalidate on any change
            to_delete.append(row["cache_key"])
        elif path_filter and (
            path_filter == changed_path
            or changed_path.startswith(path_filter.rstrip("/") + "/")
        ):
            to_delete.append(row["cache_key"])
        elif glob_filter and fnmatch.fnmatch(changed_path, glob_filter):
            to_delete.append(row["cache_key"])

    if not to_delete:
        return 0

    placeholders = ",".join("?" * len(to_delete))
    cursor = conn.execute(
        f"DELETE FROM grep_cache WHERE project_id = ? AND cache_key IN ({placeholders})",
        (project_id, *to_delete),
    )
    conn.commit()
    return cursor.rowcount
