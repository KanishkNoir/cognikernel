"""File-level skeleton authority — Stage C0.

Replaces the implicit `status` semantics scattered across `component_map`
payloads. Each row authoritatively answers two questions for the PreToolUse
hook (STEP 2) and the injection renderer (Codebase skeleton header):

  1. Is the skeleton listing for this file current? (`freshness`)
  2. How was the file processed last time we touched it? (`scan_status`)

The hook denies Read on files where:
  freshness='fresh' AND scan_status='scanned' AND symbol_count > 0.
All other states fall through to ALLOW because either the skeleton is stale,
the file failed to parse, or there's no public surface to defer to.

This module owns row upsert + status queries. Symbol-level row management
remains in `cognikernel.symbols.store`.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass

VALID_FRESHNESS = frozenset({"fresh", "stale"})
VALID_SCAN_STATUS = frozenset({"scanned", "parse_error", "ignored", "pending"})
VALID_ACTIONS = frozenset({"Write", "Edit", "scan", ""})


@dataclass(frozen=True)
class SymbolFile:
    project_id: str
    path: str
    freshness: str
    refreshed_at: int
    refreshed_in_session: str
    last_action: str
    content_sha256: str
    scan_status: str
    symbol_count: int
    last_error: str


@dataclass(frozen=True)
class CoverageStats:
    """Counts for the Codebase skeleton header (B-2)."""
    scanned: int          # scan_status='scanned'
    with_symbols: int     # scan_status='scanned' AND symbol_count > 0
    parse_errors: int     # scan_status='parse_error'
    ignored: int          # scan_status='ignored'
    pending: int          # scan_status='pending'


@dataclass(frozen=True)
class RefreshInfo:
    """Most-recent refresh metadata for the skeleton header."""
    path: str
    refreshed_in_session: str
    last_action: str
    refreshed_at: int


def upsert(
    conn: sqlite3.Connection,
    project_id: str,
    path: str,
    *,
    freshness: str = "fresh",
    refreshed_at: int | None = None,
    refreshed_in_session: str = "",
    last_action: str = "scan",
    content_sha256: str = "",
    scan_status: str = "scanned",
    symbol_count: int = 0,
    last_error: str = "",
) -> None:
    """Insert or update the file-level row.

    Callers MUST pass canonical relative paths (handled by cognikernel.utils.paths
    in Phase C2). Validates enums to catch typos before they reach SQLite.
    """
    if freshness not in VALID_FRESHNESS:
        raise ValueError(f"invalid freshness: {freshness!r}")
    if scan_status not in VALID_SCAN_STATUS:
        raise ValueError(f"invalid scan_status: {scan_status!r}")
    if last_action not in VALID_ACTIONS:
        raise ValueError(f"invalid last_action: {last_action!r}")
    if scan_status == "parse_error" and not last_error:
        raise ValueError("scan_status='parse_error' requires non-empty last_error")
    now = _now_ms() if refreshed_at is None else refreshed_at

    conn.execute(
        """
        INSERT INTO symbol_files
            (project_id, path, freshness, refreshed_at, refreshed_in_session,
             last_action, content_sha256, scan_status, symbol_count, last_error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (project_id, path) DO UPDATE SET
            freshness = excluded.freshness,
            refreshed_at = excluded.refreshed_at,
            refreshed_in_session = excluded.refreshed_in_session,
            last_action = excluded.last_action,
            content_sha256 = excluded.content_sha256,
            scan_status = excluded.scan_status,
            symbol_count = excluded.symbol_count,
            last_error = excluded.last_error
        """,
        (
            project_id, path, freshness, now, refreshed_in_session,
            last_action, content_sha256, scan_status, symbol_count, last_error,
        ),
    )
    conn.commit()


def get(
    conn: sqlite3.Connection,
    project_id: str,
    path: str,
) -> SymbolFile | None:
    """Return the row for (project_id, path), or None if no row exists."""
    row = conn.execute(
        """
        SELECT project_id, path, freshness, refreshed_at, refreshed_in_session,
               last_action, content_sha256, scan_status, symbol_count, last_error
        FROM symbol_files
        WHERE project_id=? AND path=?
        """,
        (project_id, path),
    ).fetchone()
    return _row_to_file(row) if row else None


def mark_stale(
    conn: sqlite3.Connection,
    project_id: str,
    path: str,
) -> bool:
    """Mark a row stale (e.g., external edit detected). Returns True if a row was updated."""
    cur = conn.execute(
        """
        UPDATE symbol_files SET freshness='stale'
        WHERE project_id=? AND path=? AND freshness='fresh'
        """,
        (project_id, path),
    )
    conn.commit()
    return cur.rowcount > 0


def coverage_stats(
    conn: sqlite3.Connection,
    project_id: str,
) -> CoverageStats:
    """Aggregate counts for the skeleton header (B-2)."""
    row = conn.execute(
        """
        SELECT
            SUM(CASE WHEN scan_status='scanned' THEN 1 ELSE 0 END)                          AS scanned,
            SUM(CASE WHEN scan_status='scanned' AND symbol_count > 0 THEN 1 ELSE 0 END)     AS with_symbols,
            SUM(CASE WHEN scan_status='parse_error' THEN 1 ELSE 0 END)                      AS parse_errors,
            SUM(CASE WHEN scan_status='ignored' THEN 1 ELSE 0 END)                          AS ignored,
            SUM(CASE WHEN scan_status='pending' THEN 1 ELSE 0 END)                          AS pending
        FROM symbol_files
        WHERE project_id=?
        """,
        (project_id,),
    ).fetchone()
    return CoverageStats(
        scanned=row["scanned"] or 0,
        with_symbols=row["with_symbols"] or 0,
        parse_errors=row["parse_errors"] or 0,
        ignored=row["ignored"] or 0,
        pending=row["pending"] or 0,
    )


def most_recent_refresh(
    conn: sqlite3.Connection,
    project_id: str,
) -> RefreshInfo | None:
    """Return metadata for the most-recently-refreshed file, or None if empty."""
    row = conn.execute(
        """
        SELECT path, refreshed_in_session, last_action, refreshed_at
        FROM symbol_files
        WHERE project_id=? AND refreshed_at > 0
        ORDER BY refreshed_at DESC
        LIMIT 1
        """,
        (project_id,),
    ).fetchone()
    if row is None:
        return None
    return RefreshInfo(
        path=row["path"],
        refreshed_in_session=row["refreshed_in_session"],
        last_action=row["last_action"],
        refreshed_at=row["refreshed_at"],
    )


def list_files(
    conn: sqlite3.Connection,
    project_id: str,
) -> list[SymbolFile]:
    """All rows for a project, ordered by path. Useful for tests + admin tooling."""
    rows = conn.execute(
        """
        SELECT project_id, path, freshness, refreshed_at, refreshed_in_session,
               last_action, content_sha256, scan_status, symbol_count, last_error
        FROM symbol_files
        WHERE project_id=?
        ORDER BY path
        """,
        (project_id,),
    ).fetchall()
    return [_row_to_file(r) for r in rows]


# ── internals ────────────────────────────────────────────────────────────────


def _row_to_file(row: sqlite3.Row) -> SymbolFile:
    return SymbolFile(
        project_id=row["project_id"],
        path=row["path"],
        freshness=row["freshness"],
        refreshed_at=row["refreshed_at"],
        refreshed_in_session=row["refreshed_in_session"],
        last_action=row["last_action"],
        content_sha256=row["content_sha256"],
        scan_status=row["scan_status"],
        symbol_count=row["symbol_count"],
        last_error=row["last_error"],
    )


def _now_ms() -> int:
    return int(time.time() * 1000)
