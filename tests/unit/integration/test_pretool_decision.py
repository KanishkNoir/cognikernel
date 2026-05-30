"""Tests for the C1 PreToolUse:Read decision tree.

Covers all branches of decide_pretool_read():
  STEP 1 — re-read denial (universal across both policies)
  STEP 2 — skeleton gating (strict only)
    Case A — fresh + scanned + symbols > 0: first denial / retry allowance
    Case B — stale freshness
    Case C — parse_error / ignored scan_status
    Case D — symbol_count = 0
    Case E — no symbol_files row

Plus advisory-policy fallback and edge cases (path outside project, retry
window edges, denial timer cleanup).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from memlora.integration.lookup import (
    Decision,
    decide_pretool_read,
    resolve_post_read_outcome,
)
from memlora.storage import denied_reads as dr
from memlora.storage import read_cache as rc
from memlora.storage import symbol_files as sf


# Helpers ----------------------------------------------------------------------


def _project(tmp_path: Path) -> str:
    """Create an empty project directory and return its absolute path string."""
    proj = tmp_path / "myproj"
    proj.mkdir()
    return str(proj)


def _touch(project_path: str, rel: str) -> str:
    """Create a real file under the project so resolve()/relative_to() works."""
    full = Path(project_path) / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text("placeholder", encoding="utf-8")
    return str(full)


def _decide(conn, *, file_path: str, project_path: str, policy: str = "strict",
            session_id: str = "s1", project_id: str = "p1",
            now_ms: int = 10_000) -> Decision:
    return decide_pretool_read(
        conn, project_id, session_id, file_path, project_path,
        policy=policy, now_ms=now_ms,
    )


# STEP 1 — re-read denial ------------------------------------------------------


def test_reread_after_ok_is_denied_strict(conn: sqlite3.Connection, tmp_path: Path) -> None:
    project = _project(tmp_path)
    abs_path = _touch(project, "app/main.py")
    rc.record_read(conn, "p1", "s1", "app/main.py", outcome="ok", now_ms=5_000)

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.is_deny
    assert "already read" in d.message.lower()
    assert d.reason == "reread_same_session"


def test_reread_after_body_retry_is_denied(conn: sqlite3.Connection, tmp_path: Path) -> None:
    project = _project(tmp_path)
    abs_path = _touch(project, "app/main.py")
    rc.record_read(conn, "p1", "s1", "app/main.py", outcome="body_needed_retry", now_ms=5_000)

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.is_deny
    assert "body" in d.message.lower()
    assert d.reason == "rereread_after_body_retry"


def test_reread_denial_fires_under_advisory_policy_too(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    """STEP 1 is universal — even advisory mode catches re-reads."""
    project = _project(tmp_path)
    abs_path = _touch(project, "app/main.py")
    rc.record_read(conn, "p1", "s1", "app/main.py", outcome="ok", now_ms=5_000)

    d = _decide(conn, file_path=abs_path, project_path=project, policy="advisory")

    assert d.is_deny
    assert d.reason == "reread_same_session"


def test_reread_scoped_to_session(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """A read in session s1 doesn't block a read in session s2."""
    project = _project(tmp_path)
    abs_path = _touch(project, "app/main.py")
    rc.record_read(conn, "p1", "s1", "app/main.py", outcome="ok")
    # No symbol_files row → STEP 2 Case E → ALLOW
    d = _decide(conn, file_path=abs_path, project_path=project, session_id="s2")

    assert d.action == "allow"
    assert d.reason == "not_in_symbol_files"


# STEP 2 Case E — no symbol_files row -----------------------------------------


def test_no_symbol_files_row_allows_under_strict(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    abs_path = _touch(project, "new.py")

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.action == "allow"
    assert d.reason == "not_in_symbol_files"


# STEP 2 Case D — symbol_count = 0 --------------------------------------------


def test_zero_symbols_allows_strict(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """A file like __init__.py with no public symbols must still be readable."""
    project = _project(tmp_path)
    abs_path = _touch(project, "pkg/__init__.py")
    sf.upsert(
        conn, "p1", "pkg/__init__.py",
        scan_status="scanned", symbol_count=0, freshness="fresh",
    )

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.action == "allow"
    assert d.reason == "no_public_symbols"


def test_zero_symbols_still_denied_on_reread(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    """Re-read protection (STEP 1) fires before Case D — the universal invariant."""
    project = _project(tmp_path)
    abs_path = _touch(project, "pkg/__init__.py")
    sf.upsert(
        conn, "p1", "pkg/__init__.py",
        scan_status="scanned", symbol_count=0, freshness="fresh",
    )
    rc.record_read(conn, "p1", "s1", "pkg/__init__.py", outcome="ok")

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.is_deny
    assert d.reason == "reread_same_session"


# STEP 2 Case C — parse_error / ignored ---------------------------------------


def test_parse_error_allows_strict(conn: sqlite3.Connection, tmp_path: Path) -> None:
    project = _project(tmp_path)
    abs_path = _touch(project, "broken.py")
    sf.upsert(
        conn, "p1", "broken.py",
        scan_status="parse_error", last_error="SyntaxError: invalid syntax",
    )

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.action == "allow"
    assert d.reason == "scan_status_parse_error"


def test_ignored_allows_strict(conn: sqlite3.Connection, tmp_path: Path) -> None:
    project = _project(tmp_path)
    abs_path = _touch(project, "bin.lock")
    sf.upsert(conn, "p1", "bin.lock", scan_status="ignored")

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.action == "allow"
    assert d.reason == "scan_status_ignored"


def test_pending_allows_strict(conn: sqlite3.Connection, tmp_path: Path) -> None:
    """Row exists but symbols not yet extracted — be permissive."""
    project = _project(tmp_path)
    abs_path = _touch(project, "new.py")
    sf.upsert(conn, "p1", "new.py", scan_status="pending")

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.action == "allow"
    assert d.reason == "scan_status_pending"


# STEP 2 Case B — stale freshness ---------------------------------------------


def test_stale_freshness_allows_strict(conn: sqlite3.Connection, tmp_path: Path) -> None:
    project = _project(tmp_path)
    abs_path = _touch(project, "edited_externally.py")
    sf.upsert(
        conn, "p1", "edited_externally.py",
        freshness="stale", scan_status="scanned", symbol_count=5,
    )

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.action == "allow"
    assert d.reason == "symbol_files_stale"


# STEP 2 Case A — first denial -------------------------------------------------


def test_fresh_scanned_with_symbols_denies_first_attempt(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    abs_path = _touch(project, "app/main.py")
    sf.upsert(
        conn, "p1", "app/main.py",
        freshness="fresh", scan_status="scanned", symbol_count=5,
    )

    d = _decide(conn, file_path=abs_path, project_path=project, now_ms=10_000)

    assert d.is_deny
    assert d.reason == "skeleton_fresh_first_denial"
    assert "Codebase skeleton" in d.message
    # Denial recorded for the retry window
    assert dr.was_denied_within(conn, "p1", "s1", "app/main.py", now_ms=10_000)


# STEP 2 Case A — freshness verification (external-edit detection) ------------


def test_externally_edited_fresh_file_allows_and_marks_stale(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    """#2: a 'fresh' row whose file changed since the scan (mtime > refreshed_at)
    must not be trusted — allow the read and record the row stale, so the
    skeleton never vouches for signatures it can no longer back."""
    project = _project(tmp_path)
    abs_path = _touch(project, "app/main.py")  # real file, mtime ≈ now
    # Scanned long ago relative to the file's mtime → external edit since.
    sf.upsert(
        conn, "p1", "app/main.py",
        freshness="fresh", scan_status="scanned", symbol_count=5,
        refreshed_at=1_000,  # epoch ~1970 ms; file mtime is ~now
    )

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.action == "allow"
    assert d.reason == "symbol_files_mtime_stale"
    # The drift was persisted so future lookups short-circuit via Case B.
    assert sf.get(conn, "p1", "app/main.py").freshness == "stale"


def test_unchanged_fresh_file_still_denies(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    """A file scanned at-or-after its mtime is genuinely fresh → still denied
    (the verification must not produce false staleness on freshly-scanned files)."""
    project = _project(tmp_path)
    abs_path = _touch(project, "app/main.py")
    # Default refreshed_at = now (set after the touch) → mtime ≤ refreshed_at.
    sf.upsert(
        conn, "p1", "app/main.py",
        freshness="fresh", scan_status="scanned", symbol_count=5,
    )

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.is_deny
    assert d.reason == "skeleton_fresh_first_denial"


def test_zero_refreshed_at_falls_back_to_flag(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    """A sentinel/unset refreshed_at (<=0) must not force staleness — fall back
    to the flag (deny) rather than disabling skeleton-trust for the row."""
    project = _project(tmp_path)
    abs_path = _touch(project, "app/main.py")
    sf.upsert(
        conn, "p1", "app/main.py",
        freshness="fresh", scan_status="scanned", symbol_count=5,
        refreshed_at=0,
    )

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.is_deny
    assert d.reason == "skeleton_fresh_first_denial"


def test_missing_file_falls_back_to_flag(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    """If the file can't be stat'd, fall back to the stored flag (deny), not a
    transient flip to allow."""
    project = _project(tmp_path)
    # No file on disk; row claims fresh + scanned.
    sf.upsert(
        conn, "p1", "ghost.py",
        freshness="fresh", scan_status="scanned", symbol_count=5,
        refreshed_at=1_000,
    )
    abs_path = str(Path(project) / "ghost.py")

    d = _decide(conn, file_path=abs_path, project_path=project)

    assert d.is_deny
    assert d.reason == "skeleton_fresh_first_denial"


# STEP 2 Case A — retry within window (escape hatch) --------------------------


def test_retry_within_60s_allows_as_body_needed(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    abs_path = _touch(project, "app/main.py")
    sf.upsert(
        conn, "p1", "app/main.py",
        freshness="fresh", scan_status="scanned", symbol_count=5,
    )

    # First attempt — denied, denial recorded at t=10s.
    d1 = _decide(conn, file_path=abs_path, project_path=project, now_ms=10_000)
    assert d1.is_deny

    # Second attempt at t=10s + 30s — within 60s window.
    d2 = _decide(conn, file_path=abs_path, project_path=project, now_ms=40_000)

    assert d2.action == "allow"
    assert d2.reason == "body_needed_retry_within_window"
    assert d2.outcome_hint == "body_needed_retry"


def test_retry_after_60s_denies_again(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    abs_path = _touch(project, "app/main.py")
    sf.upsert(
        conn, "p1", "app/main.py",
        freshness="fresh", scan_status="scanned", symbol_count=5,
    )

    d1 = _decide(conn, file_path=abs_path, project_path=project, now_ms=10_000)
    assert d1.is_deny

    # Second attempt beyond 60s — fresh denial cycle.
    d2 = _decide(conn, file_path=abs_path, project_path=project, now_ms=80_000)

    assert d2.is_deny
    assert d2.reason == "skeleton_fresh_first_denial"


def test_retry_allowance_clears_denied_reads_row(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    """After a successful retry-grant, the denial row is cleared so a future
    edit cycle starts cleanly."""
    project = _project(tmp_path)
    abs_path = _touch(project, "app/main.py")
    sf.upsert(
        conn, "p1", "app/main.py",
        freshness="fresh", scan_status="scanned", symbol_count=5,
    )

    _decide(conn, file_path=abs_path, project_path=project, now_ms=10_000)
    _decide(conn, file_path=abs_path, project_path=project, now_ms=40_000)

    # Cleared by the retry path.
    assert dr.get(conn, "p1", "s1", "app/main.py") is None


# Advisory policy --------------------------------------------------------------


def test_advisory_policy_allows_even_skeleton_fresh(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    abs_path = _touch(project, "app/main.py")
    sf.upsert(
        conn, "p1", "app/main.py",
        freshness="fresh", scan_status="scanned", symbol_count=5,
    )

    d = _decide(conn, file_path=abs_path, project_path=project, policy="advisory")

    assert d.action == "allow"
    assert d.reason == "advisory_policy"


# Edge cases -------------------------------------------------------------------


def test_path_outside_project_allows(conn: sqlite3.Connection, tmp_path: Path) -> None:
    project = _project(tmp_path)
    # Path far away from project
    outside = tmp_path / "elsewhere.py"
    outside.write_text("x", encoding="utf-8")

    d = _decide(conn, file_path=str(outside), project_path=project)

    assert d.action == "allow"
    assert d.reason == "path_outside_project"


def test_relative_path_input_is_canonicalized(
    conn: sqlite3.Connection, tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    _touch(project, "app/main.py")
    sf.upsert(
        conn, "p1", "app/main.py",
        freshness="fresh", scan_status="scanned", symbol_count=5,
    )

    # Pass a relative path with backslashes (Windows-style)
    d = _decide(conn, file_path=r"app\main.py", project_path=project)

    assert d.is_deny  # canonicalization succeeded → matched symbol_files row


# resolve_post_read_outcome ---------------------------------------------------


def test_resolve_outcome_returns_ok_when_no_denial(
    conn: sqlite3.Connection,
) -> None:
    outcome = resolve_post_read_outcome(conn, "p1", "s1", "app/main.py", now_ms=10_000)
    assert outcome == "ok"


def test_resolve_outcome_returns_body_needed_retry_when_recently_denied(
    conn: sqlite3.Connection,
) -> None:
    dr.record(conn, "p1", "s1", "app/main.py", now_ms=10_000)
    outcome = resolve_post_read_outcome(
        conn, "p1", "s1", "app/main.py",
        retry_window_ms=60_000, now_ms=30_000,
    )
    assert outcome == "body_needed_retry"


def test_resolve_outcome_ignores_old_denial_outside_window(
    conn: sqlite3.Connection,
) -> None:
    dr.record(conn, "p1", "s1", "app/main.py", now_ms=10_000)
    outcome = resolve_post_read_outcome(
        conn, "p1", "s1", "app/main.py",
        retry_window_ms=60_000, now_ms=100_000,
    )
    assert outcome == "ok"
