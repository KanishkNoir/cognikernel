from __future__ import annotations

import sqlite3

import pytest

from cognikernel.storage.symbol_files import (
    CoverageStats,
    SymbolFile,
    coverage_stats,
    get,
    list_files,
    mark_stale,
    most_recent_refresh,
    upsert,
)


def test_upsert_inserts_new_row(conn: sqlite3.Connection) -> None:
    upsert(
        conn, "p1", "app/main.py",
        refreshed_at=1000,
        refreshed_in_session="sess-1",
        last_action="Write",
        content_sha256="abc123",
        scan_status="scanned",
        symbol_count=3,
    )

    sf = get(conn, "p1", "app/main.py")
    assert sf is not None
    assert sf.freshness == "fresh"
    assert sf.refreshed_at == 1000
    assert sf.refreshed_in_session == "sess-1"
    assert sf.last_action == "Write"
    assert sf.content_sha256 == "abc123"
    assert sf.scan_status == "scanned"
    assert sf.symbol_count == 3


def test_upsert_updates_existing_row_on_conflict(conn: sqlite3.Connection) -> None:
    upsert(conn, "p1", "app/main.py", refreshed_at=1000, symbol_count=3)
    upsert(conn, "p1", "app/main.py", refreshed_at=2000, symbol_count=5,
           last_action="Edit", content_sha256="new-hash")

    sf = get(conn, "p1", "app/main.py")
    assert sf is not None
    assert sf.refreshed_at == 2000
    assert sf.symbol_count == 5
    assert sf.last_action == "Edit"
    assert sf.content_sha256 == "new-hash"


def test_upsert_validates_freshness(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="invalid freshness"):
        upsert(conn, "p1", "a.py", freshness="bogus")


def test_upsert_validates_scan_status(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="invalid scan_status"):
        upsert(conn, "p1", "a.py", scan_status="bogus")


def test_upsert_validates_last_action(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="invalid last_action"):
        upsert(conn, "p1", "a.py", last_action="Bogus")


def test_upsert_parse_error_requires_last_error(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="last_error"):
        upsert(conn, "p1", "a.py", scan_status="parse_error")


def test_upsert_parse_error_with_last_error_is_accepted(
    conn: sqlite3.Connection,
) -> None:
    upsert(conn, "p1", "a.py", scan_status="parse_error", last_error="SyntaxError")

    sf = get(conn, "p1", "a.py")
    assert sf is not None
    assert sf.scan_status == "parse_error"
    assert sf.last_error == "SyntaxError"


def test_get_returns_none_for_absent(conn: sqlite3.Connection) -> None:
    assert get(conn, "p1", "missing.py") is None


def test_mark_stale_flips_fresh_to_stale(conn: sqlite3.Connection) -> None:
    upsert(conn, "p1", "a.py")
    assert get(conn, "p1", "a.py").freshness == "fresh"

    flipped = mark_stale(conn, "p1", "a.py")

    assert flipped is True
    assert get(conn, "p1", "a.py").freshness == "stale"


def test_mark_stale_is_idempotent_for_already_stale(conn: sqlite3.Connection) -> None:
    upsert(conn, "p1", "a.py", freshness="stale")

    flipped = mark_stale(conn, "p1", "a.py")

    assert flipped is False  # no fresh→stale transition occurred
    assert get(conn, "p1", "a.py").freshness == "stale"


def test_mark_stale_no_op_for_missing_row(conn: sqlite3.Connection) -> None:
    assert mark_stale(conn, "p1", "missing.py") is False


def test_coverage_stats_empty(conn: sqlite3.Connection) -> None:
    stats = coverage_stats(conn, "p1")
    assert stats == CoverageStats(0, 0, 0, 0, 0)


def test_coverage_stats_counts_each_status(conn: sqlite3.Connection) -> None:
    upsert(conn, "p1", "a.py", scan_status="scanned", symbol_count=5)
    upsert(conn, "p1", "b.py", scan_status="scanned", symbol_count=0)
    upsert(conn, "p1", "c.py", scan_status="scanned", symbol_count=2)
    upsert(conn, "p1", "d.py", scan_status="parse_error", last_error="x")
    upsert(conn, "p1", "e.py", scan_status="ignored")
    upsert(conn, "p1", "f.py", scan_status="pending")

    stats = coverage_stats(conn, "p1")
    assert stats.scanned == 3
    assert stats.with_symbols == 2  # a.py and c.py (b.py has count=0)
    assert stats.parse_errors == 1
    assert stats.ignored == 1
    assert stats.pending == 1


def test_coverage_stats_scoped_to_project(conn: sqlite3.Connection) -> None:
    upsert(conn, "p1", "a.py", scan_status="scanned", symbol_count=1)
    upsert(conn, "p2", "a.py", scan_status="scanned", symbol_count=1)

    assert coverage_stats(conn, "p1").scanned == 1
    assert coverage_stats(conn, "p2").scanned == 1


def test_most_recent_refresh_returns_none_when_no_refreshed_rows(
    conn: sqlite3.Connection,
) -> None:
    # refreshed_at=0 (default) means never refreshed
    upsert(conn, "p1", "a.py", refreshed_at=0, scan_status="pending")
    assert most_recent_refresh(conn, "p1") is None


def test_most_recent_refresh_returns_latest(conn: sqlite3.Connection) -> None:
    upsert(conn, "p1", "a.py", refreshed_at=1000, refreshed_in_session="s1",
           last_action="Write")
    upsert(conn, "p1", "b.py", refreshed_at=3000, refreshed_in_session="s2",
           last_action="Edit")
    upsert(conn, "p1", "c.py", refreshed_at=2000, refreshed_in_session="s2",
           last_action="Write")

    info = most_recent_refresh(conn, "p1")
    assert info is not None
    assert info.path == "b.py"
    assert info.refreshed_at == 3000
    assert info.refreshed_in_session == "s2"
    assert info.last_action == "Edit"


def test_most_recent_refresh_scoped_to_project(conn: sqlite3.Connection) -> None:
    upsert(conn, "p1", "a.py", refreshed_at=1000)
    upsert(conn, "p2", "a.py", refreshed_at=2000)

    info = most_recent_refresh(conn, "p1")
    assert info is not None
    assert info.refreshed_at == 1000


def test_list_files_returns_sorted_by_path(conn: sqlite3.Connection) -> None:
    upsert(conn, "p1", "z.py")
    upsert(conn, "p1", "a.py")
    upsert(conn, "p1", "m.py")

    files = list_files(conn, "p1")
    assert [f.path for f in files] == ["a.py", "m.py", "z.py"]


def test_list_files_empty_returns_empty_list(conn: sqlite3.Connection) -> None:
    assert list_files(conn, "p1") == []
