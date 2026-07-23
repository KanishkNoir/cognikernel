from __future__ import annotations

import sqlite3

import pytest

from cognikernel.storage.denied_reads import (
    DEFAULT_CLEANUP_TTL_MS,
    DEFAULT_RETRY_WINDOW_MS,
    cleanup_old,
    clear,
    get,
    record,
    was_denied_within,
)


def test_record_inserts_new_row(conn: sqlite3.Connection) -> None:
    record(conn, "p1", "s1", "app/main.py", now_ms=1000)

    entry = get(conn, "p1", "s1", "app/main.py")
    assert entry is not None
    assert entry.denied_at == 1000
    assert entry.reason == "skeleton_fresh"


def test_record_redenial_bumps_denied_at(conn: sqlite3.Connection) -> None:
    record(conn, "p1", "s1", "app/main.py", now_ms=1000)
    record(conn, "p1", "s1", "app/main.py", now_ms=5000)

    entry = get(conn, "p1", "s1", "app/main.py")
    assert entry is not None
    assert entry.denied_at == 5000


def test_record_rejects_invalid_reason(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="invalid reason"):
        record(conn, "p1", "s1", "a.py", reason="oops")


def test_was_denied_within_true_inside_window(conn: sqlite3.Connection) -> None:
    record(conn, "p1", "s1", "app/main.py", now_ms=1000)

    assert was_denied_within(
        conn, "p1", "s1", "app/main.py",
        window_ms=60_000, now_ms=30_000,
    )


def test_was_denied_within_true_at_window_edge(conn: sqlite3.Connection) -> None:
    """Edge inclusive: exactly window_ms after denial still counts as 'within'."""
    record(conn, "p1", "s1", "app/main.py", now_ms=1000)

    assert was_denied_within(
        conn, "p1", "s1", "app/main.py",
        window_ms=1000, now_ms=2000,
    )


def test_was_denied_within_false_past_window(conn: sqlite3.Connection) -> None:
    record(conn, "p1", "s1", "app/main.py", now_ms=1000)

    assert not was_denied_within(
        conn, "p1", "s1", "app/main.py",
        window_ms=1000, now_ms=2001,
    )


def test_was_denied_within_false_for_missing_row(conn: sqlite3.Connection) -> None:
    assert not was_denied_within(conn, "p1", "s1", "absent.py")


def test_get_returns_none_for_absent(conn: sqlite3.Connection) -> None:
    assert get(conn, "p1", "s1", "absent.py") is None


def test_clear_removes_specific_row(conn: sqlite3.Connection) -> None:
    record(conn, "p1", "s1", "a.py")
    record(conn, "p1", "s1", "b.py")

    removed = clear(conn, "p1", "s1", "a.py")

    assert removed is True
    assert get(conn, "p1", "s1", "a.py") is None
    assert get(conn, "p1", "s1", "b.py") is not None


def test_clear_returns_false_for_missing_row(conn: sqlite3.Connection) -> None:
    assert clear(conn, "p1", "s1", "absent.py") is False


def test_cleanup_old_removes_rows_older_than_ttl(conn: sqlite3.Connection) -> None:
    record(conn, "p1", "s1", "old.py", now_ms=1_000)
    record(conn, "p1", "s1", "recent.py", now_ms=10_000)

    removed = cleanup_old(conn, ttl_ms=5_000, now_ms=11_000)

    assert removed == 1
    assert get(conn, "p1", "s1", "old.py") is None
    assert get(conn, "p1", "s1", "recent.py") is not None


def test_cleanup_old_is_idempotent_on_empty(conn: sqlite3.Connection) -> None:
    assert cleanup_old(conn) == 0


def test_separate_sessions_isolated(conn: sqlite3.Connection) -> None:
    record(conn, "p1", "s1", "a.py")
    record(conn, "p1", "s2", "a.py")

    assert conn.execute("SELECT COUNT(*) FROM denied_reads").fetchone()[0] == 2


def test_defaults_have_reasonable_values() -> None:
    """Sanity check that defaults reflect the v2 plan."""
    assert DEFAULT_RETRY_WINDOW_MS == 60 * 1000
    assert DEFAULT_CLEANUP_TTL_MS == 5 * 60 * 1000
    assert DEFAULT_CLEANUP_TTL_MS > DEFAULT_RETRY_WINDOW_MS  # cleanup must not evict active denies
