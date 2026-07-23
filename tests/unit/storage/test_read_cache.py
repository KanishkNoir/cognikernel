from __future__ import annotations

import sqlite3

import pytest

from cognikernel.storage.read_cache import (
    DEFAULT_TTL_MS,
    ReadCacheEntry,
    cleanup_old,
    clear_session,
    get_read,
    record_read,
    was_read_in_session,
)


def test_record_read_inserts_new_row(conn: sqlite3.Connection) -> None:
    record_read(conn, "p1", "s1", "app/main.py", now_ms=1000)

    entry = get_read(conn, "p1", "s1", "app/main.py")
    assert entry is not None
    assert entry.first_read_at == 1000
    assert entry.last_read_at == 1000
    assert entry.read_count == 1
    assert entry.last_read_outcome == "ok"


def test_record_read_second_call_upserts_and_increments(conn: sqlite3.Connection) -> None:
    record_read(conn, "p1", "s1", "app/main.py", now_ms=1000)
    record_read(conn, "p1", "s1", "app/main.py", now_ms=2000)

    entry = get_read(conn, "p1", "s1", "app/main.py")
    assert entry is not None
    assert entry.first_read_at == 1000     # pinned to first call
    assert entry.last_read_at == 2000      # advances
    assert entry.read_count == 2
    assert entry.last_read_outcome == "ok"


def test_record_read_outcome_can_change_to_body_needed_retry(
    conn: sqlite3.Connection,
) -> None:
    record_read(conn, "p1", "s1", "app/main.py", now_ms=1000)
    record_read(
        conn, "p1", "s1", "app/main.py",
        now_ms=2000, outcome="body_needed_retry",
    )

    entry = get_read(conn, "p1", "s1", "app/main.py")
    assert entry is not None
    assert entry.last_read_outcome == "body_needed_retry"


def test_record_read_rejects_invalid_outcome(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="invalid outcome"):
        record_read(conn, "p1", "s1", "app/main.py", outcome="oops")


def test_get_read_returns_none_when_absent(conn: sqlite3.Connection) -> None:
    assert get_read(conn, "p1", "s1", "missing.py") is None


def test_was_read_in_session_for_absent_returns_false_none(
    conn: sqlite3.Connection,
) -> None:
    flag, outcome = was_read_in_session(conn, "p1", "s1", "missing.py")
    assert flag is False
    assert outcome is None


def test_was_read_in_session_for_present_returns_true_outcome(
    conn: sqlite3.Connection,
) -> None:
    record_read(conn, "p1", "s1", "app/main.py", outcome="body_needed_retry")

    flag, outcome = was_read_in_session(conn, "p1", "s1", "app/main.py")
    assert flag is True
    assert outcome == "body_needed_retry"


def test_separate_sessions_do_not_collide(conn: sqlite3.Connection) -> None:
    record_read(conn, "p1", "s1", "app/main.py")
    record_read(conn, "p1", "s2", "app/main.py")

    assert get_read(conn, "p1", "s1", "app/main.py") is not None
    assert get_read(conn, "p1", "s2", "app/main.py") is not None
    assert conn.execute(
        "SELECT COUNT(*) FROM read_session_cache"
    ).fetchone()[0] == 2


def test_separate_projects_do_not_collide(conn: sqlite3.Connection) -> None:
    record_read(conn, "p1", "s1", "app/main.py")
    record_read(conn, "p2", "s1", "app/main.py")

    assert conn.execute(
        "SELECT COUNT(*) FROM read_session_cache"
    ).fetchone()[0] == 2


def test_cleanup_old_removes_rows_older_than_ttl(conn: sqlite3.Connection) -> None:
    record_read(conn, "p1", "s1", "old.py", now_ms=1_000)
    record_read(conn, "p1", "s1", "recent.py", now_ms=10_000)

    removed = cleanup_old(conn, ttl_ms=5_000, now_ms=11_000)

    assert removed == 1
    assert get_read(conn, "p1", "s1", "old.py") is None
    assert get_read(conn, "p1", "s1", "recent.py") is not None


def test_cleanup_old_with_default_ttl_is_noop_for_fresh_rows(
    conn: sqlite3.Connection,
) -> None:
    record_read(conn, "p1", "s1", "fresh.py")  # real-time now

    removed = cleanup_old(conn)  # default ttl: 24h
    assert removed == 0


def test_clear_session_removes_only_target_session(conn: sqlite3.Connection) -> None:
    record_read(conn, "p1", "s1", "a.py")
    record_read(conn, "p1", "s1", "b.py")
    record_read(conn, "p1", "s2", "c.py")

    removed = clear_session(conn, "p1", "s1")

    assert removed == 2
    assert get_read(conn, "p1", "s1", "a.py") is None
    assert get_read(conn, "p1", "s1", "b.py") is None
    assert get_read(conn, "p1", "s2", "c.py") is not None


def test_clear_session_for_nonexistent_session_returns_zero(
    conn: sqlite3.Connection,
) -> None:
    assert clear_session(conn, "p1", "no-such-session") == 0
