from __future__ import annotations

import sqlite3

import pytest

from cognikernel.storage.write_cache import (
    WriteCacheEntry,
    clear_session,
    get_write,
    record_write,
    was_written_in_session,
)


def test_record_write_inserts_new_row(conn: sqlite3.Connection) -> None:
    record_write(conn, "p1", "s1", "app/main.py", now_ms=1000)

    entry = get_write(conn, "p1", "s1", "app/main.py")
    assert entry is not None
    assert entry.first_write_at == 1000
    assert entry.last_write_at == 1000
    assert entry.write_count == 1
    assert entry.last_write_action == "Write"


def test_record_write_second_call_upserts_and_increments(conn: sqlite3.Connection) -> None:
    record_write(conn, "p1", "s1", "app/main.py", now_ms=1000)
    record_write(conn, "p1", "s1", "app/main.py", now_ms=2000, action="Edit")

    entry = get_write(conn, "p1", "s1", "app/main.py")
    assert entry is not None
    assert entry.first_write_at == 1000     # pinned to first call
    assert entry.last_write_at == 2000      # advances
    assert entry.write_count == 2
    assert entry.last_write_action == "Edit"


def test_record_write_accepts_multiedit_action(conn: sqlite3.Connection) -> None:
    record_write(conn, "p1", "s1", "app/main.py", action="MultiEdit")

    entry = get_write(conn, "p1", "s1", "app/main.py")
    assert entry is not None
    assert entry.last_write_action == "MultiEdit"


def test_record_write_rejects_invalid_action(conn: sqlite3.Connection) -> None:
    with pytest.raises(ValueError, match="invalid action"):
        record_write(conn, "p1", "s1", "app/main.py", action="Bash")


def test_get_write_returns_none_when_absent(conn: sqlite3.Connection) -> None:
    assert get_write(conn, "p1", "s1", "missing.py") is None


def test_was_written_in_session_for_absent_returns_false_none(
    conn: sqlite3.Connection,
) -> None:
    flag, action = was_written_in_session(conn, "p1", "s1", "missing.py")
    assert flag is False
    assert action is None


def test_was_written_in_session_for_present_returns_true_action(
    conn: sqlite3.Connection,
) -> None:
    record_write(conn, "p1", "s1", "app/main.py", action="MultiEdit")

    flag, action = was_written_in_session(conn, "p1", "s1", "app/main.py")
    assert flag is True
    assert action == "MultiEdit"


def test_separate_sessions_do_not_collide(conn: sqlite3.Connection) -> None:
    record_write(conn, "p1", "s1", "app/main.py")
    record_write(conn, "p1", "s2", "app/main.py")

    assert get_write(conn, "p1", "s1", "app/main.py") is not None
    assert get_write(conn, "p1", "s2", "app/main.py") is not None
    assert conn.execute(
        "SELECT COUNT(*) FROM write_session_cache"
    ).fetchone()[0] == 2


def test_separate_projects_do_not_collide(conn: sqlite3.Connection) -> None:
    record_write(conn, "p1", "s1", "app/main.py")
    record_write(conn, "p2", "s1", "app/main.py")

    assert conn.execute(
        "SELECT COUNT(*) FROM write_session_cache"
    ).fetchone()[0] == 2


def test_clear_session_removes_only_target_session(conn: sqlite3.Connection) -> None:
    record_write(conn, "p1", "s1", "a.py")
    record_write(conn, "p1", "s1", "b.py")
    record_write(conn, "p1", "s2", "c.py")

    removed = clear_session(conn, "p1", "s1")

    assert removed == 2
    assert get_write(conn, "p1", "s1", "a.py") is None
    assert get_write(conn, "p1", "s1", "b.py") is None
    assert get_write(conn, "p1", "s2", "c.py") is not None


def test_clear_session_for_nonexistent_session_returns_zero(
    conn: sqlite3.Connection,
) -> None:
    assert clear_session(conn, "p1", "no-such-session") == 0
