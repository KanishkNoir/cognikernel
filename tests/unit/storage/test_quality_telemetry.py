"""Per-rule gate counters.

Uses the shared `conn` fixture from tests/conftest.py, which yields an open,
migrated connection.
"""
import sqlite3

from cognikernel.storage.quality_telemetry import get_rule_counts, record_verdict


class TestRecordVerdict:
    def test_records_a_rule_hit(self, conn: sqlite3.Connection) -> None:
        record_verdict(conn, "p", "s", "D7")
        counts = get_rule_counts(conn, "p")
        assert counts[0]["rule_id"] == "D7"
        assert counts[0]["total"] == 1

    def test_accumulates_repeat_hits(self, conn: sqlite3.Connection) -> None:
        for _ in range(3):
            record_verdict(conn, "p", "s", "D7")
        assert get_rule_counts(conn, "p")[0]["total"] == 3

    def test_orders_by_frequency(self, conn: sqlite3.Connection) -> None:
        record_verdict(conn, "p", "s", "D2")
        for _ in range(5):
            record_verdict(conn, "p", "s", "D7")
        assert get_rule_counts(conn, "p")[0]["rule_id"] == "D7"

    def test_scopes_to_project(self, conn: sqlite3.Connection) -> None:
        record_verdict(conn, "p1", "s", "D7")
        assert get_rule_counts(conn, "p2") == []

    def test_empty_rule_id_is_ignored(self, conn: sqlite3.Connection) -> None:
        record_verdict(conn, "p", "s", "")
        assert get_rule_counts(conn, "p") == []

    def test_separate_sessions_sum_together(self, conn: sqlite3.Connection) -> None:
        record_verdict(conn, "p", "s1", "D7")
        record_verdict(conn, "p", "s2", "D7")
        assert get_rule_counts(conn, "p")[0]["total"] == 2


class TestMigrationIdempotency:
    def test_running_migrations_twice_is_safe(self, conn: sqlite3.Connection) -> None:
        from cognikernel.storage.migrations import run_migrations

        # The `conn` fixture already ran migrations once; a second run must be
        # a no-op rather than an error.
        run_migrations(conn)
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        assert int(row[0]) >= 19
