"""Integration tests for SQLite WAL concurrency guarantees.

Verifies that WAL mode delivers on its core promise: one writer and
multiple simultaneous readers do not block or corrupt each other.
"""
import threading
import time
from pathlib import Path

from memlora.storage.connection import get_connection
from memlora.storage.events import Event, insert_event, get_events_for_projection
from memlora.storage.migrations import run_migrations


def _make_db(tmp_path: Path, name: str = "concurrent.db") -> Path:
    db_path = tmp_path / name
    with get_connection(db_path) as conn:
        run_migrations(conn)
    return db_path


def _make_event(index: int, session_id: str = "sess1") -> Event:
    return Event(
        project_id="proj1",
        session_id=session_id,
        event_type="DECISION",
        payload={"description": f"Decision {index}"},
        content_hash=f"hash_{index}_{session_id}",
    )


class TestWALConcurrency:
    def test_multiple_readers_do_not_block_each_other(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        errors: list[str] = []
        results: list[str] = []

        def reader() -> None:
            try:
                with get_connection(db_path) as conn:
                    time.sleep(0.02)
                    conn.execute("SELECT COUNT(*) FROM events").fetchone()
                    results.append("ok")
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=reader) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert errors == [], f"Reader errors: {errors}"
        assert len(results) == 6

    def test_writer_and_reader_do_not_corrupt_each_other(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path, "wr.db")
        write_errors: list[str] = []
        read_results: list[int] = []

        def writer() -> None:
            for i in range(20):
                try:
                    with get_connection(db_path) as conn:
                        insert_event(conn, _make_event(i, session_id=f"s{i}"))
                except Exception as exc:
                    write_errors.append(str(exc))

        def reader() -> None:
            with get_connection(db_path) as conn:
                time.sleep(0.01)
                events = get_events_for_projection(conn, "proj1")
                read_results.append(len(events))

        writer_thread = threading.Thread(target=writer)
        reader_thread = threading.Thread(target=reader)

        writer_thread.start()
        reader_thread.start()
        writer_thread.join(timeout=10)
        reader_thread.join(timeout=10)

        assert write_errors == [], f"Write errors: {write_errors}"
        assert len(read_results) == 1  # reader completed successfully

    def test_sequential_writes_produce_monotonic_ids(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path, "mono.db")
        ids: list[int] = []

        with get_connection(db_path) as conn:
            for i in range(10):
                row_id = insert_event(conn, _make_event(i))
                ids.append(row_id)

        assert ids == sorted(ids)
        assert len(set(ids)) == 10  # all unique

    def test_wal_journal_mode_persists_across_connections(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path, "wal_persist.db")

        with get_connection(db_path) as conn1:
            mode1 = conn1.execute("PRAGMA journal_mode").fetchone()[0]

        with get_connection(db_path) as conn2:
            mode2 = conn2.execute("PRAGMA journal_mode").fetchone()[0]

        assert mode1 == "wal"
        assert mode2 == "wal"

    def test_concurrent_writers_serialized_by_busy_timeout(self, tmp_path: Path) -> None:
        """Two writers must not corrupt the database; busy_timeout resolves contention."""
        db_path = _make_db(tmp_path, "two_writers.db")
        errors: list[str] = []

        def writer(session_id: str, count: int) -> None:
            for i in range(count):
                try:
                    with get_connection(db_path) as conn:
                        insert_event(conn, _make_event(i, session_id=session_id))
                except Exception as exc:
                    errors.append(f"{session_id}: {exc}")

        t1 = threading.Thread(target=writer, args=("s1", 10))
        t2 = threading.Thread(target=writer, args=("s2", 10))
        t1.start()
        t2.start()
        t1.join(timeout=15)
        t2.join(timeout=15)

        assert errors == [], f"Concurrent write errors: {errors}"

        with get_connection(db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM events WHERE project_id='proj1'"
            ).fetchone()[0]
        assert count == 20
