"""Migration 022 crash-safety, legacy labelling and concurrency (S4 T-403).

022 adds the round-trip counters to `api_telemetry` and a `usage_basis` column
whose DEFAULT labels every pre-existing row 'per_line_legacy'. Those rows were
ingested while usage was summed once per transcript line, which inflated them
2-3x; rows whose transcript is gone can never be re-ingested, so the label is the
only thing that stops them being averaged in silently with corrected ones.

Mirrors tests/reliability/test_migration_021.py against 022's REAL SQL:
  1. Crash mid-script -> rolls back cleanly -> a normal re-run reaches v22.
  2. Existing rows are labelled legacy on upgrade; their token columns unchanged.
  3. Two connections racing a v21 store both reach v22, no duplicate columns.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

import cognikernel.storage.migrations as migrations_module
from cognikernel.storage.connection import get_connection
from cognikernel.storage.migrations import _MIGRATIONS_DIR, run_migrations

_MIGRATION_022 = "022_telemetry_round_trips.sql"
_NEW_COLUMNS = {
    "responses", "memory_tool_responses", "denied_responses",
    "retried_denials", "usage_basis",
}


def _build_v21_dir(dest: Path) -> None:
    dest.mkdir(exist_ok=True)
    for f in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        if int(f.stem.split("_")[0]) <= 21:
            (dest / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")


def _build_v21_store(tmp_path: Path, name: str = "v21.db") -> Path:
    db_path = tmp_path / name
    v21_dir = tmp_path / f"migrations_v21_{name}"
    _build_v21_dir(v21_dir)
    conn = sqlite3.connect(str(db_path))
    original = migrations_module._MIGRATIONS_DIR
    migrations_module._MIGRATIONS_DIR = v21_dir
    try:
        run_migrations(conn)
    finally:
        migrations_module._MIGRATIONS_DIR = original
    conn.close()
    version = sqlite3.connect(str(db_path)).execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()[0]
    assert version == "21", f"fixture setup failed: store is at v{version}, not v21"
    return db_path


def _telemetry_columns(conn: sqlite3.Connection) -> list[str]:
    return [r[1] for r in conn.execute("PRAGMA table_info(api_telemetry)").fetchall()]


class TestCrashMidScript:
    def _corrupt_022_dir(self, tmp_path: Path) -> Path:
        """Real 001-021 plus 022 with a broken trailing statement, so every real
        ALTER TABLE runs (uncommitted) before the script fails."""
        corrupt_dir = tmp_path / "migrations_corrupt"
        _build_v21_dir(corrupt_dir)
        real_022 = (_MIGRATIONS_DIR / _MIGRATION_022).read_text(encoding="utf-8")
        broken = real_022 + "\nINSERT INTO does_not_exist_table (x) VALUES (1);\n"
        (corrupt_dir / _MIGRATION_022).write_text(broken, encoding="utf-8")
        return corrupt_dir

    def test_crash_leaves_version_at_21_with_no_new_columns(self, tmp_path: Path) -> None:
        db_path = _build_v21_store(tmp_path)
        corrupt_dir = self._corrupt_022_dir(tmp_path)
        conn = sqlite3.connect(str(db_path))
        original = migrations_module._MIGRATIONS_DIR
        migrations_module._MIGRATIONS_DIR = corrupt_dir
        try:
            with pytest.raises(sqlite3.OperationalError, match="does_not_exist_table"):
                run_migrations(conn)
        finally:
            migrations_module._MIGRATIONS_DIR = original

        version = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0]
        partial = _NEW_COLUMNS & set(_telemetry_columns(conn))
        conn.close()
        assert version == "21"
        assert not partial, f"partial application survived: {partial}"

    def test_subsequent_normal_run_reaches_v22(self, tmp_path: Path) -> None:
        db_path = _build_v21_store(tmp_path)
        corrupt_dir = self._corrupt_022_dir(tmp_path)
        conn = sqlite3.connect(str(db_path))
        original = migrations_module._MIGRATIONS_DIR
        migrations_module._MIGRATIONS_DIR = corrupt_dir
        try:
            with pytest.raises(sqlite3.OperationalError):
                run_migrations(conn)
        finally:
            migrations_module._MIGRATIONS_DIR = original
        conn.close()

        with get_connection(db_path) as conn2:
            run_migrations(conn2)
            version = conn2.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()[0]
            cols = set(_telemetry_columns(conn2))
        assert version == "22"
        assert _NEW_COLUMNS <= cols


class TestLegacyRowsAreLabelled:
    def test_existing_rows_become_per_line_legacy_with_tokens_unchanged(
        self, tmp_path: Path
    ) -> None:
        db_path = _build_v21_store(tmp_path)
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT INTO api_telemetry (project_id, session_id, input_tokens, "
            "cache_creation_tokens, cache_read_tokens, output_tokens, ingested_at) "
            "VALUES ('p', 'inflated', 1500, 800, 120000, 90, 1)"
        )
        conn.commit()
        conn.close()

        with get_connection(db_path) as conn2:
            run_migrations(conn2)
            row = conn2.execute(
                "SELECT input_tokens, cache_creation_tokens, cache_read_tokens, "
                "output_tokens, responses, memory_tool_responses, denied_responses, "
                "retried_denials, usage_basis FROM api_telemetry WHERE session_id = 'inflated'"
            ).fetchone()
        assert tuple(row) == (1500, 800, 120000, 90, 0, 0, 0, 0, "per_line_legacy")


class TestConcurrentOpen:
    def test_two_connections_racing_both_reach_v22_without_error(self, tmp_path: Path) -> None:
        db_path = _build_v21_store(tmp_path, name="race.db")
        # Warm WAL once first, exactly as `cognikernel init` does before any
        # concurrent access is possible (see test_migration_021.py).
        with get_connection(db_path):
            pass

        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def _race() -> None:
            try:
                with get_connection(db_path) as conn:
                    barrier.wait(timeout=10)
                    run_migrations(conn)
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_race) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"racing connection(s) raised: {errors}"
        assert not any(t.is_alive() for t in threads), "a racing connection hung"

        with get_connection(db_path) as conn:
            version = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()[0]
            cols = _telemetry_columns(conn)
        assert version == "22"
        dupes = {c for c in cols if cols.count(c) > 1}
        assert not dupes, f"duplicate columns from a double-applied migration: {dupes}"
