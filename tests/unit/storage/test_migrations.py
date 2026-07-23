import sqlite3
from pathlib import Path

import pytest

from cognikernel.config import EXPECTED_PROJECTION_VERSION, EXPECTED_SCHEMA_VERSION
from cognikernel.storage.connection import get_connection
from cognikernel.storage.migrations import (
    _bootstrap_meta,
    _run_schema_migrations,
    run_migrations,
)


def _fresh_conn(tmp_path: Path, name: str = "test.db"):
    db_path = tmp_path / name
    conn = get_connection(db_path).__enter__()
    return conn, db_path


class TestInitialMigration:
    def test_creates_events_table(self, tmp_path: Path) -> None:
        db_path = tmp_path / "m.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "events" in tables

    def test_creates_state_projections_table(self, tmp_path: Path) -> None:
        db_path = tmp_path / "m.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "state_projections" in tables

    def test_creates_extraction_failures_table(self, tmp_path: Path) -> None:
        db_path = tmp_path / "m.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert "extraction_failures" in tables

    def test_creates_raw_evidence_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "m.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert {"raw_evidence", "event_provenance"}.issubset(tables)

    def test_creates_extraction_job_tables(self, tmp_path: Path) -> None:
        db_path = tmp_path / "m.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert {"extraction_jobs", "extraction_job_acks"}.issubset(tables)

    def test_events_table_columns(self, tmp_path: Path) -> None:
        db_path = tmp_path / "m.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(events)").fetchall()
            }
        required = {
            "id", "project_id", "session_id", "created_at", "event_type",
            "payload", "content_hash", "weight", "mention_count",
            "superseded_by", "archived", "evidence_id",
        }
        assert required.issubset(cols)

    def test_raw_evidence_columns(self, tmp_path: Path) -> None:
        db_path = tmp_path / "m.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(raw_evidence)").fetchall()
            }
        required = {
            "id", "project_id", "session_id", "source_type", "source_path",
            "captured_at", "content_sha256", "content_encoding",
            "content_blob", "original_size_bytes", "stored_size_bytes",
            "metadata",
        }
        assert required.issubset(cols)

    def test_extraction_jobs_columns(self, tmp_path: Path) -> None:
        db_path = tmp_path / "m.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(extraction_jobs)").fetchall()
            }
        required = {
            "id", "project_id", "session_id", "evidence_id", "trace_id",
            "job_category", "stage", "state", "failure_class", "last_error",
            "claimed_by", "claimed_at", "attempts", "max_attempts",
            "soft_timeout_ms", "hard_timeout_ms", "created_at", "updated_at",
        }
        assert required.issubset(cols)

    def test_state_projections_columns(self, tmp_path: Path) -> None:
        db_path = tmp_path / "m.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            cols = {
                r[1]
                for r in conn.execute("PRAGMA table_info(state_projections)").fetchall()
            }
        required = {
            "project_id", "built_at", "event_id_high_water",
            "hard_constraints", "ranked_decisions", "component_map",
            "graveyard", "active_threads", "summary",
        }
        assert required.issubset(cols)


class TestIndexes:
    def _get_indexes(self, tmp_path: Path) -> set:
        db_path = tmp_path / "i.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            return {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }

    def test_unique_content_hash_index(self, tmp_path: Path) -> None:
        assert "idx_events_content_hash" in self._get_indexes(tmp_path)

    def test_project_session_index(self, tmp_path: Path) -> None:
        assert "idx_events_project_session" in self._get_indexes(tmp_path)

    def test_project_type_archived_index(self, tmp_path: Path) -> None:
        assert "idx_events_project_type_archived" in self._get_indexes(tmp_path)

    def test_weight_index(self, tmp_path: Path) -> None:
        assert "idx_events_weight" in self._get_indexes(tmp_path)

    def test_raw_evidence_indexes(self, tmp_path: Path) -> None:
        indexes = self._get_indexes(tmp_path)
        assert "idx_raw_evidence_project_session" in indexes
        assert "idx_event_provenance_evidence" in indexes

    def test_extraction_job_indexes(self, tmp_path: Path) -> None:
        indexes = self._get_indexes(tmp_path)
        assert "idx_extraction_jobs_state_stage" in indexes
        assert "idx_extraction_job_acks_job" in indexes


class TestVersions:
    def test_schema_version_set_to_expected(self, tmp_path: Path) -> None:
        db_path = tmp_path / "v.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            version = int(
                conn.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()[0]
            )
        assert version == EXPECTED_SCHEMA_VERSION

    def test_projection_version_set_to_expected(self, tmp_path: Path) -> None:
        db_path = tmp_path / "v.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            version = int(
                conn.execute(
                    "SELECT value FROM meta WHERE key = 'projection_version'"
                ).fetchone()[0]
            )
        assert version == EXPECTED_PROJECTION_VERSION


class TestAtomicity:
    """A migration's body and its version bump must commit as one transaction.

    Regression guard for the audit P1: migration 017 does a multi-statement table
    rebuild (RENAME -> CREATE -> INSERT -> DROP). Before the fix, executescript
    auto-committed each DDL and the version was bumped separately, so a crash
    after the RENAME left the table gone, no replacement, and the OLD version
    recorded — every subsequent startup re-ran the migration, the RENAME failed,
    and the DB never booted again.
    """

    def _seed_dir(self, tmp_path: Path) -> Path:
        mdir = tmp_path / "migrations"
        mdir.mkdir()
        # 001: a clean migration that establishes a table with data.
        (mdir / "001_base.sql").write_text(
            "CREATE TABLE t (id INTEGER);\nINSERT INTO t VALUES (1);\n",
            encoding="utf-8",
        )
        # 002: mirrors the 017 rebuild shape but fails mid-script — the INSERT
        # references a table that does not exist, AFTER the RENAME has run.
        (mdir / "002_break.sql").write_text(
            "ALTER TABLE t RENAME TO t_old;\n"
            "CREATE TABLE t (id INTEGER, extra TEXT);\n"
            "INSERT INTO t (id) SELECT id FROM does_not_exist;\n"
            "DROP TABLE t_old;\n",
            encoding="utf-8",
        )
        return mdir

    def test_partial_failure_rolls_back_and_keeps_old_version(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        mdir = self._seed_dir(tmp_path)
        monkeypatch.setattr(
            "cognikernel.storage.migrations._MIGRATIONS_DIR", mdir
        )
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _bootstrap_meta(conn)

        with pytest.raises(sqlite3.OperationalError):
            _run_schema_migrations(conn)

        # 001 applied and committed -> version 1; 002 rolled back fully -> NOT 2.
        version = int(
            conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()[0]
        )
        assert version == 1

        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        # The RENAME was rolled back: original table intact, no orphaned _old,
        # no half-built replacement leaking the new column.
        assert "t" in tables
        assert "t_old" not in tables
        cols = {r[1] for r in conn.execute("PRAGMA table_info(t)").fetchall()}
        assert cols == {"id"}
        # Original row survived.
        assert conn.execute("SELECT id FROM t").fetchone()[0] == 1

    def test_connection_usable_after_failed_migration(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The rollback must leave no dangling transaction — the connection is
        immediately writable again (a left-open tx would lock the next write)."""
        mdir = self._seed_dir(tmp_path)
        monkeypatch.setattr(
            "cognikernel.storage.migrations._MIGRATIONS_DIR", mdir
        )
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _bootstrap_meta(conn)
        with pytest.raises(sqlite3.OperationalError):
            _run_schema_migrations(conn)
        # No "cannot start a transaction within a transaction" / lock errors.
        conn.execute("CREATE TABLE probe (x INTEGER)")
        conn.execute("INSERT INTO probe VALUES (1)")
        conn.commit()
        assert conn.execute("SELECT x FROM probe").fetchone()[0] == 1


class TestIdempotency:
    def test_run_twice_does_not_raise(self, tmp_path: Path) -> None:
        db_path = tmp_path / "idem.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            run_migrations(conn)  # must not raise

    def test_tables_still_present_after_second_run(self, tmp_path: Path) -> None:
        db_path = tmp_path / "idem2.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            run_migrations(conn)
            tables = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert {"events", "state_projections", "extraction_failures"}.issubset(tables)
