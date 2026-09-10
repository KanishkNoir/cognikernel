import sqlite3
from pathlib import Path

import pytest

from cognikernel.config import EXPECTED_PROJECTION_VERSION, EXPECTED_SCHEMA_VERSION
from cognikernel.storage.connection import get_connection
from cognikernel.storage.events import Event, insert_event
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


class TestMigration021BeliefHistory:
    """T-102 (#12): supersession/archival transitions + commit anchor.

    The store records supersession and archival as STATES (superseded_by,
    archived) but not as TRANSITIONS -- these four columns are what let a
    later belief-status/replay feature answer "when" and "why", not just
    "that". They are deliberately never backfilled: a guessed sha or an
    inferred timestamp on a pre-021 row would make future replay confidently
    wrong rather than honestly bounded (NULL).
    """

    _NEW_COLUMNS = {"superseded_at", "archived_at", "captured_at_sha", "supersede_reason"}

    def test_new_columns_present_on_fresh_store(self, tmp_path: Path) -> None:
        db_path = tmp_path / "fresh.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(events)").fetchall()}
        assert self._NEW_COLUMNS.issubset(cols)

    def test_new_columns_default_to_null(self, tmp_path: Path) -> None:
        db_path = tmp_path / "defaults.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            event_id = insert_event(
                conn,
                Event(
                    project_id="proj1", session_id="sess1", event_type="DECISION",
                    payload={"description": "x"}, content_hash="x",
                ),
            )
            row = conn.execute(
                "SELECT superseded_at, archived_at, captured_at_sha, supersede_reason "
                "FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
        assert tuple(row) == (None, None, None, None)

    def test_superseded_at_index_exists(self, tmp_path: Path) -> None:
        db_path = tmp_path / "idx.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            indexes = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
        assert "idx_events_superseded_at" in indexes

    def test_reaches_the_latest_schema_version(self, tmp_path: Path) -> None:
        db_path = tmp_path / "ver.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            version = int(
                conn.execute(
                    "SELECT value FROM meta WHERE key = 'schema_version'"
                ).fetchone()[0]
            )
        assert version == 22 == EXPECTED_SCHEMA_VERSION

    def test_upgrading_a_real_v20_store_preserves_existing_rows(
        self, tmp_path: Path
    ) -> None:
        """Build a store with the REAL 001-020 migrations only (a faithful v20
        shape, not a synthetic stand-in), insert data, then apply 021 and
        prove every pre-existing column survives byte-for-bit and the four
        new columns land as NULL rather than a backfilled guess.

        This is the automated, CI-resident form of the same guarantee
        rehearsed by hand against a real ~/.cognikernel store during review
        (296 real events, zero drift, original file's sha256 unchanged) --
        that manual pass isn't reproducible in CI since it depends on a
        specific local store, so this locks the same invariant in here.
        """
        from cognikernel.storage.migrations import _MIGRATIONS_DIR as REAL_DIR

        v20_dir = tmp_path / "migrations_v20"
        v20_dir.mkdir()
        for f in sorted(REAL_DIR.glob("*.sql")):
            version = int(f.stem.split("_")[0])
            if version <= 20:
                (v20_dir / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")

        db_path = tmp_path / "upgrade.db"
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = OFF")

        import cognikernel.storage.migrations as migrations_module
        original_dir = migrations_module._MIGRATIONS_DIR
        migrations_module._MIGRATIONS_DIR = v20_dir
        try:
            run_migrations(conn)
        finally:
            migrations_module._MIGRATIONS_DIR = original_dir

        version_before = int(
            conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        )
        assert version_before == 20

        # Raw SQL insert deliberately, NOT insert_event() -- this simulates a
        # row written by the application code AS IT EXISTED AT v20, before
        # captured_at_sha existed as a column. insert_event() today always
        # writes captured_at_sha (T-103), so calling it here would insert
        # into a table that doesn't have that column yet and raise.
        import json as _json
        cursor = conn.execute(
            """
            INSERT INTO events
                (project_id, session_id, created_at, event_type, payload,
                 content_hash, weight, mention_count, superseded_by, archived)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 0)
            """,
            (
                "proj1", "sess1", 1700000000000, "CONSTRAINT_HARD",
                _json.dumps({"description": "pre-existing constraint"}),
                "pre021", 2.5, 3,
            ),
        )
        event_id = cursor.lastrowid
        conn.commit()
        before_row = dict(
            conn.execute(
                "SELECT project_id, session_id, event_type, payload, content_hash, "
                "weight, mention_count, superseded_by, archived FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
        )

        # Now upgrade the SAME connection/data to the real, full migration set.
        run_migrations(conn)

        version_after = int(
            conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        )
        assert version_after == EXPECTED_SCHEMA_VERSION

        after_row = dict(
            conn.execute(
                "SELECT project_id, session_id, event_type, payload, content_hash, "
                "weight, mention_count, superseded_by, archived FROM events WHERE id = ?",
                (event_id,),
            ).fetchone()
        )
        assert after_row == before_row, "a pre-021 row's existing columns must not change"

        new_cols_row = conn.execute(
            "SELECT superseded_at, archived_at, captured_at_sha, supersede_reason "
            "FROM events WHERE id = ?",
            (event_id,),
        ).fetchone()
        assert tuple(new_cols_row) == (None, None, None, None), (
            "a pre-021 row must get NULL, never a backfilled guess"
        )
        conn.close()

    def test_rerunning_migrations_after_v21_does_not_duplicate_columns(
        self, tmp_path: Path
    ) -> None:
        """Regression guard: the fast path (current >= EXPECTED_SCHEMA_VERSION)
        must actually prevent 021 from being re-applied, or a second run would
        raise 'duplicate column name' on the ALTER TABLE ADD COLUMN lines."""
        db_path = tmp_path / "rerun.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            run_migrations(conn)  # must not raise
            cols = [r[1] for r in conn.execute("PRAGMA table_info(events)").fetchall()]
        # Exactly one of each new column, not two.
        for c in self._NEW_COLUMNS:
            assert cols.count(c) == 1


class TestMigration022TelemetryRoundTrips:
    """022 adds the round-trip counters to api_telemetry and labels every row that
    already exists as 'per_line_legacy' — those rows were ingested while usage was
    summed once per transcript line, which inflated them 2-3x (S4 T-401/T-403)."""

    _NEW_COLUMNS = {
        "responses", "memory_tool_responses", "denied_responses",
        "retried_denials", "usage_basis",
    }

    def test_new_columns_exist(self, tmp_path: Path) -> None:
        with get_connection(tmp_path / "cols.db") as conn:
            run_migrations(conn)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(api_telemetry)").fetchall()}
        assert self._NEW_COLUMNS.issubset(cols)

    def test_upgrading_a_real_v21_store_labels_existing_rows_legacy(self, tmp_path: Path) -> None:
        import sqlite3 as _sqlite3
        import cognikernel.storage.migrations as migrations_module
        from cognikernel.storage.migrations import _MIGRATIONS_DIR as REAL_DIR

        v21_dir = tmp_path / "migrations_v21"
        v21_dir.mkdir()
        for f in sorted(REAL_DIR.glob("*.sql")):
            if int(f.stem.split("_")[0]) <= 21:
                (v21_dir / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")

        db_path = tmp_path / "upgrade022.db"
        conn = _sqlite3.connect(str(db_path))
        conn.row_factory = _sqlite3.Row
        original_dir = migrations_module._MIGRATIONS_DIR
        migrations_module._MIGRATIONS_DIR = v21_dir
        try:
            run_migrations(conn)
        finally:
            migrations_module._MIGRATIONS_DIR = original_dir
        assert conn.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0] == "21"

        # Raw insert in the v21 shape — the columns 022 adds do not exist yet.
        conn.execute(
            "INSERT INTO api_telemetry (project_id, session_id, input_tokens, "
            "cache_creation_tokens, cache_read_tokens, output_tokens, ingested_at) "
            "VALUES ('p', 's-old', 600, 300, 90000, 30, 1)"
        )
        conn.commit()

        run_migrations(conn)

        row = conn.execute(
            "SELECT input_tokens, cache_creation_tokens, cache_read_tokens, output_tokens, "
            "responses, memory_tool_responses, denied_responses, retried_denials, usage_basis "
            "FROM api_telemetry WHERE session_id = 's-old'"
        ).fetchone()
        conn.close()
        assert tuple(row) == (600, 300, 90000, 30, 0, 0, 0, 0, "per_line_legacy")
