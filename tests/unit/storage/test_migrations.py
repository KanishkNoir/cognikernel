from pathlib import Path

from memlora.config import EXPECTED_PROJECTION_VERSION, EXPECTED_SCHEMA_VERSION
from memlora.storage.connection import get_connection
from memlora.storage.migrations import run_migrations


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
