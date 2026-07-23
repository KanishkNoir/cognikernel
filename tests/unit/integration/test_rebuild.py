"""Tests for cognikernel.integration.session.rebuild_from_raw (sidecar mode)."""
from __future__ import annotations

from pathlib import Path

import pytest

from cognikernel.config import Config
from cognikernel.integration.session import rebuild_from_raw, session_end
from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
from cognikernel.storage.migrations import run_migrations


TRANSCRIPT_A = (
    "We decided to use SQLite for local storage because it requires no external server. "
    "This is a hard constraint: never store secrets in plain text configuration files."
)
TRANSCRIPT_B = (
    "We decided to adopt async/await throughout the API layer. "
    "This is a hard constraint: all database calls must use connection pooling."
)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(cognikernel_dir=tmp_path / "cognikernel")


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    p = tmp_path / "myproject"
    p.mkdir()
    return p


@pytest.fixture
def seeded_project(project_path: Path, cfg: Config) -> tuple[Path, Config]:
    """Project with two sessions of evidence already written."""
    session_end(project_path, "sess1", TRANSCRIPT_A, config=cfg)
    session_end(project_path, "sess2", TRANSCRIPT_B, config=cfg)
    return project_path, cfg


# ── dry-run ───────────────────────────────────────────────────────────────────

class TestDryRun:
    def test_dry_run_returns_evidence_count(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        result = rebuild_from_raw(project_path, dry_run=True, config=cfg)
        assert result["dry_run"] is True
        assert result["evidence_count"] >= 1

    def test_dry_run_does_not_create_sidecar(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        result = rebuild_from_raw(project_path, dry_run=True, config=cfg)
        # sidecar_path is reported but the file must not exist
        sidecar = Path(result["sidecar_path"])
        assert not sidecar.exists()

    def test_dry_run_reports_sidecar_path(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        result = rebuild_from_raw(project_path, dry_run=True, config=cfg)
        assert result["sidecar_path"].endswith(".rebuild")

    def test_dry_run_empty_project_returns_zero(
        self, project_path: Path, cfg: Config
    ) -> None:
        result = rebuild_from_raw(project_path, dry_run=True, config=cfg)
        assert result["evidence_count"] == 0


# ── sidecar creation ──────────────────────────────────────────────────────────

class TestSidecarCreation:
    def test_creates_sidecar_db_file(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        result = rebuild_from_raw(project_path, config=cfg)
        assert Path(result["sidecar_path"]).exists()

    def test_sidecar_path_format(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)
        result = rebuild_from_raw(project_path, config=cfg)
        assert result["sidecar_path"] == str(db_path.parent / (db_path.name + ".rebuild"))

    def test_source_db_is_untouched(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)

        with get_connection(db_path) as conn:
            before = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

        rebuild_from_raw(project_path, config=cfg)

        with get_connection(db_path) as conn:
            after = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

        assert before == after

    def test_sidecar_has_valid_schema(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        result = rebuild_from_raw(project_path, config=cfg)
        sidecar_path = Path(result["sidecar_path"])
        with get_connection(sidecar_path) as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        assert {"events", "raw_evidence", "event_provenance"} <= tables

    def test_rebuild_records_source_in_meta(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        result = rebuild_from_raw(project_path, config=cfg)
        sidecar_path = Path(result["sidecar_path"])
        with get_connection(sidecar_path) as conn:
            src = conn.execute(
                "SELECT value FROM meta WHERE key='rebuild_source'"
            ).fetchone()
        assert src is not None


# ── content integrity ─────────────────────────────────────────────────────────

class TestContentIntegrity:
    def test_sidecar_has_events(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        result = rebuild_from_raw(project_path, config=cfg)
        sidecar_path = Path(result["sidecar_path"])
        with get_connection(sidecar_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count > 0
        assert result["total_extracted"] >= 0

    def test_sidecar_evidence_rows_match_source(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)

        with get_connection(db_path) as conn:
            src_count = conn.execute(
                "SELECT COUNT(*) FROM raw_evidence WHERE project_id=?",
                (project_id,),
            ).fetchone()[0]

        result = rebuild_from_raw(project_path, config=cfg)
        sidecar_path = Path(result["sidecar_path"])
        with get_connection(sidecar_path) as conn:
            sidecar_count = conn.execute(
                "SELECT COUNT(*) FROM raw_evidence WHERE project_id=?",
                (project_id,),
            ).fetchone()[0]

        assert sidecar_count == src_count == result["evidence_count"]

    def test_determinism_content_hash_set_matches(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        """Core audit invariant: (event_type, content_hash) set is identical."""
        project_path, cfg = seeded_project
        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)

        with get_connection(db_path) as conn:
            src_keys = {
                (row["event_type"], row["content_hash"])
                for row in conn.execute(
                    "SELECT event_type, content_hash FROM events WHERE project_id=?",
                    (project_id,),
                ).fetchall()
            }

        result = rebuild_from_raw(project_path, config=cfg)
        sidecar_path = Path(result["sidecar_path"])
        with get_connection(sidecar_path) as conn:
            sidecar_keys = {
                (row["event_type"], row["content_hash"])
                for row in conn.execute(
                    "SELECT event_type, content_hash FROM events WHERE project_id=?",
                    (project_id,),
                ).fetchall()
            }

        assert src_keys == sidecar_keys

    def test_sidecar_provenance_links_valid(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        result = rebuild_from_raw(project_path, config=cfg)
        sidecar_path = Path(result["sidecar_path"])
        with get_connection(sidecar_path) as conn:
            # Every provenance row must reference an existing event and evidence.
            orphans = conn.execute(
                """
                SELECT COUNT(*) FROM event_provenance ep
                LEFT JOIN events e ON e.id = ep.event_id
                LEFT JOIN raw_evidence re ON re.id = ep.raw_evidence_id
                WHERE e.id IS NULL OR re.id IS NULL
                """
            ).fetchone()[0]
        assert orphans == 0

    def test_returns_stats_dict_with_expected_keys(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        result = rebuild_from_raw(project_path, config=cfg)
        for key in (
            "sidecar_path",
            "evidence_count",
            "sessions_processed",
            "total_extracted",
            "total_inserted",
            "total_updated",
            "errors",
            "since_evidence_id",
        ):
            assert key in result

    def test_stats_values_are_non_negative(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        result = rebuild_from_raw(project_path, config=cfg)
        for key in ("evidence_count", "sessions_processed", "total_extracted",
                    "total_inserted", "total_updated", "errors"):
            assert result[key] >= 0


# ── since filter ──────────────────────────────────────────────────────────────

class TestSinceFilter:
    def test_since_zero_processes_all_evidence(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        result = rebuild_from_raw(project_path, since_evidence_id=0, config=cfg)
        assert result["evidence_count"] >= 2

    def test_since_high_id_processes_nothing(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        result = rebuild_from_raw(project_path, since_evidence_id=999999, config=cfg)
        assert result["evidence_count"] == 0

    def test_since_filters_to_subset(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)

        with get_connection(db_path) as conn:
            first_id = conn.execute(
                "SELECT MIN(id) FROM raw_evidence WHERE project_id=?",
                (project_id,),
            ).fetchone()[0]

        full = rebuild_from_raw(project_path, since_evidence_id=0, config=cfg)
        partial = rebuild_from_raw(project_path, since_evidence_id=first_id, config=cfg)
        assert partial["evidence_count"] < full["evidence_count"]


# ── idempotency ───────────────────────────────────────────────────────────────

class TestIdempotency:
    def test_second_rebuild_overwrites_first(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project
        r1 = rebuild_from_raw(project_path, config=cfg)
        r2 = rebuild_from_raw(project_path, config=cfg)
        # Both runs succeed and produce a valid sidecar at the same path.
        assert r1["sidecar_path"] == r2["sidecar_path"]
        sidecar_path = Path(r2["sidecar_path"])
        assert sidecar_path.exists()

    def test_repeated_rebuild_produces_same_key_set(
        self, seeded_project: tuple[Path, Config]
    ) -> None:
        project_path, cfg = seeded_project

        def _key_set(sidecar_path: str) -> set:
            with get_connection(Path(sidecar_path)) as conn:
                return {
                    (row["event_type"], row["content_hash"])
                    for row in conn.execute(
                        "SELECT event_type, content_hash FROM events"
                    ).fetchall()
                }

        r1 = rebuild_from_raw(project_path, config=cfg)
        r2 = rebuild_from_raw(project_path, config=cfg)
        assert _key_set(r1["sidecar_path"]) == _key_set(r2["sidecar_path"])


# ── empty project ─────────────────────────────────────────────────────────────

class TestEmptyProject:
    def test_rebuild_empty_project_succeeds(
        self, project_path: Path, cfg: Config
    ) -> None:
        result = rebuild_from_raw(project_path, config=cfg)
        assert result["evidence_count"] == 0
        assert result["errors"] == 0
        assert Path(result["sidecar_path"]).exists()

    def test_rebuild_empty_project_zero_events(
        self, project_path: Path, cfg: Config
    ) -> None:
        result = rebuild_from_raw(project_path, config=cfg)
        sidecar_path = Path(result["sidecar_path"])
        with get_connection(sidecar_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 0
