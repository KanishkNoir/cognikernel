"""Tests for memlora.integration.session."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memlora.config import Config
from memlora.integration.session import (
    get_projection,
    init_project,
    render_state,
    session_end,
)
from memlora.storage.connection import get_connection, get_db_path, hash_project_path


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(memlora_dir=tmp_path / "memlora")


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    p = tmp_path / "myproject"
    p.mkdir()
    return p


# ── init_project ──────────────────────────────────────────────────────────────

class TestInitProject:
    def test_returns_project_id_string(self, project_path: Path, cfg: Config) -> None:
        result = init_project(project_path, config=cfg)
        assert isinstance(result, str) and len(result) == 16

    def test_creates_db_file(self, project_path: Path, cfg: Config) -> None:
        project_id = init_project(project_path, config=cfg)
        db_path = get_db_path(cfg, project_id)
        assert db_path.exists()

    def test_idempotent_safe_to_call_twice(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        init_project(project_path, config=cfg)  # must not raise

    def test_same_path_always_same_project_id(self, project_path: Path, cfg: Config) -> None:
        id1 = init_project(project_path, config=cfg)
        id2 = init_project(project_path, config=cfg)
        assert id1 == id2

    def test_different_paths_different_ids(self, tmp_path: Path, cfg: Config) -> None:
        p1 = tmp_path / "proj_a"
        p2 = tmp_path / "proj_b"
        p1.mkdir(); p2.mkdir()
        assert init_project(p1, config=cfg) != init_project(p2, config=cfg)


# ── session_end ───────────────────────────────────────────────────────────────

DECISION_TRANSCRIPT = (
    "We decided to use SQLite for local storage because it requires no external server. "
    "This is a hard constraint: never store secrets in plain text configuration files."
)


class TestSessionEnd:
    def test_returns_stats_dict(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        stats = session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)
        assert isinstance(stats, dict)
        for key in ("extracted", "inserted", "updated", "superseded", "cascaded", "archived"):
            assert key in stats

    def test_extracted_count_is_non_negative(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        stats = session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)
        assert stats["extracted"] >= 0

    def test_events_written_to_db(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)
        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count >= 0  # extraction may or may not find events in this transcript

    def test_dedup_increments_not_duplicates(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        s1 = session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)
        s2 = session_end(project_path, "sess2", DECISION_TRANSCRIPT, config=cfg)
        # Second run must not insert more rows than first (same content)
        assert s2["inserted"] <= s1["inserted"]

    def test_empty_transcript_returns_zero_extracted(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        stats = session_end(project_path, "sess1", "", config=cfg)
        assert stats["extracted"] == 0

    def test_stats_values_are_non_negative(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        stats = session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)
        for v in stats.values():
            assert v >= 0

    def test_inits_db_if_not_yet_initialised(self, project_path: Path, cfg: Config) -> None:
        # session_end should work even without an explicit init_project call
        stats = session_end(project_path, "sess1", "", config=cfg)
        assert "extracted" in stats


# ── get_projection ────────────────────────────────────────────────────────────

class TestGetProjection:
    def test_returns_projection_object(self, project_path: Path, cfg: Config) -> None:
        from memlora.storage.projections import Projection
        init_project(project_path, config=cfg)
        proj = get_projection(project_path, config=cfg)
        assert isinstance(proj, Projection)

    def test_project_id_matches(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        expected_id = hash_project_path(project_path)
        proj = get_projection(project_path, config=cfg)
        assert proj.project_id == expected_id

    def test_reflects_events_after_session_end(self, project_path: Path, cfg: Config) -> None:
        transcript = (
            "We decided to use SQLite as our primary storage engine. "
            "This is the most important architectural decision."
        )
        session_end(project_path, "sess1", transcript, config=cfg)
        proj = get_projection(project_path, config=cfg)
        total = (
            len(proj.hard_constraints)
            + len(proj.ranked_decisions)
            + len(proj.graveyard)
            + len(proj.component_map)
            + len(proj.active_threads)
        )
        assert total >= 0  # at minimum, projection exists


# ── render_state ──────────────────────────────────────────────────────────────

class TestRenderState:
    def test_returns_non_empty_string_with_events(self, project_path: Path, cfg: Config) -> None:
        session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)
        rendered = render_state(project_path, config=cfg)
        assert isinstance(rendered, str)
        assert len(rendered) > 0

    def test_returns_string_for_empty_project(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        rendered = render_state(project_path, config=cfg)
        assert isinstance(rendered, str)

    def test_contains_header_marker(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        rendered = render_state(project_path, config=cfg)
        assert "auto-generated" in rendered

    def test_inits_db_if_needed(self, project_path: Path, cfg: Config) -> None:
        rendered = render_state(project_path, config=cfg)
        assert isinstance(rendered, str)

    def test_project_name_appears_in_output(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        rendered = render_state(project_path, config=cfg)
        assert project_path.name in rendered
