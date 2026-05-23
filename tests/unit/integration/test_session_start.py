"""Tests for memlora.integration.session_start — SessionStart hook logic."""
from __future__ import annotations

from pathlib import Path

import pytest

from memlora.config import Config
from memlora.integration.session import init_project
from memlora.integration.session_start import handle_compact_event


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(memlora_dir=tmp_path / "memlora")


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    p = tmp_path / "myproject"
    p.mkdir()
    return p


class TestHandleCompactEvent:
    def test_returns_string_for_initialized_project(
        self, project_path: Path, cfg: Config
    ) -> None:
        init_project(project_path, config=cfg)
        result = handle_compact_event(str(project_path), config=cfg)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_returns_empty_for_nonexistent_project(
        self, tmp_path: Path, cfg: Config
    ) -> None:
        result = handle_compact_event(str(tmp_path / "ghost"), config=cfg)
        assert result == ""

    def test_returns_empty_for_empty_cwd(self, cfg: Config) -> None:
        result = handle_compact_event("", config=cfg)
        assert result == ""

    def test_contains_session_header(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        result = handle_compact_event(str(project_path), config=cfg)
        assert "auto-generated" in result

    def test_contains_project_name(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        result = handle_compact_event(str(project_path), config=cfg)
        assert project_path.name in result

    def test_idempotent_called_twice(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        r1 = handle_compact_event(str(project_path), config=cfg)
        r2 = handle_compact_event(str(project_path), config=cfg)
        assert r1 == r2

    def test_none_cwd_returns_empty(self, cfg: Config) -> None:
        result = handle_compact_event(None, config=cfg)  # type: ignore[arg-type]
        assert result == ""
