"""Tests for cognikernel.integration.session_start — SessionStart hook logic."""
from __future__ import annotations

from pathlib import Path

import pytest

from cognikernel.config import Config
from cognikernel.integration.session import init_project
from cognikernel.integration.session_start import handle_compact_event, handle_session_start


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(cognikernel_dir=tmp_path / "cognikernel")


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


class TestProjectOverlayPickup:
    """Regression: session_start must load Config with project_path so the
    per-project `.cognikernel/config.toml` overlay (e.g. hook_policy="strict")
    flows into the rendered injection. The original bug was a no-arg
    `Config.load()` call that silently dropped strict-mode rendering."""

    def test_strict_overlay_renders_tool_policy_section(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        cognikernel_dir = tmp_path / "cognikernel_data"
        monkeypatch.setenv("COGNIKERNEL_DIR", str(cognikernel_dir))

        project_path = tmp_path / "myproject"
        project_path.mkdir()
        # Per-project overlay: enable strict mode for this project only.
        (project_path / ".cognikernel").mkdir()
        (project_path / ".cognikernel" / "config.toml").write_text(
            'hook_policy = "strict"\n', encoding="utf-8"
        )

        # init must also flow project_path through Config.load (same bug class)
        init_project(project_path)

        # Critical: no `config=` argument — exercises the Config.load path.
        result = handle_session_start(str(project_path))
        assert "### Tool policy" in result, (
            "Tool Policy section missing — the per-project hook_policy='strict' "
            "overlay was not picked up by Config.load in session_start."
        )

    def test_advisory_default_omits_tool_policy(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Counter-test: with no overlay, default is 'advisory' and the Tool
        Policy section is correctly absent."""
        cognikernel_dir = tmp_path / "cognikernel_data"
        monkeypatch.setenv("COGNIKERNEL_DIR", str(cognikernel_dir))

        project_path = tmp_path / "myproject"
        project_path.mkdir()
        # No .cognikernel/config.toml — should default to advisory mode.
        init_project(project_path)

        result = handle_session_start(str(project_path))
        assert "### Tool policy" not in result
