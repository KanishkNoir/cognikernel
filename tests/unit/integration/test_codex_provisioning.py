"""L4: init provisions Codex (.codex/config.toml + AGENTS.md) + doctor codex check."""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import pytest

from cognikernel.config import Config
from cognikernel.integration.cli import _cmd_init
from cognikernel.integration.health import check_codex


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("COGNIKERNEL_DISABLE_AUTO_WARM", "1")
    p = tmp_path / "proj"
    p.mkdir()
    return p


class TestInitProvisionsCodex:
    def test_writes_codex_config_and_agents(self, project: Path):
        _cmd_init(argparse.Namespace(project_path=str(project)))
        codex_cfg = (project / ".codex" / "config.toml").read_text(encoding="utf-8")
        assert "[mcp_servers.cognikernel]" in codex_cfg
        assert "mcp-serve" in codex_cfg
        assert "cwd =" in codex_cfg
        assert "[mcp_servers.cognikernel.env]" in codex_cfg
        assert "COGNIKERNEL_PROJECT_PATH" in codex_cfg
        agents = (project / "AGENTS.md").read_text(encoding="utf-8")
        assert "CogniKernel" in agents and "codex-sync" in agents and "get_session_state" in agents

    def test_appends_to_existing_codex_config_without_clobber(self, project: Path):
        codex_dir = project / ".codex"
        codex_dir.mkdir()
        (codex_dir / "config.toml").write_text(
            'notify = ["my-existing-notify.exe", "turn-ended"]\n', encoding="utf-8"
        )
        _cmd_init(argparse.Namespace(project_path=str(project)))
        cfg = (codex_dir / "config.toml").read_text(encoding="utf-8")
        assert "my-existing-notify.exe" in cfg          # preserved
        assert "[mcp_servers.cognikernel]" in cfg        # added

    def test_init_is_idempotent_on_codex_config(self, project: Path):
        _cmd_init(argparse.Namespace(project_path=str(project)))
        _cmd_init(argparse.Namespace(project_path=str(project)))  # second run
        cfg = (project / ".codex" / "config.toml").read_text(encoding="utf-8")
        assert cfg.count("[mcp_servers.cognikernel]") == 1   # not duplicated

    def test_rewrites_existing_managed_codex_block(self, project: Path):
        codex_dir = project / ".codex"
        codex_dir.mkdir()
        (codex_dir / "config.toml").write_text(
            'notify = ["keep-me"]\n'
            "\n"
            "[mcp_servers.cognikernel]\n"
            'command = "python"\n'
            'args = ["-m", "cognikernel", "mcp-serve"]\n',
            encoding="utf-8",
        )
        _cmd_init(argparse.Namespace(project_path=str(project)))
        cfg = (codex_dir / "config.toml").read_text(encoding="utf-8")
        assert cfg.count("[mcp_servers.cognikernel]") == 1
        assert "cwd =" in cfg and "COGNIKERNEL_PROJECT_PATH" in cfg
        assert "keep-me" in cfg

    def test_preserves_existing_agents_md(self, project: Path):
        (project / "AGENTS.md").write_text("# My agent rules\nDo the thing.\n", encoding="utf-8")
        _cmd_init(argparse.Namespace(project_path=str(project)))
        agents = (project / "AGENTS.md").read_text(encoding="utf-8")
        assert "My agent rules" in agents and "CogniKernel" in agents


class TestCheckCodex:
    def test_disabled_is_healthy(self):
        c = check_codex(dataclasses.replace(Config(), codex_sync_enabled=False))
        assert c.ok and "disabled" in c.detail

    def test_absent_sessions_dir_is_healthy(self, tmp_path: Path):
        cfg = dataclasses.replace(Config(), codex_home=tmp_path / "no_codex")
        c = check_codex(cfg)
        assert c.ok and "nothing to sync" in c.detail

    def test_present_sessions_dir_reports_count(self, tmp_path: Path):
        sessions = tmp_path / "codex" / "sessions" / "2026" / "06" / "21"
        sessions.mkdir(parents=True)
        (sessions / "rollout-x.jsonl").write_text("{}", encoding="utf-8")
        cfg = dataclasses.replace(Config(), codex_home=tmp_path / "codex")
        c = check_codex(cfg)
        assert c.ok and "1 rollouts" in c.detail
