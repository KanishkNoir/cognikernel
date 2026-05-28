"""Tests for `memlora init` CLI handler — verifies C1 artifacts are written.

The CLI `init` is responsible for setting up a project so the hook chain works
end-to-end. C1 adds two new responsibilities:
  - Register PostToolUse:Read in .claude/settings.json
  - Write .memlora/config.toml with hook_policy=strict
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from memlora.config import Config
from memlora.integration.cli import _cmd_init


@pytest.fixture
def init_args(tmp_path: Path) -> argparse.Namespace:
    """Build the argparse.Namespace _cmd_init expects."""
    project_path = tmp_path / "newproj"
    project_path.mkdir()
    return argparse.Namespace(project_path=str(project_path))


def test_init_writes_claude_settings_with_posttool_read_hook(
    init_args: argparse.Namespace, monkeypatch, tmp_path: Path,
) -> None:
    """The new PostToolUse:Read entry is registered alongside Write/Edit."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))

    _cmd_init(init_args)

    settings_path = Path(init_args.project_path) / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    matchers = {entry["matcher"] for entry in settings["hooks"]["PostToolUse"]}

    assert matchers == {"Write", "Edit", "Read"}
    # Each entry should reference a memlora_*_hook.py script.
    for entry in settings["hooks"]["PostToolUse"]:
        cmd = entry["hooks"][0]["command"]
        assert "memlora_posttool" in cmd


def test_init_writes_per_project_config_with_strict_policy(
    init_args: argparse.Namespace, monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))

    _cmd_init(init_args)

    cfg_path = Path(init_args.project_path) / ".memlora" / "config.toml"
    assert cfg_path.exists()
    content = cfg_path.read_text(encoding="utf-8")
    assert 'hook_policy = "strict"' in content


def test_config_load_picks_up_init_artifacts(
    init_args: argparse.Namespace, monkeypatch, tmp_path: Path,
) -> None:
    """End-to-end: after init, Config.load(project_path=...) reads strict policy."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))

    _cmd_init(init_args)

    cfg = Config.load(project_path=Path(init_args.project_path))
    assert cfg.hook_policy == "strict"


def test_init_preserves_existing_settings_keys(
    init_args: argparse.Namespace, monkeypatch, tmp_path: Path,
) -> None:
    """If .claude/settings.json already has user-defined keys, they survive init."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))

    settings_dir = Path(init_args.project_path) / ".claude"
    settings_dir.mkdir(parents=True)
    settings_path = settings_dir / "settings.json"
    settings_path.write_text(
        json.dumps({"customUserKey": "should_survive"}, indent=2),
        encoding="utf-8",
    )

    _cmd_init(init_args)

    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["customUserKey"] == "should_survive"
    # And the new artifacts are present too.
    assert "Read" in {e["matcher"] for e in settings["hooks"]["PostToolUse"]}


def test_init_idempotent_does_not_overwrite_existing_project_config(
    init_args: argparse.Namespace, monkeypatch, tmp_path: Path,
) -> None:
    """Re-running init must not stomp a user-modified .memlora/config.toml."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))

    # Pre-create the project config with advisory policy (simulating a user opt-out).
    memlora_dir = Path(init_args.project_path) / ".memlora"
    memlora_dir.mkdir(parents=True)
    (memlora_dir / "config.toml").write_text(
        'hook_policy = "advisory"\n',
        encoding="utf-8",
    )

    _cmd_init(init_args)

    content = (memlora_dir / "config.toml").read_text(encoding="utf-8")
    # Their advisory setting must survive.
    assert 'hook_policy = "advisory"' in content


def test_init_claude_md_mentions_strict_mode_and_skeleton(
    init_args: argparse.Namespace, monkeypatch, tmp_path: Path,
) -> None:
    """B-4: the CLAUDE.md template tells Claude about strict mode + skeleton
    authority so the deny behavior isn't a surprise."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))

    _cmd_init(init_args)

    claude_md_path = Path(init_args.project_path) / "CLAUDE.md"
    text = claude_md_path.read_text(encoding="utf-8")
    assert "Codebase skeleton" in text
    assert "PreToolUse" in text
    assert "60 seconds" in text  # retry window mentioned explicitly
