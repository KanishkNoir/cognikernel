"""Tests for Config.load() project-local overlay (Stage C1)."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from memlora.config import VALID_HOOK_POLICIES, Config


def test_defaults_when_no_files_exist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MEMLORA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "noprofile")

    cfg = Config.load()
    assert cfg.hook_policy == "advisory"
    assert cfg.read_cache_ttl_hours == 24
    assert cfg.deny_retry_window_seconds == 60


def test_project_overlay_sets_hook_policy(tmp_path: Path, monkeypatch) -> None:
    """project-local config.toml overrides the global default."""
    monkeypatch.delenv("MEMLORA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "noprofile")

    project = tmp_path / "myproj"
    (project / ".memlora").mkdir(parents=True)
    (project / ".memlora" / "config.toml").write_text(
        'hook_policy = "strict"\n',
        encoding="utf-8",
    )

    cfg = Config.load(project_path=project)
    assert cfg.hook_policy == "strict"


def test_project_overlay_layered_on_global(tmp_path: Path, monkeypatch) -> None:
    """Project config overrides global; global values still apply when not overridden."""
    monkeypatch.delenv("MEMLORA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "userhome")

    # Global ~/.memlora/config.toml sets one field.
    global_dir = tmp_path / "userhome" / ".memlora"
    global_dir.mkdir(parents=True)
    (global_dir / "config.toml").write_text(
        'token_budget = 3000\nhook_policy = "advisory"\n',
        encoding="utf-8",
    )

    # Project config overrides only hook_policy.
    project = tmp_path / "myproj"
    (project / ".memlora").mkdir(parents=True)
    (project / ".memlora" / "config.toml").write_text(
        'hook_policy = "strict"\n',
        encoding="utf-8",
    )

    cfg = Config.load(project_path=project)
    assert cfg.hook_policy == "strict"      # from project overlay
    assert cfg.token_budget == 3000         # inherited from global


def test_memlora_dir_env_var_preserved_when_project_overlay_applies(
    tmp_path: Path, monkeypatch,
) -> None:
    """The MEMLORA_DIR env override survives project overlay (key fix for hooks)."""
    custom_dir = tmp_path / "custom_memlora"
    monkeypatch.setenv("MEMLORA_DIR", str(custom_dir))

    project = tmp_path / "myproj"
    (project / ".memlora").mkdir(parents=True)
    (project / ".memlora" / "config.toml").write_text(
        'hook_policy = "strict"\n',
        encoding="utf-8",
    )

    cfg = Config.load(project_path=project)
    assert cfg.memlora_dir == custom_dir
    assert cfg.hook_policy == "strict"      # both env and project apply


def test_project_config_can_override_memlora_dir(
    tmp_path: Path, monkeypatch,
) -> None:
    """Project config explicitly setting memlora_dir wins over env."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "env_dir"))

    project = tmp_path / "myproj"
    custom = tmp_path / "project_specific"
    (project / ".memlora").mkdir(parents=True)
    (project / ".memlora" / "config.toml").write_text(
        f'memlora_dir = "{custom.as_posix()}"\n',
        encoding="utf-8",
    )

    cfg = Config.load(project_path=project)
    assert cfg.memlora_dir == Path(custom.as_posix())


def test_invalid_hook_policy_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("MEMLORA_DIR", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "noprofile")

    project = tmp_path / "myproj"
    (project / ".memlora").mkdir(parents=True)
    (project / ".memlora" / "config.toml").write_text(
        'hook_policy = "yolo"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid hook_policy"):
        Config.load(project_path=project)


def test_valid_hook_policies_set() -> None:
    assert VALID_HOOK_POLICIES == frozenset({"advisory", "strict"})
