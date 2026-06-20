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

    assert matchers == {"Write", "Edit", "Read", "Grep"}
    # CK-6a: path-portable `python -m memlora hook-posttool[-read]` (no abs script path).
    for entry in settings["hooks"]["PostToolUse"]:
        cmd = entry["hooks"][0]["command"]
        assert "-m memlora hook-posttool" in cmd


def test_init_pretool_matcher_routes_read_and_write_edit(
    init_args: argparse.Namespace, monkeypatch, tmp_path: Path,
) -> None:
    """PreToolUse must route Read (C1 gate) AND Write/Edit/MultiEdit (K2 JIT
    prohibition surfacing) to hook-pretool — regression guard: the matcher was
    'Read'-only, which silently disabled K2 even though pretool_main dispatches
    those tools."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))
    _cmd_init(init_args)
    settings = json.loads(
        (Path(init_args.project_path) / ".claude" / "settings.json").read_text(encoding="utf-8")
    )
    entries = settings["hooks"]["PreToolUse"]
    matcher_blob = "|".join(e["matcher"] for e in entries)
    for tool in ("Read", "Write", "Edit", "MultiEdit"):
        assert tool in matcher_blob, f"PreToolUse does not route {tool} → hook-pretool"
    assert all("-m memlora hook-pretool" in e["hooks"][0]["command"] for e in entries)


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


def test_init_writes_claude_slash_commands(
    init_args: argparse.Namespace, monkeypatch, tmp_path: Path,
) -> None:
    """Every project gets in-session `.claude/commands/ck-*.md` slash commands."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))

    _cmd_init(init_args)

    commands_dir = Path(init_args.project_path) / ".claude" / "commands"
    expected = {"ck-doctor", "ck-show", "ck-failures", "ck-lookup", "ck-recall", "ck-related"}
    written = {p.stem for p in commands_dir.glob("ck-*.md")}
    assert expected <= written

    # CLI-backed command: frontmatter scopes the Bash permission and the body
    # `!`-executes the CLI against `.` (the project root the `!`-bash runs from).
    doctor = (commands_dir / "ck-doctor.md").read_text(encoding="utf-8")
    assert "allowed-tools: Bash(python -m memlora doctor:*)" in doctor
    assert "!`python -m memlora doctor .`" in doctor

    # Regression guard: the `!`-exec must NOT contain a shell expansion. Claude
    # Code rejects `$VAR` in the permission pre-check ("simple_expansion") and the
    # command silently fails — which is exactly the /ck-show, /ck-doctor breakage.
    for p in commands_dir.glob("ck-*.md"):
        assert "$CLAUDE_PROJECT_DIR" not in p.read_text(encoding="utf-8")

    # MCP-backed command: steers to the tool, no Bash execution.
    recall = (commands_dir / "ck-recall.md").read_text(encoding="utf-8")
    assert "cognikernel `recall` MCP tool" in recall
    assert "allowed-tools" not in recall
    assert "argument-hint: <query>" in recall


def test_init_writes_codex_skills(
    init_args: argparse.Namespace, monkeypatch, tmp_path: Path,
) -> None:
    """Codex picks up repo-level skills from `.agents/skills/<name>/SKILL.md`."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))

    _cmd_init(init_args)

    skills_dir = Path(init_args.project_path) / ".agents" / "skills"
    doctor = skills_dir / "ck-doctor" / "SKILL.md"
    assert doctor.exists()
    text = doctor.read_text(encoding="utf-8")
    assert "name: ck-doctor" in text
    assert "python -m memlora doctor ." in text

    # MCP-backed skill points at the registered cognikernel MCP tool.
    recall = (skills_dir / "ck-recall" / "SKILL.md").read_text(encoding="utf-8")
    assert "cognikernel `recall` MCP tool" in recall


def test_init_claude_md_is_minimal_trust_fallback(
    init_args: argparse.Namespace, monkeypatch, tmp_path: Path,
) -> None:
    """CLAUDE.md is the minimal block-ABSENT fallback: trust the canonical block,
    recover via get_session_state, silent persistence. The full directive contract
    (skeleton re-read gate, recall/find_related) lives in the SessionStart block
    header, NOT here — research/claude_md_design.md."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))

    _cmd_init(init_args)

    text = (Path(init_args.project_path) / "CLAUDE.md").read_text(encoding="utf-8")
    assert "canonical" in text.lower()
    assert "supersede" in text.lower()
    assert "get_session_state" in text          # block-absent recovery


def test_init_claude_md_has_no_meta_narration_or_catalog(
    init_args: argparse.Namespace, monkeypatch, tmp_path: Path,
) -> None:
    """Anti-narration + de-bloat: the silent-persistence rule is present (fixes the
    'lock this into CogniKernel?' behavior) and the mechanism/tool-catalog is NOT
    duplicated here (it's injected every session — keep it small)."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))

    _cmd_init(init_args)

    text = (Path(init_args.project_path) / "CLAUDE.md").read_text(encoding="utf-8")
    assert "never ask to save" in text                 # anti-narration rule
    assert "PageRank" not in text                       # no mechanism explanation
    assert "cognikernel://project/" not in text         # no resource catalog
    # small: the section is a handful of lines, not a feature manual
    assert len(text) < 900
