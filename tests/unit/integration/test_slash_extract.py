"""Tests for the /memlora-extract slash command installer (Phase A-5)."""
from __future__ import annotations

from pathlib import Path

import pytest

from memlora.integration.slash_extract import (
    SLASH_COMMAND_BODY,
    install_slash_command,
)


def test_writes_to_project_local_commands_dir(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()

    result = install_slash_command(project)

    assert result == project / ".claude" / "commands" / "memlora-extract.md"
    assert result.exists()


def test_writes_full_body(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()

    target = install_slash_command(project)

    text = target.read_text(encoding="utf-8")
    assert text == SLASH_COMMAND_BODY


def test_creates_commands_dir_if_missing(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    assert not (project / ".claude").exists()

    install_slash_command(project)

    assert (project / ".claude" / "commands").is_dir()


def test_idempotent_does_not_overwrite_existing(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    commands = project / ".claude" / "commands"
    commands.mkdir(parents=True)
    existing = commands / "memlora-extract.md"
    existing.write_text("USER_CUSTOMIZED\n", encoding="utf-8")

    install_slash_command(project)

    assert existing.read_text(encoding="utf-8") == "USER_CUSTOMIZED\n"


def test_overwrite_flag_replaces_existing(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    project.mkdir()
    commands = project / ".claude" / "commands"
    commands.mkdir(parents=True)
    existing = commands / "memlora-extract.md"
    existing.write_text("USER_CUSTOMIZED\n", encoding="utf-8")

    install_slash_command(project, overwrite=True)

    text = existing.read_text(encoding="utf-8")
    assert text != "USER_CUSTOMIZED\n"
    assert text == SLASH_COMMAND_BODY


def test_body_includes_required_mcp_tool_names(tmp_path: Path) -> None:
    """The slash command must reference both MCP tools so Claude knows
    to call them. A regression here would silently break /memlora-extract."""
    body = SLASH_COMMAND_BODY
    assert "mcp__cognikernel__get_unprocessed_evidence" in body
    assert "mcp__cognikernel__store_extracted_events" in body


def test_body_tells_llm_to_use_version_from_response(tmp_path: Path) -> None:
    """The slash command must tell the LLM to read extractor_version from
    the get_unprocessed_evidence response, NOT hardcode it.

    Version literals may appear in example prose (e.g. "a string like llm-v1")
    — what's forbidden is a *directive* like "use llm-v1"."""
    body_lower = SLASH_COMMAND_BODY.lower()
    # Must mention extractor_version.
    assert "extractor_version" in body_lower
    # Must tell the LLM to use the value from step 1.
    assert "value from step 1" in body_lower or "do not hardcode" in body_lower
    # Must NOT contain a directive that hardcodes the literal.
    assert 'use "llm-v1"' not in body_lower
    assert "extractor_version=\"llm-v1\"" not in body_lower


def test_body_explains_partial_retry_semantics(tmp_path: Path) -> None:
    """The slash command body should explain that version_bumped=false
    means a retry will happen — so the LLM doesn't try to "fix" the
    apparent failure."""
    assert "version_bumped" in SLASH_COMMAND_BODY
    assert "retry" in SLASH_COMMAND_BODY.lower()


# ── cli init wires it ────────────────────────────────────────────────────────


def test_cli_init_installs_slash_command(tmp_path: Path, monkeypatch) -> None:
    """`memlora init` must call install_slash_command for new projects."""
    import argparse
    from memlora.integration.cli import _cmd_init

    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))
    project = tmp_path / "newproj"
    project.mkdir()

    _cmd_init(argparse.Namespace(project_path=str(project)))

    assert (project / ".claude" / "commands" / "memlora-extract.md").exists()
