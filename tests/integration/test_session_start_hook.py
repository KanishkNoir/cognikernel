"""E2E subprocess test for memlora_session_start_hook.py.

Spawns the hook script as a real subprocess (just as Claude Code would),
passes a compact payload on stdin, and asserts the correct JSON output
is written to stdout.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from memlora.config import Config
from memlora.integration.session import init_project


HOOK_SCRIPT = (
    Path(__file__).parent.parent.parent / "scripts" / "memlora_session_start_hook.py"
)


def _run_hook(payload: dict, env_overrides: dict | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_overrides or {})}
    return subprocess.run(
        [sys.executable, str(HOOK_SCRIPT)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


class TestSessionStartHookSubprocess:
    def test_compact_event_returns_json_with_additional_context(
        self, tmp_path: Path
    ) -> None:
        memlora_dir = tmp_path / "memlora"
        project_path = tmp_path / "myproject"
        project_path.mkdir()

        cfg = Config(memlora_dir=memlora_dir)
        init_project(project_path, config=cfg)

        payload = {"source": "compact", "cwd": str(project_path)}
        result = _run_hook(payload, env_overrides={"MEMLORA_DIR": str(memlora_dir)})

        assert result.returncode == 0
        assert result.stdout.strip(), "Hook printed nothing to stdout"
        data = json.loads(result.stdout)
        assert "hookSpecificOutput" in data
        hook_out = data["hookSpecificOutput"]
        assert hook_out.get("hookEventName") == "SessionStart"
        assert "additionalContext" in hook_out
        assert "auto-generated" in hook_out["additionalContext"]

    def test_clear_event_also_triggers_reinject(self, tmp_path: Path) -> None:
        memlora_dir = tmp_path / "memlora"
        project_path = tmp_path / "proj2"
        project_path.mkdir()

        cfg = Config(memlora_dir=memlora_dir)
        init_project(project_path, config=cfg)

        payload = {"source": "clear", "cwd": str(project_path)}
        result = _run_hook(payload, env_overrides={"MEMLORA_DIR": str(memlora_dir)})

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["hookSpecificOutput"]["hookEventName"] == "SessionStart"

    def test_startup_source_injects_context(self, tmp_path: Path) -> None:
        # startup now injects just like compact/clear — hook fires on all sources
        memlora_dir = tmp_path / "memlora"
        project_path = tmp_path / "proj3"
        project_path.mkdir()

        cfg = Config(memlora_dir=memlora_dir)
        init_project(project_path, config=cfg)

        payload = {"source": "startup", "cwd": str(project_path)}
        result = _run_hook(payload, env_overrides={"MEMLORA_DIR": str(memlora_dir)})

        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data["hookSpecificOutput"]["hookEventName"] == "SessionStart"
        assert "CogniKernel is active" in data["hookSpecificOutput"]["additionalContext"]

    def test_missing_cwd_produces_no_output(self, tmp_path: Path) -> None:
        payload = {"source": "compact"}
        result = _run_hook(payload)
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_uninitialised_project_produces_no_output(self, tmp_path: Path) -> None:
        memlora_dir = tmp_path / "memlora"
        # Project path exists on disk but was never memlora-initialised
        project_path = tmp_path / "ghost_proj"
        project_path.mkdir()

        payload = {"source": "compact", "cwd": str(project_path)}
        result = _run_hook(payload, env_overrides={"MEMLORA_DIR": str(memlora_dir)})

        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_hook_never_crashes_on_malformed_stdin(self) -> None:
        result = subprocess.run(
            [sys.executable, str(HOOK_SCRIPT)],
            input="not-valid-json!!!",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_hook_never_crashes_on_empty_stdin(self) -> None:
        result = subprocess.run(
            [sys.executable, str(HOOK_SCRIPT)],
            input="",
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0

    def test_additional_context_contains_project_name(self, tmp_path: Path) -> None:
        memlora_dir = tmp_path / "memlora"
        project_path = tmp_path / "wonderproject"
        project_path.mkdir()

        cfg = Config(memlora_dir=memlora_dir)
        init_project(project_path, config=cfg)

        payload = {"source": "compact", "cwd": str(project_path)}
        result = _run_hook(payload, env_overrides={"MEMLORA_DIR": str(memlora_dir)})

        assert result.returncode == 0
        data = json.loads(result.stdout)
        context = data["hookSpecificOutput"]["additionalContext"]
        assert "wonderproject" in context
