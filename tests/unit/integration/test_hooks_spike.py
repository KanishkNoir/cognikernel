"""CK-1/CK-3a/CK-4 — dispatch, flag gates, and silence contract for new hooks."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import cognikernel.integration.cli as cli


class TestNewEntrypoints:
    @pytest.mark.parametrize("sub,fn", [
        ("hook-user-prompt", "user_prompt_submit_main"),
        ("hook-subagent-stop", "subagent_stop_main"),
        ("hook-posttool-grep", "posttool_grep_main"),
    ])
    def test_main_dispatches(self, monkeypatch, sub: str, fn: str) -> None:
        called: list[str] = []
        monkeypatch.setattr(f"cognikernel.integration.hooks.{fn}", lambda: called.append(fn))
        monkeypatch.setattr(sys, "argv", ["cognikernel", sub])
        cli.main()
        assert called == [fn]

    def test_all_nine_entrypoints_have_callables(self) -> None:
        import cognikernel.integration.hooks as hooks
        assert len(cli._HOOK_ENTRYPOINTS) == 8
        for fn in cli._HOOK_ENTRYPOINTS.values():
            assert callable(getattr(hooks, fn)), fn


class TestUserPromptSubmitTimeout:
    """Audit P1: a stalled recall must not block the prompt past the budget.

    The old `with ThreadPoolExecutor()` joined the worker on block-exit (and at
    interpreter shutdown), so the result-timeout was illusory. The daemon-thread
    form abandons a slow recall and returns within budget.
    """

    def test_stalled_recall_returns_within_budget_and_silent(
        self, monkeypatch, capsys
    ) -> None:
        import threading
        import time

        from cognikernel.integration import hooks

        monkeypatch.setattr(hooks, "_HOOK_TIMEOUT_S", 0.1)
        monkeypatch.setattr(
            hooks, "_read_payload",
            lambda: {"cwd": "/proj", "prompt": "q", "session_id": "s"},
        )

        class _Cfg:
            query_time_injection = True

        monkeypatch.setattr("cognikernel.config.Config.load", lambda **k: _Cfg())

        release = threading.Event()

        def _stalled(*args, **kwargs):
            release.wait(5.0)  # would block far past the budget
            return "LATE — should never be printed"

        monkeypatch.setattr(
            "cognikernel.integration.query.recall_for_prompt", _stalled
        )

        start = time.monotonic()
        hooks.user_prompt_submit_main()
        elapsed = time.monotonic() - start
        release.set()  # let the abandoned daemon finish

        assert elapsed < 1.0, f"hook blocked {elapsed:.2f}s past its 0.1s budget"
        assert capsys.readouterr().out.strip() == ""  # timed out → silence


class TestUserPromptSubmitSilenceContract:
    def test_flag_off_exits_silently(self, tmp_path: Path) -> None:
        """With query_time_injection=False (default), the hook prints nothing."""
        proj = tmp_path / "proj"
        proj.mkdir()
        payload = json.dumps({
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(proj),
            "prompt": "which database should we use?",
        })
        env = {**os.environ, "COGNIKERNEL_DIR": str(tmp_path / "data")}
        r = subprocess.run(
            [sys.executable, "-m", "cognikernel", "hook-user-prompt"],
            input=payload, text=True, capture_output=True, timeout=30, env=env,
        )
        assert r.returncode == 0
        assert r.stdout.strip() == ""  # silence — flag is off

    def test_no_project_exits_silently(self, tmp_path: Path) -> None:
        """Missing project DB → silence, no crash, even with flag on."""
        proj = tmp_path / "no_such_project"
        payload = json.dumps({
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(proj),
            "prompt": "any question",
        })
        env = {**os.environ, "COGNIKERNEL_DIR": str(tmp_path / "data")}
        r = subprocess.run(
            [sys.executable, "-m", "cognikernel", "hook-user-prompt"],
            input=payload, text=True, capture_output=True, timeout=30, env=env,
        )
        assert r.returncode == 0
        assert r.stdout.strip() == ""

    def test_bad_payload_exits_silently(self) -> None:
        """Malformed stdin → silence, exit 0."""
        r = subprocess.run(
            [sys.executable, "-m", "cognikernel", "hook-user-prompt"],
            input="not json", text=True, capture_output=True, timeout=30,
        )
        assert r.returncode == 0

    def test_init_registers_user_prompt_hook(self, tmp_path: Path, monkeypatch) -> None:
        """CK-1 is default-on since the gamma evidence: imperative-update prompts
        don't reliably trigger the agent's pull path, and the block's section
        budgets can't carry the full decision surface by design. The hook is
        fail-open and silent when nothing clears the relevance gate; the
        query_time_injection flag (also default-on in the project template)
        still gates the work."""
        import argparse
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
        proj = tmp_path / "proj"
        proj.mkdir()
        cli._cmd_init(argparse.Namespace(project_path=str(proj)))
        settings = json.loads((proj / ".claude" / "settings.json").read_text())
        assert "UserPromptSubmit" in settings.get("hooks", {})
        cfg = (proj / ".cognikernel" / "config.toml").read_text()
        assert "query_time_injection = true" in cfg

    def test_init_registers_subagent_stop(self, tmp_path: Path, monkeypatch) -> None:
        """SubagentStop IS registered by init (capture_subagents default True)."""
        import argparse
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
        proj = tmp_path / "proj"
        proj.mkdir()
        cli._cmd_init(argparse.Namespace(project_path=str(proj)))
        settings = json.loads((proj / ".claude" / "settings.json").read_text())
        assert "SubagentStop" in settings.get("hooks", {})
        cmds = [e["hooks"][0]["command"]
                for e in settings["hooks"]["SubagentStop"]]
        assert any("hook-subagent-stop" in c for c in cmds)

    def test_init_registers_posttool_grep(self, tmp_path: Path, monkeypatch) -> None:
        """PostToolUse:Grep IS registered by init."""
        import argparse
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
        proj = tmp_path / "proj"
        proj.mkdir()
        cli._cmd_init(argparse.Namespace(project_path=str(proj)))
        settings = json.loads((proj / ".claude" / "settings.json").read_text())
        matchers = {e["matcher"] for e in settings["hooks"]["PostToolUse"]}
        assert "Grep" in matchers
