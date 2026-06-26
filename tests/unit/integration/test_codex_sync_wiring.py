"""L3 wiring: CLI `codex-sync` + SessionStart auto-sync (Sprint L)."""
from __future__ import annotations

import argparse
import io
import json

import memlora.integration.codex_sync as codex_sync
import memlora.integration.hooks as hooks
from memlora.integration.cli import _cmd_codex_sync


class TestCodexSyncCLI:
    def test_cmd_invokes_sync_and_prints_stats(self, monkeypatch, capsys):
        calls = {}

        def fake_sync(project_path, *, scan_window_days=None, spawn_worker=False):
            calls.update(project=project_path, scan=scan_window_days, spawn=spawn_worker)
            return {"scanned": 3, "matched": 1, "captured": 1, "jobs": 1, "enabled": True}

        monkeypatch.setattr(codex_sync, "sync_codex_rollouts", fake_sync)
        _cmd_codex_sync(argparse.Namespace(project_path="/p", scan_days=7, no_drain=False))

        out = json.loads(capsys.readouterr().out)
        assert out["captured"] == 1
        assert calls == {"project": "/p", "scan": 7, "spawn": True}

    def test_no_drain_disables_spawn(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(codex_sync, "sync_codex_rollouts",
                            lambda p, **k: seen.update(k) or {"jobs": 0})
        _cmd_codex_sync(argparse.Namespace(project_path="/p", scan_days=None, no_drain=True))
        assert seen["spawn_worker"] is False


class TestSessionStartAutoSync:
    def _run(self, monkeypatch, payload):
        monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
        hooks.session_start_main()

    def test_startup_triggers_codex_sync(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(codex_sync, "sync_codex_rollouts",
                            lambda cwd, *a, **k: seen.update(cwd=cwd) or {"jobs": 0})
        # no project DB -> handle_session_start returns "" after sync; that's fine.
        self._run(monkeypatch, {"source": "startup", "cwd": "/proj", "session_id": "s1"})
        assert seen.get("cwd") == "/proj"

    def test_sync_failure_never_breaks_session_start(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("codex exploded")

        monkeypatch.setattr(codex_sync, "sync_codex_rollouts", boom)
        # Must not raise — fail-open swallow.
        self._run(monkeypatch, {"source": "startup", "cwd": "/proj"})

    def test_non_reinject_source_skips(self, monkeypatch):
        called = {"n": 0}
        monkeypatch.setattr(codex_sync, "sync_codex_rollouts",
                            lambda *a, **k: called.update(n=called["n"] + 1) or {})
        self._run(monkeypatch, {"source": "other", "cwd": "/proj"})
        assert called["n"] == 0  # gated before sync
