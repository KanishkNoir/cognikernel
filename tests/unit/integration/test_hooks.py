"""CK-6a — portable hook entrypoints (`python -m memlora hook-*`) + dispatch."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import pytest

import memlora.integration.cli as cli


@pytest.mark.parametrize(
    "subcommand,fn",
    [
        ("hook-session-start", "session_start_main"),
        ("hook-stop", "stop_main"),
        ("hook-pretool", "pretool_main"),
        ("hook-posttool", "posttool_main"),
        ("hook-posttool-read", "posttool_read_main"),
    ],
)
def test_main_dispatches_each_hook(monkeypatch, subcommand: str, fn: str) -> None:
    """main() routes a hook subcommand straight to the matching hooks entrypoint,
    bypassing argparse and the heavy session import (hot-path fast dispatch)."""
    called: list[str] = []
    monkeypatch.setattr(f"memlora.integration.hooks.{fn}", lambda: called.append(fn))
    monkeypatch.setattr(sys, "argv", ["memlora", subcommand])
    cli.main()
    assert called == [fn]


def test_hook_entrypoints_cover_all_five() -> None:
    import memlora.integration.hooks as hooks
    for fn in cli._HOOK_ENTRYPOINTS.values():
        assert callable(getattr(hooks, fn))


def test_hook_path_does_not_import_session_stack() -> None:
    """Importing `cli` (what `python -m memlora hook-*` does) must NOT pull the
    session / extraction / symbol stack — keeps the per-Read hook light (CK-6a).
    Run in a fresh process so the check isn't polluted by other tests' imports."""
    code = (
        "import sys, memlora.integration.cli; "
        "heavy=[m for m in ('memlora.integration.session','memlora.extraction.trie',"
        "'memlora.extraction.pipeline','memlora.symbols.extractor') if m in sys.modules]; "
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60)
    assert r.returncode == 0, f"hot path pulled heavy modules: {r.stdout!r} {r.stderr!r}"


def test_python_m_memlora_hook_runs_fail_open(tmp_path) -> None:
    """End-to-end: `python -m memlora hook-session-start` runs (exit 0) and emits
    nothing for an uninitialised project — proving the portable entrypoint works."""
    proj = tmp_path / "proj"
    proj.mkdir()
    payload = json.dumps({
        "source": "startup", "cwd": str(proj), "hook_event_name": "SessionStart",
    })
    env = {**os.environ, "MEMLORA_DIR": str(tmp_path / "data")}
    r = subprocess.run(
        [sys.executable, "-m", "memlora", "hook-session-start"],
        input=payload, text=True, capture_output=True, timeout=60, env=env,
    )
    assert r.returncode == 0
    assert r.stdout.strip() == ""  # no project DB → nothing injected, no crash


def test_hook_pretool_denies_fresh_skeleton_read_e2e(tmp_path, monkeypatch) -> None:
    """The hot path end-to-end: `python -m memlora hook-pretool` DENIES a Read of a
    fresh+scanned+has-symbols file under strict mode — the deny output contract."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "data"))
    proj = tmp_path / "proj"
    proj.mkdir()
    # init writes .claude/settings.json (the project-root marker), strict config, DB.
    cli._cmd_init(argparse.Namespace(project_path=str(proj)))

    (proj / "app").mkdir()
    target = proj / "app" / "main.py"
    target.write_text("def go():\n    return 1\n", encoding="utf-8")

    from memlora.config import Config
    from memlora.storage import symbol_files as sf
    from memlora.storage.connection import get_connection, get_db_path, hash_project_path

    pid = hash_project_path(str(proj))
    db = get_db_path(Config.load(project_path=str(proj)), pid)
    with get_connection(db) as conn:
        # refreshed AFTER the file's mtime so the freshness check trusts the skeleton.
        sf.upsert(conn, pid, "app/main.py", freshness="fresh", scan_status="scanned",
                  symbol_count=5, refreshed_at=int(time.time() * 1000) + 5000)

    payload = json.dumps({
        "hook_event_name": "PreToolUse", "tool_name": "Read",
        "tool_input": {"file_path": str(target)},
        "session_id": "sess-e2e", "cwd": str(proj),
    })
    r = subprocess.run(
        [sys.executable, "-m", "memlora", "hook-pretool"],
        input=payload, text=True, capture_output=True, timeout=60,
        env={**os.environ, "MEMLORA_DIR": str(tmp_path / "data")},
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
