"""CK-6a — portable hook entrypoints (`python -m cognikernel hook-*`) + dispatch."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import pytest

import cognikernel.integration.cli as cli


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
    monkeypatch.setattr(f"cognikernel.integration.hooks.{fn}", lambda: called.append(fn))
    monkeypatch.setattr(sys, "argv", ["cognikernel", subcommand])
    cli.main()
    assert called == [fn]


def test_hook_entrypoints_cover_all_five() -> None:
    import cognikernel.integration.hooks as hooks
    for fn in cli._HOOK_ENTRYPOINTS.values():
        assert callable(getattr(hooks, fn))


def test_hook_path_does_not_import_session_stack() -> None:
    """Importing `cli` (what `python -m cognikernel hook-*` does) must NOT pull the
    session / extraction / symbol stack — keeps the per-Read hook light (CK-6a).
    Run in a fresh process so the check isn't polluted by other tests' imports."""
    code = (
        "import sys, cognikernel.integration.cli; "
        "heavy=[m for m in ('cognikernel.integration.session','cognikernel.extraction.trie',"
        "'cognikernel.extraction.pipeline','cognikernel.symbols.extractor') if m in sys.modules]; "
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=60)
    assert r.returncode == 0, f"hot path pulled heavy modules: {r.stdout!r} {r.stderr!r}"


def test_python_m_cognikernel_hook_runs_fail_open(tmp_path) -> None:
    """End-to-end: `python -m cognikernel hook-session-start` runs (exit 0) and emits
    nothing for an uninitialised project — proving the portable entrypoint works."""
    proj = tmp_path / "proj"
    proj.mkdir()
    payload = json.dumps({
        "source": "startup", "cwd": str(proj), "hook_event_name": "SessionStart",
    })
    env = {**os.environ, "COGNIKERNEL_DIR": str(tmp_path / "data")}
    r = subprocess.run(
        [sys.executable, "-m", "cognikernel", "hook-session-start"],
        input=payload, text=True, capture_output=True, timeout=60, env=env,
    )
    assert r.returncode == 0
    assert r.stdout.strip() == ""  # no project DB → nothing injected, no crash


def test_hook_pretool_denies_fresh_skeleton_read_e2e(tmp_path, monkeypatch) -> None:
    """The hot path end-to-end: `python -m cognikernel hook-pretool` DENIES a Read of a
    fresh+scanned+has-symbols file under strict mode — the deny output contract."""
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
    proj = tmp_path / "proj"
    proj.mkdir()
    # init writes .claude/settings.json (the project-root marker), config, DB.
    cli._cmd_init(argparse.Namespace(project_path=str(proj)))
    # init defaults to advisory; this test covers strict mode's deny contract, so
    # it opts in explicitly rather than depending on init's default.
    cfg = proj / ".cognikernel" / "config.toml"
    cfg.write_text(
        cfg.read_text(encoding="utf-8").replace(
            'hook_policy = "advisory"', 'hook_policy = "strict"'
        ),
        encoding="utf-8",
    )
    assert 'hook_policy = "strict"' in cfg.read_text(encoding="utf-8")

    (proj / "app").mkdir()
    target = proj / "app" / "main.py"
    target.write_text("def go():\n    return 1\n", encoding="utf-8")

    from cognikernel.config import Config
    from cognikernel.storage import symbol_files as sf
    from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path

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
        [sys.executable, "-m", "cognikernel", "hook-pretool"],
        input=payload, text=True, capture_output=True, timeout=60,
        env={**os.environ, "COGNIKERNEL_DIR": str(tmp_path / "data")},
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_pretool_partial_read_is_exempt_and_uncached(tmp_path, monkeypatch) -> None:
    """L7: a Read with offset/limit targets a slice of the file. It is exempt
    from the gate (the deny's 'content is in your context' premise is false for
    a slice) and PostToolUse must not record it — a recorded slice would deny
    the later read of the rest of a large file."""
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
    proj = tmp_path / "proj"
    proj.mkdir()
    cli._cmd_init(argparse.Namespace(project_path=str(proj)))

    (proj / "app").mkdir()
    target = proj / "app" / "main.py"
    target.write_text("def go():\n    return 1\n", encoding="utf-8")

    from cognikernel.config import Config
    from cognikernel.storage import symbol_files as sf
    from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path

    pid = hash_project_path(str(proj))
    db = get_db_path(Config.load(project_path=str(proj)), pid)
    with get_connection(db) as conn:
        # A fresh+scanned+has-symbols row — a FULL read would be denied here.
        sf.upsert(conn, pid, "app/main.py", freshness="fresh", scan_status="scanned",
                  symbol_count=5, refreshed_at=int(time.time() * 1000) + 5000)

    env = {**os.environ, "COGNIKERNEL_DIR": str(tmp_path / "data")}
    payload = json.dumps({
        "hook_event_name": "PreToolUse", "tool_name": "Read",
        "tool_input": {"file_path": str(target), "offset": 1, "limit": 100},
        "session_id": "sess-partial", "cwd": str(proj),
    })
    r = subprocess.run(
        [sys.executable, "-m", "cognikernel", "hook-pretool"],
        input=payload, text=True, capture_output=True, timeout=60, env=env,
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"

    # PostToolUse:Read on the slice must not populate the read cache.
    post_payload = json.dumps({
        "hook_event_name": "PostToolUse", "tool_name": "Read",
        "tool_input": {"file_path": str(target), "offset": 1, "limit": 100},
        "session_id": "sess-partial", "cwd": str(proj),
    })
    r = subprocess.run(
        [sys.executable, "-m", "cognikernel", "hook-posttool-read"],
        input=post_payload, text=True, capture_output=True, timeout=60, env=env,
    )
    assert r.returncode == 0, r.stderr
    with get_connection(db) as conn:
        cached = conn.execute("SELECT COUNT(*) FROM read_session_cache").fetchone()[0]
    assert cached == 0


def test_hook_pretool_write_surfaces_prohibition_e2e(tmp_path, monkeypatch) -> None:
    """K2 end-to-end: `hook-pretool` on a Write that reintroduces a graveyarded
    approach ALLOWS but attaches the prohibition as additionalContext — the JIT
    bind. It must never deny a Write."""
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
    proj = tmp_path / "proj"
    proj.mkdir()
    cli._cmd_init(argparse.Namespace(project_path=str(proj)))

    from cognikernel.config import Config
    from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path

    pid = hash_project_path(str(proj))
    db = get_db_path(Config.load(project_path=str(proj)), pid)
    with get_connection(db) as conn:
        conn.execute(
            "INSERT INTO events (project_id, session_id, created_at, event_type, "
            "payload, content_hash, weight, mention_count) VALUES (?,?,1,?,?,?,1.0,1)",
            (pid, "s", "APPROACH_ABANDONED_DO_NOT_RETRY",
             json.dumps({"description": "do not use in-process rate limit counters; "
                                        "use Redis for the shared gateway budget",
                         "subject": "rate limiting"}), "gh1"),
        )
        conn.commit()

    target = proj / "gateway.py"
    payload = json.dumps({
        "hook_event_name": "PreToolUse", "tool_name": "Write",
        "tool_input": {"file_path": str(target),
                       "content": "self._counter = 0  # in-process rate limit "
                                  "counter for the gateway budget\n"},
        "session_id": "sess-k2", "cwd": str(proj),
    })
    r = subprocess.run(
        [sys.executable, "-m", "cognikernel", "hook-pretool"],
        input=payload, text=True, capture_output=True, timeout=60,
        env={**os.environ, "COGNIKERNEL_DIR": str(tmp_path / "data")},
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "allow"  # never blocks a Write
    assert "Redis" in hso.get("additionalContext", "")


def test_hook_posttool_write_records_write_session_cache_e2e(
    tmp_path, monkeypatch,
) -> None:
    """#31 Commit A: a real PostToolUse:Write populates write_session_cache
    (collection only — nothing reads this table for ranking yet)."""
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
    proj = tmp_path / "proj"
    proj.mkdir()
    cli._cmd_init(argparse.Namespace(project_path=str(proj)))

    target = proj / "gateway.py"
    target.write_text("def go():\n    return 1\n", encoding="utf-8")

    env = {**os.environ, "COGNIKERNEL_DIR": str(tmp_path / "data")}
    payload = json.dumps({
        "hook_event_name": "PostToolUse", "tool_name": "Write",
        "tool_input": {"file_path": str(target)},
        "session_id": "sess-write", "cwd": str(proj),
    })
    r = subprocess.run(
        [sys.executable, "-m", "cognikernel", "hook-posttool"],
        input=payload, text=True, capture_output=True, timeout=60, env=env,
    )
    assert r.returncode == 0, r.stderr

    from cognikernel.config import Config
    from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
    from cognikernel.storage.write_cache import get_write

    pid = hash_project_path(str(proj))
    db = get_db_path(Config.load(project_path=str(proj)), pid)
    with get_connection(db) as conn:
        entry = get_write(conn, pid, "sess-write", "gateway.py")
    assert entry is not None
    assert entry.last_write_action == "Write"
    assert entry.write_count == 1


def test_hook_posttool_multiedit_updates_symbol_graph_and_write_cache_e2e(
    tmp_path, monkeypatch,
) -> None:
    """MultiEdit was previously invisible to PostToolUse entirely (the tool_name
    gate excluded it) — the symbol graph never refreshed and no write signal was
    recorded. Both must now fire, same as Write/Edit."""
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
    proj = tmp_path / "proj"
    proj.mkdir()
    cli._cmd_init(argparse.Namespace(project_path=str(proj)))

    target = proj / "api.py"
    target.write_text("def handler():\n    return 1\n", encoding="utf-8")

    env = {**os.environ, "COGNIKERNEL_DIR": str(tmp_path / "data")}
    payload = json.dumps({
        "hook_event_name": "PostToolUse", "tool_name": "MultiEdit",
        "tool_input": {"file_path": str(target), "edits": [
            {"old_string": "return 1", "new_string": "return 2"},
        ]},
        "session_id": "sess-multiedit", "cwd": str(proj),
    })
    r = subprocess.run(
        [sys.executable, "-m", "cognikernel", "hook-posttool"],
        input=payload, text=True, capture_output=True, timeout=60, env=env,
    )
    assert r.returncode == 0, r.stderr

    from cognikernel.config import Config
    from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
    from cognikernel.storage.write_cache import get_write

    pid = hash_project_path(str(proj))
    db = get_db_path(Config.load(project_path=str(proj)), pid)
    with get_connection(db) as conn:
        entry = get_write(conn, pid, "sess-multiedit", "api.py")
        symbol_row = conn.execute(
            "SELECT scan_status FROM symbol_files WHERE project_id=? AND path=?",
            (pid, "api.py"),
        ).fetchone()
    assert entry is not None
    assert entry.last_write_action == "MultiEdit"
    assert symbol_row is not None  # symbol graph was refreshed for this file


# ── T-103 (#13): captured_at_sha ──────────────────────────────────────────────

def _git(args: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30,
    )


class TestCaptureHeadSha:
    """`_capture_head_sha` is the whole of T-103's git-anchoring logic,
    extracted from stop_main so it's testable without spawning the
    capture/process-jobs subprocess chain stop_main itself spawns. Three
    acceptance criteria from #13 map directly to the three tests here."""

    def test_real_repo_with_a_commit_returns_the_sha(self, tmp_path) -> None:
        from cognikernel.integration.hooks import _capture_head_sha

        repo = tmp_path / "repo"
        repo.mkdir()
        _git(["init", "-q"], str(repo))
        _git(["config", "user.email", "t@example.com"], str(repo))
        _git(["config", "user.name", "T"], str(repo))
        (repo / "f.txt").write_text("x", encoding="utf-8")
        _git(["add", "."], str(repo))
        commit = _git(["commit", "-q", "-m", "init"], str(repo))
        assert commit.returncode == 0, commit.stderr
        expected = _git(["rev-parse", "HEAD"], str(repo)).stdout.strip()

        sha = _capture_head_sha(str(repo))
        assert sha == expected
        assert len(sha) == 40  # a full sha, not an abbreviation

    def test_not_a_git_work_tree_returns_empty_and_does_not_raise(self, tmp_path) -> None:
        """Acceptance: 'a capture outside a git work tree produces NULL and
        nothing raises' — verified here, not by inspection."""
        from cognikernel.integration.hooks import _capture_head_sha

        plain_dir = tmp_path / "not_a_repo"
        plain_dir.mkdir()
        assert _capture_head_sha(str(plain_dir)) == ""

    def test_repo_with_zero_commits_returns_empty_and_does_not_raise(self, tmp_path) -> None:
        """Acceptance: 'a capture in a repo with zero commits produces NULL
        and nothing raises' — `git rev-parse HEAD` fails with 'ambiguous
        argument HEAD' on an empty repo; that must not propagate."""
        from cognikernel.integration.hooks import _capture_head_sha

        repo = tmp_path / "empty_repo"
        repo.mkdir()
        init = _git(["init", "-q"], str(repo))
        assert init.returncode == 0, init.stderr

        assert _capture_head_sha(str(repo)) == ""

    def test_timeout_warns_and_returns_empty_without_raising(self, monkeypatch, tmp_path) -> None:
        """Acceptance: 'git rev-parse failure/timeout does not delay or fail
        the Stop hook' + 'the rev-parse failure branch logs at WARNING'."""
        import cognikernel.integration.hooks as hooks_mod

        def _raise_timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="git", timeout=10)

        monkeypatch.setattr(hooks_mod.subprocess, "run", _raise_timeout)
        warnings: list[str] = []
        monkeypatch.setattr(hooks_mod, "_warn", lambda msg: warnings.append(msg))

        sha = hooks_mod._capture_head_sha(str(tmp_path))

        assert sha == ""
        assert len(warnings) == 1
        assert "rev-parse" in warnings[0]

    def test_nonzero_exit_is_silent_not_a_warning(self, monkeypatch, tmp_path) -> None:
        """A non-zero exit (no repo / no commits) is the ORDINARY case and
        must stay silent — only a genuine exception (git missing, timeout)
        warns. Conflating the two would make every fresh non-git project
        noisy on every session end."""
        import cognikernel.integration.hooks as hooks_mod

        class _FakeResult:
            returncode = 128
            stdout = ""

        monkeypatch.setattr(hooks_mod.subprocess, "run", lambda *a, **k: _FakeResult())
        warnings: list[str] = []
        monkeypatch.setattr(hooks_mod, "_warn", lambda msg: warnings.append(msg))

        assert hooks_mod._capture_head_sha(str(tmp_path)) == ""
        assert warnings == []


class TestStopMainPassesHeadSha:
    def test_stop_main_includes_head_sha_flag_when_available(
        self, monkeypatch, tmp_path
    ) -> None:
        """stop_main must thread a resolved sha into the capture subprocess
        command as --head-sha, and must NOT pass the flag at all when no sha
        is available (an empty --head-sha "" would be a worse signal than
        omitting the flag — argparse's own default=None already means 'no
        sha' cleanly)."""
        import cognikernel.integration.hooks as hooks_mod

        jsonl_dir = tmp_path / ".claude" / "projects" / "proj"
        jsonl_dir.mkdir(parents=True)
        session_id = "sess-headsha"
        (jsonl_dir / f"{session_id}.jsonl").write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(hooks_mod.Path, "home", lambda: tmp_path)

        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(
            hooks_mod, "_read_payload",
            lambda: {"session_id": session_id, "cwd": str(project_dir)},
        )
        monkeypatch.setattr(hooks_mod, "_capture_head_sha", lambda project_dir: "deadbeef" * 5)

        captured_cmds: list[list[str]] = []

        class _FakeCompleted:
            returncode = 0
            stdout = ""
            stderr = ""

        def _fake_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            return _FakeCompleted()

        monkeypatch.setattr(hooks_mod.subprocess, "run", _fake_run)

        hooks_mod.stop_main()

        capture_cmds = [c for c in captured_cmds if "capture" in c]
        assert capture_cmds, f"no capture subprocess launched: {captured_cmds}"
        assert "--head-sha" in capture_cmds[0]
        idx = capture_cmds[0].index("--head-sha")
        assert capture_cmds[0][idx + 1] == "deadbeef" * 5

    def test_stop_main_omits_head_sha_flag_when_unavailable(
        self, monkeypatch, tmp_path
    ) -> None:
        import cognikernel.integration.hooks as hooks_mod

        jsonl_dir = tmp_path / ".claude" / "projects" / "proj"
        jsonl_dir.mkdir(parents=True)
        session_id = "sess-nosha"
        (jsonl_dir / f"{session_id}.jsonl").write_text("{}\n", encoding="utf-8")
        monkeypatch.setattr(hooks_mod.Path, "home", lambda: tmp_path)

        project_dir = tmp_path / "proj"
        project_dir.mkdir()

        monkeypatch.setattr(
            hooks_mod, "_read_payload",
            lambda: {"session_id": session_id, "cwd": str(project_dir)},
        )
        monkeypatch.setattr(hooks_mod, "_capture_head_sha", lambda project_dir: "")

        captured_cmds: list[list[str]] = []

        class _FakeCompleted:
            returncode = 0
            stdout = ""
            stderr = ""

        def _fake_run(cmd, **kwargs):
            captured_cmds.append(cmd)
            return _FakeCompleted()

        monkeypatch.setattr(hooks_mod.subprocess, "run", _fake_run)

        hooks_mod.stop_main()

        capture_cmds = [c for c in captured_cmds if "capture" in c]
        assert capture_cmds, f"no capture subprocess launched: {captured_cmds}"
        assert "--head-sha" not in capture_cmds[0]
