"""End-to-end subprocess test for PreToolUse + PostToolUse:Read (Stage C1).

Spawns the hook scripts as Claude Code would and validates the full
deny → retry → cache-write → re-read-deny lifecycle.

This is the canonical regression test for the Arm C 3× main.py re-read bug.
A passing test here proves the gate would have prevented that pattern.
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
from memlora.storage import symbol_files as sf
from memlora.storage.connection import get_connection, get_db_path, hash_project_path

PRETOOL = (
    Path(__file__).parent.parent.parent / "scripts" / "memlora_pretool_hook.py"
)
POSTTOOL_READ = (
    Path(__file__).parent.parent.parent / "scripts" / "memlora_posttool_read_hook.py"
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_project(tmp_path: Path) -> tuple[Path, Path]:
    """Set up a tmp project with the minimal scaffolding hooks expect."""
    memlora_dir = tmp_path / "memlora"
    project_path = tmp_path / "proj"
    project_path.mkdir()

    # Hooks walk up looking for `.claude/settings.json` to find the project root.
    claude = project_path / ".claude"
    claude.mkdir()
    (claude / "settings.json").write_text("{}", encoding="utf-8")

    # Force strict mode at the project layer (mirrors what `memlora init` writes).
    memlora_cfg_dir = project_path / ".memlora"
    memlora_cfg_dir.mkdir()
    (memlora_cfg_dir / "config.toml").write_text(
        'hook_policy = "strict"\n', encoding="utf-8"
    )

    cfg = Config(memlora_dir=memlora_dir)
    init_project(project_path, config=cfg)

    return memlora_dir, project_path


def _seed_symbol_files(memlora_dir: Path, project_path: Path, rel: str) -> None:
    """Seed a fresh+scanned+symbols-present row so STEP 2 Case A is reachable."""
    cfg = Config(memlora_dir=memlora_dir)
    project_id = hash_project_path(str(project_path))
    db_path = get_db_path(cfg, project_id)
    with get_connection(db_path) as conn:
        # symbol_files needs a row indicating dense skeleton coverage.
        sf.upsert(
            conn, project_id, rel,
            freshness="fresh", scan_status="scanned", symbol_count=5,
        )
        # Touch the file on disk so canonicalization can resolve() it.
        target = project_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder", encoding="utf-8")


def _run_pretool(payload: dict, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(PRETOOL)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def _run_posttool_read(payload: dict, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(POSTTOOL_READ)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def _build_env(memlora_dir: Path) -> dict:
    return {**os.environ, "MEMLORA_DIR": str(memlora_dir)}


# ── tests ────────────────────────────────────────────────────────────────────


class TestPreToolStrictMode:
    def test_skeleton_fresh_first_read_is_denied(self, tmp_path: Path) -> None:
        memlora_dir, project_path = _make_project(tmp_path)
        _seed_symbol_files(memlora_dir, project_path, "app/main.py")
        env = _build_env(memlora_dir)

        result = _run_pretool({
            "tool_name": "Read",
            "session_id": "sess-1",
            "cwd": str(project_path),
            "tool_input": {"file_path": str(project_path / "app/main.py")},
        }, env)

        assert result.returncode == 0
        out = json.loads(result.stdout)
        hook = out["hookSpecificOutput"]
        assert hook["permissionDecision"] == "deny"
        assert "Codebase skeleton" in hook["permissionDecisionReason"]

    def test_retry_within_window_is_allowed_with_body_needed_context(
        self, tmp_path: Path,
    ) -> None:
        memlora_dir, project_path = _make_project(tmp_path)
        _seed_symbol_files(memlora_dir, project_path, "app/main.py")
        env = _build_env(memlora_dir)
        payload = {
            "tool_name": "Read",
            "session_id": "sess-1",
            "cwd": str(project_path),
            "tool_input": {"file_path": str(project_path / "app/main.py")},
        }

        # First attempt — denied.
        r1 = _run_pretool(payload, env)
        out1 = json.loads(r1.stdout)
        assert out1["hookSpecificOutput"]["permissionDecision"] == "deny"

        # Immediate retry — allowed (still well within 60s).
        r2 = _run_pretool(payload, env)
        out2 = json.loads(r2.stdout)
        assert out2["hookSpecificOutput"]["permissionDecision"] == "allow"
        # The "body-needed retry" hint flows as additionalContext.
        ctx = out2["hookSpecificOutput"].get("additionalContext", "")
        assert "body-needed retry" in ctx

    def test_full_lifecycle_blocks_3x_reread(self, tmp_path: Path) -> None:
        """The Arm C regression case: same file Read 3 times in one session.

        Expected:
          1. First Read   — denied (skeleton has it)
          2. Retry        — allowed (body-needed retry within 60s)
          3. PostToolUse  — records cache as 'body_needed_retry'
          4. Third Read   — denied (already read this session, even via body_retry)
        """
        memlora_dir, project_path = _make_project(tmp_path)
        _seed_symbol_files(memlora_dir, project_path, "app/main.py")
        env = _build_env(memlora_dir)

        file_abs = str(project_path / "app/main.py")
        payload = {
            "tool_name": "Read",
            "session_id": "sess-1",
            "cwd": str(project_path),
            "tool_input": {"file_path": file_abs},
        }

        # 1. First denial — skeleton-fresh.
        r1 = _run_pretool(payload, env)
        d1 = json.loads(r1.stdout)["hookSpecificOutput"]
        assert d1["permissionDecision"] == "deny"
        assert "Codebase skeleton" in d1["permissionDecisionReason"]

        # 2. Retry within 60s — allowed (body_needed_retry).
        r2 = _run_pretool(payload, env)
        d2 = json.loads(r2.stdout)["hookSpecificOutput"]
        assert d2["permissionDecision"] == "allow"

        # 3. PostToolUse:Read records the successful read in the cache.
        # In production Claude Code fires this automatically. We invoke it
        # directly to simulate that flow.
        _run_posttool_read(payload, env)

        # 4. Third attempt — must be denied as a re-read.
        r3 = _run_pretool(payload, env)
        d3 = json.loads(r3.stdout)["hookSpecificOutput"]
        assert d3["permissionDecision"] == "deny"
        msg = d3["permissionDecisionReason"]
        # Either "already read" (from the 'ok' branch — unlikely after retry)
        # or "body was already provided" (from the 'body_needed_retry' branch).
        assert "already" in msg.lower() or "previous read" in msg.lower(), (
            f"expected re-read denial, got: {msg!r}"
        )

    def test_universal_reread_block_for_plain_ok_read(self, tmp_path: Path) -> None:
        """A non-skeleton file (no symbol_files row) still gets re-read blocked
        once it's been read once. Covers the v2 STEP 1 universal invariant."""
        memlora_dir, project_path = _make_project(tmp_path)
        env = _build_env(memlora_dir)

        # NO symbol_files seeding — file should fall to Case E (allow first time).
        file_abs = str(project_path / "novel.py")
        (project_path / "novel.py").write_text("hi", encoding="utf-8")
        payload = {
            "tool_name": "Read",
            "session_id": "sess-1",
            "cwd": str(project_path),
            "tool_input": {"file_path": file_abs},
        }

        # First Read — allowed (no symbol_files row → Case E).
        r1 = _run_pretool(payload, env)
        d1 = json.loads(r1.stdout)["hookSpecificOutput"]
        assert d1["permissionDecision"] == "allow"

        # Post records the read.
        _run_posttool_read(payload, env)

        # Second Read — denied as re-read (STEP 1).
        r2 = _run_pretool(payload, env)
        d2 = json.loads(r2.stdout)["hookSpecificOutput"]
        assert d2["permissionDecision"] == "deny"
        assert "already read" in d2["permissionDecisionReason"].lower()

    def test_other_tools_pass_through_untouched(self, tmp_path: Path) -> None:
        memlora_dir, project_path = _make_project(tmp_path)
        env = _build_env(memlora_dir)

        result = _run_pretool({
            "tool_name": "Bash",
            "session_id": "sess-1",
            "cwd": str(project_path),
            "tool_input": {"command": "echo hi"},
        }, env)

        assert result.returncode == 0
        out = json.loads(result.stdout)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"

    def test_uninitialised_project_falls_through_to_allow(
        self, tmp_path: Path,
    ) -> None:
        """If the project DB doesn't exist yet, hook must not block."""
        memlora_dir = tmp_path / "memlora"
        project_path = tmp_path / "untouched"
        project_path.mkdir()
        (project_path / ".claude").mkdir()
        (project_path / ".claude" / "settings.json").write_text("{}", encoding="utf-8")
        env = _build_env(memlora_dir)

        result = _run_pretool({
            "tool_name": "Read",
            "session_id": "sess-1",
            "cwd": str(project_path),
            "tool_input": {"file_path": str(project_path / "anything.py")},
        }, env)

        assert result.returncode == 0
        out = json.loads(result.stdout)
        assert out["hookSpecificOutput"]["permissionDecision"] == "allow"


class TestPostToolReadCacheWrite:
    def test_successful_read_writes_cache(self, tmp_path: Path) -> None:
        memlora_dir, project_path = _make_project(tmp_path)
        env = _build_env(memlora_dir)

        file_abs = str(project_path / "novel.py")
        (project_path / "novel.py").write_text("x", encoding="utf-8")

        _run_posttool_read({
            "tool_name": "Read",
            "session_id": "sess-1",
            "cwd": str(project_path),
            "tool_input": {"file_path": file_abs},
        }, env)

        # Query the DB directly to verify the row landed.
        cfg = Config(memlora_dir=memlora_dir)
        project_id = hash_project_path(str(project_path))
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            row = conn.execute(
                "SELECT file_path, read_count, last_read_outcome "
                "FROM read_session_cache WHERE project_id=? AND session_id=?",
                (project_id, "sess-1"),
            ).fetchone()
        assert row is not None
        assert row["file_path"] == "novel.py"
        assert row["read_count"] == 1
        assert row["last_read_outcome"] == "ok"

    def test_non_read_tool_is_ignored(self, tmp_path: Path) -> None:
        memlora_dir, project_path = _make_project(tmp_path)
        env = _build_env(memlora_dir)

        _run_posttool_read({
            "tool_name": "Bash",
            "session_id": "sess-1",
            "cwd": str(project_path),
            "tool_input": {"command": "echo hi"},
        }, env)

        cfg = Config(memlora_dir=memlora_dir)
        project_id = hash_project_path(str(project_path))
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            n = conn.execute("SELECT COUNT(*) FROM read_session_cache").fetchone()[0]
        assert n == 0
