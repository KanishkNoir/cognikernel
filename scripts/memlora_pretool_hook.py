"""Claude Code PreToolUse hook — strict-mode skeleton gate (Stage C1).

Registered as a PreToolUse hook matching "Read" by `memlora init`. Replaces the
v0 subprocess-based gate with in-process logic to cut ~80-150ms of Python-cold-
start cost per Read.

Decision tree lives in `memlora.integration.lookup.decide_pretool_read`. This
script just:
  1. Reads JSON payload from stdin (file_path, session_id, cwd).
  2. Resolves the project root, project_id, and config.
  3. Calls decide_pretool_read(), translates Decision → Claude Code JSON.

Never raises — any exception falls through to `_allow()` so a broken hook
never blocks Claude. Errors are logged to stderr for the hook event log.

Grep handling is a separate concern (see memlora_posttool_grep_hook.py).
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def main() -> None:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8-sig", errors="replace")
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        _allow()
        return

    tool_name = payload.get("tool_name", "")
    if tool_name == "Read":
        _handle_read(payload)
    elif tool_name == "Grep":
        _handle_grep(payload)
    else:
        _allow()


# ── Read handler ─────────────────────────────────────────────────────────────


def _handle_read(payload: dict) -> None:
    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd", "")

    if not file_path:
        _allow()
        return

    try:
        # Wire in the src/ tree so the hook can import memlora when invoked
        # from a project directory that doesn't have memlora on PYTHONPATH.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

        from memlora.config import Config
        from memlora.integration.lookup import decide_pretool_read
        from memlora.storage.connection import (
            get_connection,
            get_db_path,
            hash_project_path,
        )
        from memlora.storage.migrations import run_migrations

        project_root = _find_project_root(Path(file_path))
        # If we can't find a project root, the read isn't governed by us.
        if project_root is None:
            project_root = Path(cwd) if cwd else Path(file_path).parent

        config = Config.load(project_path=project_root)
        project_id = hash_project_path(str(project_root))
        db_path = get_db_path(config, project_id)

        if not db_path.exists():
            # Project not yet initialised — let the read through.
            _allow()
            return

        retry_window_ms = config.deny_retry_window_seconds * 1000

        with get_connection(db_path) as conn:
            run_migrations(conn)
            decision = decide_pretool_read(
                conn,
                project_id=project_id,
                session_id=session_id or "__unknown__",
                file_path=file_path,
                project_path=str(project_root),
                policy=config.hook_policy,
                retry_window_ms=retry_window_ms,
            )

        if decision.is_deny:
            _deny(decision.message)
        elif decision.outcome_hint == "body_needed_retry":
            # Tell Claude Code we're allowing but signal in context that this
            # was a retry — the PostToolUse:Read hook will detect via the
            # denied_reads timer and record the cache row accordingly.
            _allow_with_context(
                "[CogniKernel] body-needed retry granted — record this read in your "
                "context; the next attempt to re-read this file will be denied."
            )
        else:
            _allow()

    except Exception:
        traceback.print_exc(file=sys.stderr)
        _allow()


# ── Grep handler (unchanged from v0; keeps grep_cache integration) ───────────


def _handle_grep(payload: dict) -> None:
    tool_input = payload.get("tool_input", {})
    cwd = payload.get("cwd", "")
    pattern = tool_input.get("pattern", "")

    if not pattern or not cwd:
        _allow()
        return

    path_filter = tool_input.get("path", "") or ""
    glob_filter = tool_input.get("glob", "") or ""

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

        from memlora.config import Config
        from memlora.storage.connection import (
            get_connection,
            get_db_path,
            hash_project_path,
        )
        from memlora.storage.grep_cache import lookup_grep_result
        from memlora.storage.migrations import run_migrations

        cfg = Config.load()
        if not cfg.grep_cache_enabled:
            _allow()
            return

        project_id = hash_project_path(cwd)
        db_path = get_db_path(cfg, project_id)
        if not db_path.exists():
            _allow()
            return

        with get_connection(db_path) as conn:
            run_migrations(conn)
            cached = lookup_grep_result(conn, project_id, pattern, path_filter, glob_filter)

        if cached is not None:
            reason = (
                f"[CogniKernel grep-cache] Pattern `{pattern}` matched "
                f"{path_filter or '(all)'} — cached result:\n\n{cached}"
            )
            _deny(reason)
        else:
            _allow()
    except Exception:
        traceback.print_exc(file=sys.stderr)
        _allow()


# ── project root discovery ───────────────────────────────────────────────────


def _find_project_root(file_path: Path) -> Path | None:
    """Walk upward from file_path to find a dir containing .claude/settings.json."""
    current = file_path.resolve().parent if file_path.is_absolute() else file_path.parent
    for _ in range(12):
        if (current / ".claude" / "settings.json").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


# ── Claude Code hook protocol helpers ────────────────────────────────────────


def _allow() -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }))


def _allow_with_context(context: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "additionalContext": context,
        }
    }))


def _deny(reason: str) -> None:
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))


if __name__ == "__main__":
    main()
