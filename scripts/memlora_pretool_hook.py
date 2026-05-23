"""Claude Code PreToolUse hook — cached context for Read and Grep deduplication.

Registered in .claude/settings.json as a PreToolUse hook matching "Read".

For Read calls:
  - Looks up the file path in MemLoRA's component map via `memlora lookup`
  - Denies with the cached component description if the file is known
  - Allows the read otherwise

  Current limitation: the component_map is populated only from COMPONENT_UPDATE events
  extracted by the Stop hook. In practice most code files are not tagged with
  COMPONENT_UPDATE events, so the map is sparse and most reads fall through to allow.

  Once the Symbol Graph layer (PostToolUse hook) is deployed, every Write/Edit will
  populate symbol_nodes for that file. The Read lookup path can then be extended to
  serve symbol-node summaries in place of the full file read. Until then, this hook
  provides correct behaviour but minimal interception.

For Grep calls (requires grep_cache_enabled = true in config):
  - Looks up the (pattern, path, glob) triple in the grep_cache table
  - Denies with the cached result if found
  - Allows the grep otherwise

Exits 0 on all paths — never blocks session flow.
"""
from __future__ import annotations

import json
import subprocess
import sys


def main() -> None:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8-sig")  # strips BOM if present
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


def _handle_read(payload: dict) -> None:
    tool_input = payload.get("tool_input", {})
    cwd = payload.get("cwd", "")
    file_path = tool_input.get("file_path", "")

    if not file_path or not cwd:
        _allow()
        return

    try:
        result = subprocess.run(
            [sys.executable, "-m", "memlora", "lookup", cwd, file_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except Exception:
        _allow()
        return

    if result.returncode == 0 and result.stdout.strip():
        _deny(result.stdout.strip())
    elif result.returncode == 3 and result.stdout.strip():
        _allow_with_context(result.stdout.strip())
    else:
        _allow()


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
        from memlora.config import Config
        from memlora.storage.connection import get_connection, get_db_path, hash_project_path
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
            reason = f"[grep-cache] Pattern `{pattern}` matched {path_filter or '(all)'} — cached result:\n\n{cached}"
            _deny(reason)
        else:
            _allow()
    except Exception:
        _allow()


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
