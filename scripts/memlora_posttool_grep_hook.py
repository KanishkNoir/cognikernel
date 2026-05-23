"""Claude Code PostToolUse hook — caches Grep results for deduplication.

Fires after every Grep tool call. Stores the result in the grep_cache table
so the PreToolUse hook can serve repeated identical greps from cache.

Enabled only when grep_cache_enabled = true in memlora config.
Exits 0 on all paths — never blocks session flow.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    try:
        raw = sys.stdin.buffer.read().decode("utf-8-sig")
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        return

    if payload.get("tool_name") != "Grep":
        return

    tool_input = payload.get("tool_input", {})
    pattern = tool_input.get("pattern", "")
    path_filter = tool_input.get("path", "") or ""
    glob_filter = tool_input.get("glob", "") or ""
    cwd = payload.get("cwd", "")

    if not pattern or not cwd:
        return

    # Claude Code sends the tool result as "tool_response"
    result_text = payload.get("tool_response", "") or ""
    if not isinstance(result_text, str):
        result_text = json.dumps(result_text)

    try:
        from memlora.config import Config
        from memlora.storage.connection import get_connection, get_db_path, hash_project_path
        from memlora.storage.grep_cache import store_grep_result
        from memlora.storage.migrations import run_migrations

        cfg = Config.load()
        if not cfg.grep_cache_enabled:
            return

        project_id = hash_project_path(cwd)
        db_path = get_db_path(cfg, project_id)
        if not db_path.exists():
            return

        with get_connection(db_path) as conn:
            run_migrations(conn)
            store_grep_result(conn, project_id, pattern, path_filter, glob_filter, result_text)
    except Exception:
        pass  # hook must never crash Claude


if __name__ == "__main__":
    main()
