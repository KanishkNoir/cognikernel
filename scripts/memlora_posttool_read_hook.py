"""Claude Code PostToolUse hook — records successful Reads in read_session_cache.

Fires only after a Read tool call succeeds (per Anthropic hooks docs verified
during C0). Registered as a PostToolUse hook matching "Read" by `memlora init`.

The hook does not block (PostToolUse cannot affect tool outcome). Its job is
to populate read_session_cache so the next PreToolUse:Read attempt for the
same file in the same session can be denied as a re-read.

Outcome resolution: PreToolUse may have allowed this read as a
`body_needed_retry` (the 60s escape hatch). We detect that by checking
denied_reads — if a recent denial row exists for this (project, session, file),
the read is recorded with outcome='body_needed_retry'; otherwise 'ok'.

Never raises — any exception falls through silently so PostToolUse can't break
the user's Read.
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
        return

    if payload.get("tool_name") != "Read":
        return

    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd", "")

    if not file_path or not session_id:
        return

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

        from memlora.config import Config
        from memlora.integration.lookup import resolve_post_read_outcome
        from memlora.storage import read_cache as rc
        from memlora.storage.connection import (
            get_connection,
            get_db_path,
            hash_project_path,
        )
        from memlora.storage.migrations import run_migrations
        from memlora.utils.paths import canonicalize_path

        project_root = _find_project_root(Path(file_path))
        if project_root is None:
            project_root = Path(cwd) if cwd else Path(file_path).parent

        config = Config.load(project_path=project_root)
        project_id = hash_project_path(str(project_root))
        db_path = get_db_path(config, project_id)

        if not db_path.exists():
            return

        canonical = canonicalize_path(file_path, str(project_root))
        if not canonical:
            return

        retry_window_ms = config.deny_retry_window_seconds * 1000

        with get_connection(db_path) as conn:
            run_migrations(conn)
            outcome = resolve_post_read_outcome(
                conn,
                project_id=project_id,
                session_id=session_id,
                canonical_path=canonical,
                retry_window_ms=retry_window_ms,
            )
            rc.record_read(conn, project_id, session_id, canonical, outcome=outcome)

    except Exception:
        traceback.print_exc(file=sys.stderr)


def _find_project_root(file_path: Path) -> Path | None:
    current = file_path.resolve().parent if file_path.is_absolute() else file_path.parent
    for _ in range(12):
        if (current / ".claude" / "settings.json").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass
