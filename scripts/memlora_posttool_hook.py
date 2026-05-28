"""PostToolUse hook — updates symbol graph + symbol_files after Write/Edit.

Fires after every successful Write or Edit. Delegates to
`apply_symbol_update(..., project_path=..., session_id=..., last_action=...)`
which keeps both symbol_nodes/edges AND symbol_files in lockstep (C1).

Never raises — any exception is swallowed so Claude is never blocked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    raw = sys.stdin.buffer.read()
    try:
        payload = json.loads(raw.decode("utf-8-sig", errors="replace"))
    except Exception:
        return

    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Write", "Edit"):
        return

    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    session_id = payload.get("session_id", "")
    if not file_path:
        return

    abs_path = Path(file_path).resolve()
    if not abs_path.exists():
        return  # deleted or never written — nothing to parse

    project_path = _find_project_root(abs_path)
    if project_path is None:
        return

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

        from memlora.config import Config
        from memlora.extraction.git_augment import FileChange
        from memlora.storage.connection import (
            get_connection,
            get_db_path,
            hash_project_path,
        )
        from memlora.storage.migrations import run_migrations
        from memlora.symbols.extractor import build_symbol_update
        from memlora.symbols.store import apply_symbol_update

        config = Config.load(project_path=project_path)
        project_id = hash_project_path(project_path)
        db_path = get_db_path(config, project_id)

        if not db_path.exists():
            return  # project not yet initialised — Stop hook will handle it

        rel_path = str(abs_path.relative_to(Path(project_path).resolve())).replace("\\", "/")
        changed_files = [FileChange(path=rel_path, change_type="modified", lines_changed=0)]

        update = build_symbol_update(project_id, str(project_path), changed_files)

        with get_connection(db_path) as conn:
            run_migrations(conn)
            apply_symbol_update(
                conn,
                update,
                project_path=str(project_path),
                session_id=session_id,
                last_action=tool_name,  # 'Write' or 'Edit'
            )

            if config.grep_cache_enabled:
                from memlora.storage.grep_cache import invalidate_project_cache
                invalidate_project_cache(conn, project_id, changed_path=rel_path)

    except Exception:
        pass  # posttool hook must never block Claude


def _find_project_root(file_path: Path) -> Path | None:
    """Walk upward from file_path to find a dir containing .claude/settings.json."""
    current = file_path.parent
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
