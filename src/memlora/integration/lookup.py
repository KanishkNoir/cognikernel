"""File-path lookup against the component map projection and symbol graph.

Used by the `memlora lookup` CLI subcommand and the PreToolUse hook script.
"""
from __future__ import annotations

from pathlib import Path


_ALLOW_STATUSES = frozenset({"modified", "in_flux", "added", "deleted"})


def lookup_file(
    project_path: str,
    file_path: str,
    config=None,
) -> tuple[int, str]:
    """Look up a file path in the project's component map and symbol graph.

    Returns:
      (0, message) — found in component map, status "referenced"; hook should deny with message
      (1, "")      — not found anywhere or DB absent; hook should allow with no context
      (2, "")      — found in component map but status is modified/in_flux; hook should allow
      (3, message) — not in component map but symbol nodes exist; hook should allow WITH context
    """
    from memlora.config import Config
    from memlora.storage.connection import get_connection, get_db_path, hash_project_path
    from memlora.storage.projections import load_or_rebuild

    config = config or Config.load()
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)

    if not db_path.exists():
        return 1, ""

    with get_connection(db_path) as conn:
        projection = load_or_rebuild(conn, project_id)
        rec = _find_in_map(projection.component_map, file_path, project_path)

        if rec is None:
            # Not in component map — check if symbol graph has data for this file
            context_msg = _symbol_context(conn, project_id, file_path, project_path)
            if context_msg:
                return 3, context_msg
            return 1, ""

    payload = rec.get("payload", {})
    status = payload.get("status", "referenced")

    if status in _ALLOW_STATUSES:
        return 2, ""

    path = payload.get("path", file_path)
    intent = payload.get("intent", "")
    session_id = rec.get("session_id", "?")
    session_short = session_id[:12] if len(session_id) >= 12 else session_id

    intent_part = f" · {intent}" if intent and intent != path else ""
    message = (
        f"MemLoRA: {path} · session {session_short}{intent_part} · status: {status}"
        f" — Re-issue Read if you need current file content."
    )
    return 0, message


def _symbol_context(conn, project_id: str, file_path: str, project_path: str) -> str:
    """Return a skeleton-pointer message if symbol nodes exist for this file, else ''."""
    rel_path = _to_rel_path(file_path, project_path)
    if not rel_path:
        return ""
    row = conn.execute(
        "SELECT COUNT(*) FROM symbol_nodes WHERE project_id = ? AND path = ?",
        (project_id, rel_path),
    ).fetchone()
    if not row or row[0] == 0:
        return ""
    return (
        f"[skeleton] {rel_path} structure is in the § Codebase skeleton section of your "
        f"session context — classes, methods, and signatures are listed there. "
        f"Read this file only if you need the full implementation body."
    )


def _to_rel_path(file_path: str, project_path: str) -> str:
    """Convert an absolute file path to a forward-slash relative path, or '' on failure."""
    try:
        rel = Path(file_path).relative_to(Path(project_path).resolve())
        return str(rel).replace("\\", "/")
    except ValueError:
        normalised = file_path.replace("\\", "/")
        # Already relative or can't resolve — return as-is
        return Path(normalised).name  # last resort: just the filename


def _find_in_map(
    component_map: dict,
    file_path: str,
    project_path: str,
) -> dict | None:
    """Try exact match then relative-path normalisation."""
    # 1. Exact match
    if file_path in component_map:
        return component_map[file_path]

    # 2. Normalise separators (Windows absolute → forward-slash)
    normalised = file_path.replace("\\", "/")
    if normalised in component_map:
        return component_map[normalised]

    # 3. If absolute, try relative to project root
    try:
        rel = Path(file_path).relative_to(Path(project_path).resolve())
        rel_str = str(rel).replace("\\", "/")
        if rel_str in component_map:
            return component_map[rel_str]
    except ValueError:
        pass

    # 4. Suffix match — last resort (avoids false positives with exact basename)
    for key, rec in component_map.items():
        if key.endswith("/" + normalised) or key == normalised:
            return rec

    return None
