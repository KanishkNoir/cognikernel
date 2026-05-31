"""MCP Resource renderers — CK-5: structured project memory as MCP Resources.

Each function renders one section of CogniKernel's event store as clean,
client-agnostic text suitable for any MCP client (Claude Code, Cursor, Copilot,
Codex, etc.) that can read resources. Unlike the injection block (a single dump
optimised for session-start), resources are queryable and typed individually —
a client can subscribe to 'constraints' without receiving threads or skeleton.

URI scheme:
  cognikernel://projects                        → list of all known projects
  cognikernel://project/{project_id}/state      → full session-start block
  cognikernel://project/{project_id}/constraints → CONSTRAINT_HARD events
  cognikernel://project/{project_id}/decisions   → ranked DECISION events
  cognikernel://project/{project_id}/graveyard   → APPROACH_ABANDONED_DO_NOT_RETRY
  cognikernel://project/{project_id}/skeleton    → AST symbol graph
  cognikernel://project/{project_id}/threads     → THREAD_OPEN events

project_id = SHA-256(resolved_path)[:16] — the DB filename stem. Clients
discover the mapping via cognikernel://projects.
"""
from __future__ import annotations

import json
from pathlib import Path

from memlora.config import Config
from memlora.storage.connection import get_connection, get_db_path, hash_project_path
from memlora.storage.migrations import run_migrations

_NOT_FOUND = "No CogniKernel data for this project. Run `memlora init <project_path>`."


# ── helpers ───────────────────────────────────────────────────────────────────


def _open(project_id: str, config: Config):
    """Return (conn_cm, project_id) for a project DB identified by its hex id."""
    db_path = config.projects_dir / f"{project_id}.db"
    if not db_path.exists():
        return None, None
    return get_connection(db_path), db_path


def _events(conn, project_id: str, event_type: str, limit: int = 50) -> list:
    return conn.execute(
        "SELECT payload, weight FROM events "
        "WHERE project_id=? AND event_type=? AND archived=0 AND superseded_by IS NULL "
        "ORDER BY weight DESC LIMIT ?",
        (project_id, event_type, limit),
    ).fetchall()


def _fmt_event(row, idx: int, *, show_weight: bool = False) -> list[str]:
    payload = json.loads(row["payload"])
    desc = (payload.get("description") or "").strip()
    rationale = (payload.get("rationale") or "").strip()
    weight_tag = f" [weight: {row['weight']:.2f}]" if show_weight else ""
    lines = [f"{idx}. {desc}{weight_tag}"]
    if rationale:
        lines.append(f"   Rationale: {rationale}")
    return lines


# ── project discovery ─────────────────────────────────────────────────────────


def list_projects(config: Config | None = None) -> str:
    """Return JSON array of all known CogniKernel projects.

    Scans ~/.memlora/projects/*.db and reads the project_path from meta.
    Projects without a stored path (pre-migration) are listed with path=null.
    """
    config = config or Config.load()
    projects_dir = config.projects_dir
    if not projects_dir.exists():
        return json.dumps([])

    projects: list[dict] = []
    for db_file in sorted(projects_dir.glob("*.db")):
        project_id = db_file.stem
        try:
            with get_connection(db_file) as conn:
                run_migrations(conn)
                row = conn.execute(
                    "SELECT value FROM meta WHERE key='project_path'"
                ).fetchone()
                project_path = (row["value"] if row and row["value"] else None)
                name = Path(project_path).name if project_path else project_id
                event_count = conn.execute(
                    "SELECT COUNT(*) FROM events WHERE project_id=? AND archived=0 "
                    "AND superseded_by IS NULL", (project_id,),
                ).fetchone()[0]
            projects.append({
                "id": project_id,
                "path": project_path,
                "name": name,
                "active_events": event_count,
                "resources": {
                    "state":       f"cognikernel://project/{project_id}/state",
                    "constraints": f"cognikernel://project/{project_id}/constraints",
                    "decisions":   f"cognikernel://project/{project_id}/decisions",
                    "graveyard":   f"cognikernel://project/{project_id}/graveyard",
                    "skeleton":    f"cognikernel://project/{project_id}/skeleton",
                    "threads":     f"cognikernel://project/{project_id}/threads",
                },
            })
        except Exception:
            continue

    return json.dumps(projects, indent=2)


# ── section renderers ─────────────────────────────────────────────────────────


def render_constraints(project_id: str, config: Config | None = None) -> str:
    """CONSTRAINT_HARD events — decisions that must never be violated."""
    config = config or Config.load()
    db_path = config.projects_dir / f"{project_id}.db"
    if not db_path.exists():
        return _NOT_FOUND
    try:
        with get_connection(db_path) as conn:
            run_migrations(conn)
            rows = _events(conn, project_id, "CONSTRAINT_HARD")
        if not rows:
            return "No hard constraints established yet."
        lines = ["### Hard constraints — never violate\n"]
        for i, row in enumerate(rows, 1):
            lines.extend(_fmt_event(row, i))
        return "\n".join(lines)
    except Exception as exc:
        return f"Error reading constraints: {exc}"


def render_decisions(project_id: str, config: Config | None = None) -> str:
    """DECISION events ranked by weight — key architectural choices."""
    config = config or Config.load()
    db_path = config.projects_dir / f"{project_id}.db"
    if not db_path.exists():
        return _NOT_FOUND
    try:
        with get_connection(db_path) as conn:
            run_migrations(conn)
            rows = _events(conn, project_id, "DECISION", limit=20)
        if not rows:
            return "No decisions recorded yet."
        lines = ["### Key decisions (highest weight first)\n"]
        for i, row in enumerate(rows, 1):
            lines.extend(_fmt_event(row, i, show_weight=True))
        return "\n".join(lines)
    except Exception as exc:
        return f"Error reading decisions: {exc}"


def render_graveyard(project_id: str, config: Config | None = None) -> str:
    """APPROACH_ABANDONED_DO_NOT_RETRY — explicitly rejected, never revisit."""
    config = config or Config.load()
    db_path = config.projects_dir / f"{project_id}.db"
    if not db_path.exists():
        return _NOT_FOUND
    try:
        with get_connection(db_path) as conn:
            run_migrations(conn)
            rows = _events(conn, project_id, "APPROACH_ABANDONED_DO_NOT_RETRY")
        if not rows:
            return "No rejected approaches recorded yet."
        lines = ["### Rejected approaches — do not re-suggest\n"]
        for i, row in enumerate(rows, 1):
            lines.extend(_fmt_event(row, i))
        return "\n".join(lines)
    except Exception as exc:
        return f"Error reading graveyard: {exc}"


def render_threads(project_id: str, config: Config | None = None) -> str:
    """THREAD_OPEN events — active work items and open questions."""
    config = config or Config.load()
    db_path = config.projects_dir / f"{project_id}.db"
    if not db_path.exists():
        return _NOT_FOUND
    try:
        with get_connection(db_path) as conn:
            run_migrations(conn)
            rows = _events(conn, project_id, "THREAD_OPEN", limit=10)
        if not rows:
            return "No open threads."
        lines = ["### Open work items\n"]
        for i, row in enumerate(rows, 1):
            payload = json.loads(row["payload"])
            desc = (payload.get("description") or "").strip()
            lines.append(f"{i}. {desc}")
        return "\n".join(lines)
    except Exception as exc:
        return f"Error reading threads: {exc}"


def render_skeleton(project_id: str, config: Config | None = None) -> str:
    """AST symbol graph — classes, functions, imports per file."""
    config = config or Config.load()
    db_path = config.projects_dir / f"{project_id}.db"
    if not db_path.exists():
        return _NOT_FOUND
    try:
        with get_connection(db_path) as conn:
            run_migrations(conn)
            from memlora.symbols.store import load_symbol_edges, load_symbol_nodes
            nodes = load_symbol_nodes(conn, project_id)
            edges = load_symbol_edges(conn, project_id)
        if not nodes:
            return "No codebase skeleton yet. Skeleton is built by the PostToolUse hook after each Write/Edit."
        from memlora.symbols.projection import compress_to_skeleton
        from memlora.symbols.render import render_skeleton_section
        entries = compress_to_skeleton(
            nodes, edges,
            budget_tokens=config.skeleton_budget,
        )
        return render_skeleton_section(entries) or "Skeleton is empty."
    except Exception as exc:
        return f"Error reading skeleton: {exc}"


def render_state(project_id: str, config: Config | None = None) -> str:
    """Full session-start injection block for the project."""
    config = config or Config.load()
    db_path = config.projects_dir / f"{project_id}.db"
    if not db_path.exists():
        return _NOT_FOUND
    try:
        # Resolve project_path from meta so we can call the existing render_state.
        with get_connection(db_path) as conn:
            run_migrations(conn)
            row = conn.execute(
                "SELECT value FROM meta WHERE key='project_path'"
            ).fetchone()
        project_path = row["value"] if row and row["value"] else None
        if not project_path:
            return "Project path not yet recorded. Run `memlora init <path>` to register."
        from memlora.integration.session import render_state as _render
        return _render(project_path, config=config)
    except Exception as exc:
        return f"Error rendering state: {exc}"


# ── dispatcher (used by mcp_server) ──────────────────────────────────────────


_SECTION_MAP = {
    "constraints": render_constraints,
    "decisions":   render_decisions,
    "graveyard":   render_graveyard,
    "threads":     render_threads,
    "skeleton":    render_skeleton,
    "state":       render_state,
}


def render_section(project_id: str, section: str, config: Config | None = None) -> str:
    fn = _SECTION_MAP.get(section)
    if fn is None:
        return f"Unknown section '{section}'. Valid: {', '.join(_SECTION_MAP)}"
    return fn(project_id, config)
