"""Integration layer — high-level orchestration used by the CLI and external callers."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from memlora.config import Config
from memlora.storage.connection import get_connection, get_db_path, hash_project_path
from memlora.storage.migrations import run_migrations
from memlora.storage.projections import Projection, load_or_rebuild

_log = logging.getLogger("memlora.integration")


def init_project(
    project_path: str | Path,
    config: Config | None = None,
) -> str:
    """Create and migrate the DB for a project. Idempotent. Returns project_id."""
    config = config or Config.load()
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)
    with get_connection(db_path) as conn:
        run_migrations(conn)
    _log.info("init_project.done", extra={"project_id": project_id, "db": str(db_path)})
    return project_id


def session_end(
    project_path: str | Path,
    session_id: str,
    transcript: str,
    config: Config | None = None,
    git_diff: str | None = None,
) -> dict[str, Any]:
    """Extract events from *transcript* and merge them into the project DB.

    Returns a stats dict with keys: extracted, inserted, updated,
    superseded, cascaded, archived.
    """
    from memlora.extraction.pipeline import SessionMetadata, extract_session
    from memlora.delta.merge import execute_merge

    config = config or Config.load()
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)

    with get_connection(db_path) as conn:
        run_migrations(conn)

    now = int(time.time() * 1000)
    session_meta = SessionMetadata(
        project_id=project_id,
        session_id=session_id,
        started_at=now,
        ended_at=now,
    )
    candidates = extract_session(transcript, session_meta, git_diff=git_diff)

    with get_connection(db_path) as conn:
        stats = execute_merge(conn, session_id, candidates)
        _update_symbol_graph(conn, project_id, str(project_path), git_diff)

    stats["extracted"] = len(candidates)
    _log.info("session_end.done", extra={"session_id": session_id, **stats})
    return stats


def get_projection(
    project_path: str | Path,
    config: Config | None = None,
) -> Projection:
    """Return the current (possibly rebuilt) Projection for *project_path*."""
    config = config or Config.load()
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)
    with get_connection(db_path) as conn:
        run_migrations(conn)
        return load_or_rebuild(conn, project_id)


def render_state(
    project_path: str | Path,
    config: Config | None = None,
) -> str:
    """Return the rendered injection block — what would be prepended to the LLM system prompt."""
    from memlora.storage.events import get_events_for_projection
    from memlora.compression.greedy import greedy_fill
    from memlora.injection.ordering import make_injection_context
    from memlora.injection.template import render_with_budget_enforcement
    from memlora.symbols.store import load_symbol_nodes, load_symbol_edges
    from memlora.symbols.projection import compress_to_skeleton

    config = config or Config.load()
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)

    with get_connection(db_path) as conn:
        run_migrations(conn)
        events = get_events_for_projection(conn, project_id, after_id=0)
        session_count: int = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM events WHERE project_id = ?",
            (project_id,),
        ).fetchone()[0]
        nodes = load_symbol_nodes(conn, project_id)
        edges = load_symbol_edges(conn, project_id)

    project_name = Path(project_path).resolve().name
    hot_files = _compute_hot_files(events)
    selected = greedy_fill(events, config.token_budget)
    hot_paths = frozenset(hf[0] for hf in hot_files)
    skeleton = compress_to_skeleton(
        nodes, edges,
        budget_tokens=config.skeleton_budget,
        hot_paths=hot_paths,
    )

    ctx = make_injection_context(
        events=selected,
        project_name=project_name,
        session_number=session_count,
        total_sessions=session_count,
        state_version=1,
        token_budget=config.token_budget,
    )
    ctx.hot_files = hot_files
    ctx.skeleton = skeleton
    return render_with_budget_enforcement(ctx)


def _update_symbol_graph(
    conn,
    project_id: str,
    project_path: str,
    git_diff: str | None,
) -> None:
    """Parse changed files and upsert symbol graph. Errors are logged, never raised."""
    try:
        from memlora.symbols.extractor import build_symbol_update
        from memlora.symbols.store import apply_symbol_update
        from memlora.extraction.git_augment import parse_diff

        changed_files = parse_diff(git_diff) if git_diff else []
        update = build_symbol_update(project_id, project_path, changed_files)
        apply_symbol_update(conn, update)
    except Exception as exc:
        _log.warning("symbol_graph.update_failed", extra={"error": str(exc)})


def _compute_hot_files(
    events: list,
    min_mentions: int = 2,
) -> list[tuple[str, int, str]]:
    """Aggregate COMPONENT_STATUS events by path, return paths with total_mentions >= min_mentions."""
    from collections import defaultdict
    files: dict = defaultdict(lambda: {"mentions": 0, "intent": ""})
    for e in events:
        if e.event_type != "COMPONENT_STATUS":
            continue
        path = e.payload.get("path", "")
        if not path:
            continue
        files[path]["mentions"] += e.mention_count or 1
        if not files[path]["intent"]:
            intent = e.payload.get("intent", "")
            if intent and intent != path:
                files[path]["intent"] = intent
    return sorted(
        [(p, d["mentions"], d["intent"]) for p, d in files.items()
         if d["mentions"] >= min_mentions],
        key=lambda x: -x[1],
    )
