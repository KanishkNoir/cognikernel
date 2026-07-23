"""CK-5 — MCP Resource renderers and project discovery."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

import cognikernel.integration.cli as cli
from cognikernel.integration.resources import (
    list_projects,
    render_constraints,
    render_decisions,
    render_graveyard,
    render_section,
    render_skeleton,
    render_threads,
)
from cognikernel.storage.migrations import run_migrations


@pytest.fixture
def project(tmp_path: Path, monkeypatch):
    """Initialise a real project dir + DB, return (project_dir, project_id)."""
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
    proj = tmp_path / "myproject"
    proj.mkdir()
    cli._cmd_init(argparse.Namespace(project_path=str(proj)))
    from cognikernel.config import Config
    from cognikernel.storage.connection import get_db_path, hash_project_path
    pid = hash_project_path(str(proj))
    cfg = Config.load(project_path=str(proj))
    db = get_db_path(cfg, pid)
    return proj, pid, db, cfg


def _insert_event(conn, pid, etype, desc, rationale="", weight=1.0):
    conn.execute(
        "INSERT INTO events (project_id, session_id, created_at, event_type, "
        "payload, content_hash, weight, mention_count) VALUES (?,?,1,?,?,?,?,1)",
        (pid, "s", etype,
         json.dumps({"description": desc, "rationale": rationale}),
         f"h-{etype[:4]}-{desc[:8]}", weight),
    )
    conn.commit()


# ── project_path stored in meta ───────────────────────────────────────────────

def test_init_stores_project_path_in_meta(project, tmp_path: Path) -> None:
    proj, pid, db, cfg = project
    from cognikernel.storage.connection import get_connection
    with get_connection(db) as conn:
        row = conn.execute("SELECT value FROM meta WHERE key='project_path'").fetchone()
    assert row is not None
    assert Path(row["value"]).resolve() == proj.resolve()


# ── list_projects ─────────────────────────────────────────────────────────────

def test_list_projects_returns_known_project(project, tmp_path: Path) -> None:
    proj, pid, db, cfg = project
    result = json.loads(list_projects(cfg))
    ids = [p["id"] for p in result]
    assert pid in ids
    match = next(p for p in result if p["id"] == pid)
    assert "constraints" in match["resources"]
    assert match["resources"]["constraints"] == f"cognikernel://project/{pid}/constraints"


def test_list_projects_empty_dir_returns_empty_list(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "empty"))
    from cognikernel.config import Config
    cfg = Config.load()
    result = json.loads(list_projects(cfg))
    assert result == []


# ── constraints renderer ──────────────────────────────────────────────────────

def test_render_constraints_returns_not_found_for_missing_project(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
    from cognikernel.config import Config
    cfg = Config.load()
    result = render_constraints("deadbeefdeadbeef", cfg)
    assert "cognikernel init" in result.lower() or "no cognikernel" in result.lower()


def test_render_constraints_returns_events(project) -> None:
    proj, pid, db, cfg = project
    from cognikernel.storage.connection import get_connection
    with get_connection(db) as conn:
        _insert_event(conn, pid, "CONSTRAINT_HARD", "Never store secrets in plaintext",
                      rationale="Security baseline")
    result = render_constraints(pid, cfg)
    assert "Never store secrets in plaintext" in result
    assert "Security baseline" in result


def test_render_constraints_empty_message(project) -> None:
    proj, pid, db, cfg = project
    result = render_constraints(pid, cfg)
    assert "No hard constraints" in result


# ── decisions renderer ────────────────────────────────────────────────────────

def test_render_decisions_shows_weight(project) -> None:
    proj, pid, db, cfg = project
    from cognikernel.storage.connection import get_connection
    with get_connection(db) as conn:
        _insert_event(conn, pid, "DECISION", "Use PostgreSQL", weight=1.5)
    result = render_decisions(pid, cfg)
    assert "PostgreSQL" in result
    assert "1.50" in result  # weight shown


# ── graveyard renderer ────────────────────────────────────────────────────────

def test_render_graveyard_shows_rejected(project) -> None:
    proj, pid, db, cfg = project
    from cognikernel.storage.connection import get_connection
    with get_connection(db) as conn:
        _insert_event(conn, pid, "APPROACH_ABANDONED_DO_NOT_RETRY",
                      "Celery broker", rationale="Too heavy for local dev")
    result = render_graveyard(pid, cfg)
    assert "Celery broker" in result
    assert "Too heavy" in result


# ── threads renderer ──────────────────────────────────────────────────────────

def test_render_threads_shows_open_items(project) -> None:
    proj, pid, db, cfg = project
    from cognikernel.storage.connection import get_connection
    with get_connection(db) as conn:
        _insert_event(conn, pid, "THREAD_OPEN", "Implement auth end-to-end")
    result = render_threads(pid, cfg)
    assert "auth end-to-end" in result


# ── dispatcher ────────────────────────────────────────────────────────────────

def test_render_section_dispatcher(project) -> None:
    proj, pid, db, cfg = project
    for section in ("constraints", "decisions", "graveyard", "threads", "skeleton", "state"):
        result = render_section(pid, section, cfg)
        assert isinstance(result, str)
        assert len(result) > 0


def test_render_section_unknown_returns_error(project) -> None:
    proj, pid, db, cfg = project
    result = render_section(pid, "nonexistent", cfg)
    assert "Unknown section" in result


def test_render_section_rejects_malformed_project_id(project, tmp_path: Path) -> None:
    """L3: project_id arrives from a client-supplied resource URI. Anything but
    16 lowercase hex chars is not-found — a `../` traversal must never become a
    filesystem path (run_migrations would write schema into the target .db)."""
    proj, pid, db, cfg = project
    # An existing .db OUTSIDE projects_dir that a traversal id would reach.
    outside = cfg.projects_dir.parent.parent / "outside.db"
    outside.write_bytes(b"")
    evil = "../../outside"
    for section in ("constraints", "decisions", "graveyard", "threads", "skeleton", "state"):
        out = render_section(evil, section, cfg)
        assert "cognikernel init" in out.lower() or "no cognikernel" in out.lower()
    assert outside.read_bytes() == b""  # never opened as SQLite, never migrated
    # Well-formed but unknown ids still read as not-found, not as errors.
    assert "cognikernel init" in render_section("0123456789abcdef", "state", cfg).lower()


# ── MCP server registers resources ───────────────────────────────────────────

def test_mcp_server_has_seven_resources() -> None:
    """FastMCP should register 1 static + 6 template resources (CK-5)."""
    from cognikernel.integration.mcp_server import _mcp
    # list_resources covers static; list_resource_templates covers templates.
    # We just verify both lists are populated.
    import asyncio
    async def _check():
        static = await _mcp.list_resources()
        templates = await _mcp.list_resource_templates()
        return static, templates
    static, templates = asyncio.run(_check())
    assert len(static) >= 1, "cognikernel://projects must be listed"
    assert len(templates) >= 6, "6 template resources expected"
    uris = [str(r.uri) for r in static]
    assert any("projects" in u for u in uris)
    template_uris = [str(t.uriTemplate) for t in templates]
    for section in ("state", "constraints", "decisions", "graveyard", "skeleton", "threads"):
        assert any(section in u for u in template_uris), f"Missing template for {section}"


# ── skeleton path filter (ck-skeleton / skeleton MCP tool) ────────────────────

def _seed_skeleton(db, pid):
    from cognikernel.storage.connection import get_connection
    from cognikernel.symbols.extractor import SymbolNode, SymbolUpdate
    nodes = [
        SymbolNode(path="src/router.py", node_type="function", name="route",
                   parent_name="", signature="route(request) -> Response",
                   return_type="Response", fields="", project_id=pid, updated_at=0),
        SymbolNode(path="src/cache.py", node_type="class", name="CompletionCache",
                   parent_name="", signature="", return_type="", fields="ttl_s",
                   project_id=pid, updated_at=0),
    ]
    update = SymbolUpdate(project_id=pid, upsert_nodes=nodes, upsert_edges=[], delete_paths=[])
    from cognikernel.symbols.store import apply_symbol_update
    with get_connection(db) as conn:
        apply_symbol_update(conn, update)


def test_render_skeleton_filter_matches_single_file(project) -> None:
    proj, pid, db, cfg = project
    _seed_skeleton(db, pid)
    result = render_skeleton(pid, cfg, path_filter="router")
    assert "router.py" in result
    assert "cache.py" not in result          # only the matched file renders


def test_render_skeleton_filter_no_match_lists_known_files(project) -> None:
    proj, pid, db, cfg = project
    _seed_skeleton(db, pid)
    result = render_skeleton(pid, cfg, path_filter="does_not_exist.py")
    assert "No skeleton entry matches" in result
    assert "router.py" in result             # suggests known files


def test_render_skeleton_no_filter_renders_all(project) -> None:
    proj, pid, db, cfg = project
    _seed_skeleton(db, pid)
    result = render_skeleton(pid, cfg)
    assert "router.py" in result and "cache.py" in result
