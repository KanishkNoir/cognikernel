"""CK-5 — MCP Resource renderers and project discovery."""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest

import memlora.integration.cli as cli
from memlora.integration.resources import (
    list_projects,
    render_constraints,
    render_decisions,
    render_graveyard,
    render_section,
    render_skeleton,
    render_threads,
)
from memlora.storage.migrations import run_migrations


@pytest.fixture
def project(tmp_path: Path, monkeypatch):
    """Initialise a real project dir + DB, return (project_dir, project_id)."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "data"))
    proj = tmp_path / "myproject"
    proj.mkdir()
    cli._cmd_init(argparse.Namespace(project_path=str(proj)))
    from memlora.config import Config
    from memlora.storage.connection import get_db_path, hash_project_path
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
    from memlora.storage.connection import get_connection
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
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "empty"))
    from memlora.config import Config
    cfg = Config.load()
    result = json.loads(list_projects(cfg))
    assert result == []


# ── constraints renderer ──────────────────────────────────────────────────────

def test_render_constraints_returns_not_found_for_missing_project(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "data"))
    from memlora.config import Config
    cfg = Config.load()
    result = render_constraints("deadbeefdeadbeef", cfg)
    assert "memlora init" in result.lower() or "no cognikernel" in result.lower()


def test_render_constraints_returns_events(project) -> None:
    proj, pid, db, cfg = project
    from memlora.storage.connection import get_connection
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
    from memlora.storage.connection import get_connection
    with get_connection(db) as conn:
        _insert_event(conn, pid, "DECISION", "Use PostgreSQL", weight=1.5)
    result = render_decisions(pid, cfg)
    assert "PostgreSQL" in result
    assert "1.50" in result  # weight shown


# ── graveyard renderer ────────────────────────────────────────────────────────

def test_render_graveyard_shows_rejected(project) -> None:
    proj, pid, db, cfg = project
    from memlora.storage.connection import get_connection
    with get_connection(db) as conn:
        _insert_event(conn, pid, "APPROACH_ABANDONED_DO_NOT_RETRY",
                      "Celery broker", rationale="Too heavy for local dev")
    result = render_graveyard(pid, cfg)
    assert "Celery broker" in result
    assert "Too heavy" in result


# ── threads renderer ──────────────────────────────────────────────────────────

def test_render_threads_shows_open_items(project) -> None:
    proj, pid, db, cfg = project
    from memlora.storage.connection import get_connection
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


# ── MCP server registers resources ───────────────────────────────────────────

def test_mcp_server_has_seven_resources() -> None:
    """FastMCP should register 1 static + 6 template resources (CK-5)."""
    from memlora.integration.mcp_server import _mcp
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
