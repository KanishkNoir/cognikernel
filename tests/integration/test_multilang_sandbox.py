"""End-to-end verification on a multi-language sandbox project.

This exists because two bugs shipped that every unit test missed:

  1. The tree-sitter binding break — TS/JS extraction returned 0 symbols for
     every pip install, invisible because extraction fails open.
  2. Grounding penalised languages the discovery walk does not glob, so every
     component event in a Go/Rust/Java project was downgraded off the block.

Both were found by running the REAL capture path over a realistic project and
looking at what was actually stored. That is what this test does.

The fixture under tests/fixtures/sandbox/ is a small but genuine multi-language
repo: Go with grouped type declarations and pointer/value receivers, Python with
classes and methods, Java with annotations and a nested class, and TypeScript
with an import edge between two files.
"""
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parents[1] / "fixtures" / "sandbox"

TRANSCRIPT = """\
User: We're building paykit. Decide the connection strategy.
Assistant: I edited internal/db/pool.go to add a bounded pool.
Assistant: We must never open more than 25 database connections in production.
Assistant: Record Redis as an explicitly abandoned approach we will never revisit.
Assistant: I updated src/payments/charge.py so ChargeService.authorize returns the auth id.
Assistant: I also changed web/src/checkout.ts and java/com/acme/billing/Invoice.java.
Assistant: Pick up the last task as if the break never happened.
"""


@pytest.fixture()
def captured(tmp_path, monkeypatch):
    """Run the real capture path over a copy of the sandbox; yield (conn, pid)."""
    import shutil

    proj = tmp_path / "demo"
    shutil.copytree(FIXTURE, proj)
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "home"))

    from cognikernel.config import Config
    from cognikernel.integration.session import init_project, session_end
    from cognikernel.storage.connection import get_db_path, resolve_project_id

    init_project(proj)
    session_end(proj, "sess-1", TRANSCRIPT)

    cfg = Config.load(project_path=proj)
    pid = resolve_project_id(proj, cfg)
    conn = sqlite3.connect(get_db_path(cfg, pid))
    yield conn, pid, proj
    conn.close()


class TestSymbolExtractionByLanguage:
    def test_python_and_typescript_yield_symbols(self, captured) -> None:
        """Guards the binding break: TS returning 0 symbols is the failure mode."""
        conn, pid, _ = captured
        by_ext: dict[str, int] = {}
        for (p,) in conn.execute(
            "SELECT path FROM symbol_nodes WHERE project_id=?", (pid,)
        ):
            by_ext[Path(p).suffix] = by_ext.get(Path(p).suffix, 0) + 1
        assert by_ext.get(".py", 0) > 0, "Python extraction produced nothing"
        assert by_ext.get(".ts", 0) > 0, "TypeScript extraction produced nothing"

    def test_typescript_import_edge_resolves_locally(self, captured) -> None:
        conn, pid, _ = captured
        edges = conn.execute(
            "SELECT from_path, to_path, is_external FROM symbol_edges "
            "WHERE project_id=? AND from_path LIKE '%checkout.ts'", (pid,)
        ).fetchall()
        assert any(not ext and "service" in to for _f, to, ext in edges), edges


class TestGroundingDoesNotPenaliseUnparsedLanguages:
    """A .go / .java file the walk cannot parse must not be treated as absent."""

    def test_go_and_java_components_keep_full_weight(self, captured) -> None:
        conn, pid, _ = captured
        rows = dict(conn.execute(
            "SELECT json_extract(payload,'$.path'), weight FROM events "
            "WHERE project_id=? AND event_type='COMPONENT_STATUS'", (pid,)
        ))
        go = rows.get("internal/db/pool.go")
        java = rows.get("java/com/acme/billing/Invoice.java")
        py = rows.get("src/payments/charge.py")
        assert go is not None and java is not None and py is not None, rows
        assert go == py, f"Go component downgraded vs Python: {go} != {py}"
        assert java == py, f"Java component downgraded vs Python: {java} != {py}"

    def test_no_unverified_marker_on_unparsed_languages(self, captured) -> None:
        conn, pid, _ = captured
        for (payload,) in conn.execute(
            "SELECT payload FROM events WHERE project_id=? "
            "AND event_type='COMPONENT_STATUS'", (pid,)
        ):
            if ".go" in payload or ".java" in payload:
                assert "unverified" not in payload, payload


class TestQualityGateOnRealCapture:
    def test_harness_boilerplate_never_reaches_the_store(self, captured) -> None:
        conn, pid, _ = captured
        rows = conn.execute(
            "SELECT payload FROM events WHERE project_id=?", (pid,)
        ).fetchall()
        blob = " ".join(r[0] for r in rows)
        assert "as if the break never happened" not in blob

    def test_real_constraint_is_kept_at_full_weight(self, captured) -> None:
        conn, pid, _ = captured
        row = conn.execute(
            "SELECT weight FROM events WHERE project_id=? AND event_type='CONSTRAINT_HARD' "
            "AND json_extract(payload,'$.description') LIKE '%25 database connections%'",
            (pid,),
        ).fetchone()
        assert row is not None and row[0] == 1.0


class TestRenderedBlock:
    def test_block_lists_first_party_files_and_no_boilerplate(self, captured) -> None:
        from cognikernel.integration.session import render_state

        _conn, _pid, proj = captured
        block = render_state(str(proj))
        assert "src/payments/charge.py" in block
        assert "web/src/checkout.ts" in block
        assert "as if the break never happened" not in block

    def test_cross_type_duplicate_renders_once(self, captured) -> None:
        """The Redis abandonment is stored under several types; the render-time
        structural filter must collapse it to a single line."""
        from cognikernel.integration.session import render_state

        _conn, _pid, proj = captured
        block = render_state(str(proj))
        assert block.lower().count("explicitly abandoned approach") <= 1
