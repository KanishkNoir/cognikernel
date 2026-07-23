"""Unit tests for symbol graph SQLite CRUD (store.py).

Uses the shared `conn` fixture from conftest.py (real migrated DB) so the full
schema including symbol_files (added in C0 migration 008) is available.
"""
import pytest
from cognikernel.symbols.extractor import SymbolNode, SymbolEdge, SymbolUpdate
from cognikernel.symbols.store import apply_symbol_update, load_symbol_nodes, load_symbol_edges


def _node(path: str, name: str, pid: str = "p1") -> SymbolNode:
    return SymbolNode(path=path, node_type="class", name=name, parent_name="",
                      signature="", return_type="", fields="", project_id=pid, updated_at=0)


def _edge(from_path: str, to_path: str, pid: str = "p1", external: bool = False) -> SymbolEdge:
    return SymbolEdge(project_id=pid, from_path=from_path, to_path=to_path,
                      edge_type="imports", is_external=external)


def _update(pid: str = "p1", nodes=None, edges=None, deletes=None) -> SymbolUpdate:
    return SymbolUpdate(project_id=pid, upsert_nodes=nodes or [],
                        upsert_edges=edges or [], delete_paths=deletes or [])


class TestApplySymbolUpdate:
    def test_nodes_inserted(self, conn) -> None:
        apply_symbol_update(conn, _update(nodes=[_node("src/m.py", "Quote")]))
        rows = conn.execute("SELECT name FROM symbol_nodes").fetchall()
        assert any(r["name"] == "Quote" for r in rows)

    def test_edges_inserted(self, conn) -> None:
        apply_symbol_update(conn, _update(edges=[_edge("src/a.py", "src/b.py")]))
        rows = conn.execute("SELECT * FROM symbol_edges").fetchall()
        assert len(rows) == 1

    def test_upsert_replaces_old_nodes_for_path(self, conn) -> None:
        apply_symbol_update(conn, _update(nodes=[_node("src/m.py", "OldClass")]))
        apply_symbol_update(conn, _update(nodes=[_node("src/m.py", "NewClass")]))
        rows = conn.execute("SELECT name FROM symbol_nodes WHERE path = 'src/m.py'").fetchall()
        names = {r["name"] for r in rows}
        assert "OldClass" not in names
        assert "NewClass" in names

    def test_delete_path_removes_nodes(self, conn) -> None:
        apply_symbol_update(conn, _update(nodes=[_node("src/m.py", "X")]))
        apply_symbol_update(conn, _update(deletes=["src/m.py"]))
        rows = conn.execute("SELECT * FROM symbol_nodes WHERE path = 'src/m.py'").fetchall()
        assert rows == []

    def test_delete_path_removes_edges(self, conn) -> None:
        apply_symbol_update(conn, _update(edges=[_edge("src/a.py", "src/b.py")]))
        apply_symbol_update(conn, _update(deletes=["src/a.py"]))
        rows = conn.execute("SELECT * FROM symbol_edges WHERE from_path = 'src/a.py'").fetchall()
        assert rows == []

    def test_delete_path_removes_incoming_edges(self, conn) -> None:
        # src/api.py → src/models.py; deleting models.py should clean the incoming edge
        apply_symbol_update(conn, _update(edges=[_edge("src/api.py", "src/models.py")]))
        apply_symbol_update(conn, _update(deletes=["src/models.py"]))
        rows = conn.execute("SELECT * FROM symbol_edges WHERE to_path = 'src/models.py'").fetchall()
        assert rows == [], "Incoming edges to deleted file must be cleaned"

    def test_project_path_populates_symbol_files(self, conn, tmp_path) -> None:
        """When project_path is passed, symbol_files rows are upserted (C1 invariant).

        This is the gate that makes strict mode work on session 2 of a project:
        the first session's symbol walk populates symbol_files even for files
        that never went through PostToolUse:Write/Edit.
        """
        # Real file on disk so the SHA256 computation succeeds.
        f = tmp_path / "src" / "m.py"
        f.parent.mkdir(parents=True)
        f.write_text("class X:\n    pass\n", encoding="utf-8")

        apply_symbol_update(
            conn,
            _update(nodes=[_node("src/m.py", "X")]),
            project_path=str(tmp_path),
            session_id="s1",
            last_action="scan",
        )

        row = conn.execute(
            "SELECT freshness, scan_status, symbol_count, last_action, content_sha256 "
            "FROM symbol_files WHERE project_id='p1' AND path='src/m.py'"
        ).fetchone()
        assert row is not None
        assert row["freshness"] == "fresh"
        assert row["scan_status"] == "scanned"
        assert row["symbol_count"] == 1
        assert row["last_action"] == "scan"
        assert len(row["content_sha256"]) == 64  # SHA256 hex digest

    def test_project_path_none_skips_symbol_files(self, conn) -> None:
        """Without project_path, symbol_files is untouched (back-compat behavior)."""
        apply_symbol_update(conn, _update(nodes=[_node("src/m.py", "X")]))

        n = conn.execute(
            "SELECT COUNT(*) FROM symbol_files WHERE project_id='p1'"
        ).fetchone()[0]
        assert n == 0

    def test_delete_path_also_removes_symbol_files_row(self, conn, tmp_path) -> None:
        """Deleting a file removes its symbol_files row too."""
        f = tmp_path / "src" / "m.py"
        f.parent.mkdir(parents=True)
        f.write_text("x = 1\n", encoding="utf-8")
        apply_symbol_update(
            conn,
            _update(nodes=[_node("src/m.py", "X")]),
            project_path=str(tmp_path),
        )

        apply_symbol_update(
            conn,
            _update(deletes=["src/m.py"]),
            project_path=str(tmp_path),
        )

        n = conn.execute(
            "SELECT COUNT(*) FROM symbol_files WHERE path='src/m.py'"
        ).fetchone()[0]
        assert n == 0

    def test_other_paths_unaffected_by_upsert(self, conn) -> None:
        apply_symbol_update(conn, _update(nodes=[_node("src/a.py", "A"), _node("src/b.py", "B")]))
        apply_symbol_update(conn, _update(nodes=[_node("src/a.py", "A2")]))
        rows = conn.execute("SELECT name FROM symbol_nodes WHERE path = 'src/b.py'").fetchall()
        assert any(r["name"] == "B" for r in rows)

    def test_idempotent_reparse(self, conn) -> None:
        update = _update(nodes=[_node("src/m.py", "Quote")])
        apply_symbol_update(conn, update)
        apply_symbol_update(conn, update)
        count = conn.execute("SELECT COUNT(*) FROM symbol_nodes").fetchone()[0]
        assert count == 1


class TestLoadSymbolNodes:
    def test_load_all(self, conn) -> None:
        apply_symbol_update(conn, _update(nodes=[
            _node("src/a.py", "A"), _node("src/b.py", "B"),
        ]))
        nodes = load_symbol_nodes(conn, "p1")
        assert {n.name for n in nodes} == {"A", "B"}

    def test_filter_by_path(self, conn) -> None:
        apply_symbol_update(conn, _update(nodes=[
            _node("src/a.py", "A"), _node("src/b.py", "B"),
        ]))
        nodes = load_symbol_nodes(conn, "p1", paths=["src/a.py"])
        assert all(n.path == "src/a.py" for n in nodes)

    def test_empty_paths_list_returns_empty(self, conn) -> None:
        apply_symbol_update(conn, _update(nodes=[_node("src/a.py", "A")]))
        assert load_symbol_nodes(conn, "p1", paths=[]) == []


class TestLoadSymbolEdges:
    def test_local_edges_returned(self, conn) -> None:
        apply_symbol_update(conn, _update(edges=[_edge("src/a.py", "src/b.py")]))
        edges = load_symbol_edges(conn, "p1")
        assert len(edges) == 1

    def test_external_edges_excluded(self, conn) -> None:
        apply_symbol_update(conn, _update(edges=[_edge("src/a.py", "sqlalchemy", external=True)]))
        edges = load_symbol_edges(conn, "p1")
        assert edges == []
