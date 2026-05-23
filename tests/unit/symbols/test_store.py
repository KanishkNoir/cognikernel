"""Unit tests for symbol graph SQLite CRUD (store.py)."""
import sqlite3
import pytest
from memlora.symbols.extractor import SymbolNode, SymbolEdge, SymbolUpdate
from memlora.symbols.store import apply_symbol_update, load_symbol_nodes, load_symbol_edges


@pytest.fixture
def conn():
    """In-memory SQLite with symbol graph schema."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript("""
        CREATE TABLE symbol_nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL, path TEXT NOT NULL, node_type TEXT NOT NULL,
            name TEXT NOT NULL, parent_name TEXT NOT NULL DEFAULT '',
            signature TEXT NOT NULL DEFAULT '', return_type TEXT NOT NULL DEFAULT '',
            fields TEXT NOT NULL DEFAULT '', updated_at INTEGER NOT NULL
        );
        CREATE UNIQUE INDEX idx_symbol_nodes_unique
            ON symbol_nodes (project_id, path, node_type, name, parent_name);
        CREATE TABLE symbol_edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL, from_path TEXT NOT NULL,
            to_path TEXT NOT NULL, edge_type TEXT NOT NULL DEFAULT 'imports',
            is_external INTEGER NOT NULL DEFAULT 0
        );
        CREATE UNIQUE INDEX idx_symbol_edges_unique
            ON symbol_edges (project_id, from_path, to_path, edge_type);
    """)
    yield c
    c.close()


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
