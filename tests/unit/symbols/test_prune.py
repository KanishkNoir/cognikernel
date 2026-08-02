"""Out-of-scope symbol pruning (completes the D3 fix for existing stores).

build_symbol_update only ever emitted delete_paths for git-DELETED files, so a
path that leaves scope — because it is vendored, cached, or gitignored — stayed
in the graph forever, competing for PageRank and the skeleton token budget. One
real store carried 4,655 such nodes.

This is invalidation of a DERIVED cache, not repair of memory: the symbol graph
is rebuilt from disk and these nodes reappear if they ever come back into scope.
"""
import sqlite3
from pathlib import Path

from cognikernel.symbols.store import prune_out_of_scope_symbols


def _insert_node(conn: sqlite3.Connection, project_id: str, path: str) -> None:
    conn.execute(
        "INSERT INTO symbol_nodes "
        "(project_id, path, node_type, name, parent_name, signature, updated_at) "
        "VALUES (?, ?, 'function', ?, '', 'f()', 0)",
        (project_id, path, f"f_{path}"),
    )


def _paths(conn: sqlite3.Connection, project_id: str) -> set[str]:
    return {
        r[0] for r in conn.execute(
            "SELECT DISTINCT path FROM symbol_nodes WHERE project_id = ?", (project_id,)
        )
    }


class TestPruneOutOfScope:
    def test_removes_vendored_and_cache_paths(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        for p in (
            "src/app.py",
            ".uv-cache/archive-v0/attr/_make.py",
            ".pytest_tmp/basetemp/proj/main.py",
            "node_modules/pkg/index.js",
        ):
            _insert_node(conn, "p", p)
        removed = prune_out_of_scope_symbols(conn, "p", tmp_path)
        assert removed == 3
        assert _paths(conn, "p") == {"src/app.py"}

    def test_removes_gitignored_paths(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        _insert_node(conn, "p", "src/app.py")
        _insert_node(conn, "p", "generated/pb2.py")
        (tmp_path / ".gitignore").write_text("generated/\n", encoding="utf-8")
        prune_out_of_scope_symbols(conn, "p", tmp_path)
        assert _paths(conn, "p") == {"src/app.py"}

    def test_does_not_prune_by_set_difference(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        """CRITICAL: discovery caps at _MAX_FILES=500. A project with more
        in-scope files than the cap has legitimate modules absent from any
        single discovery run — pruning 'anything not currently discovered'
        would permanently purge them. Pruning must key on the EXCLUSION
        predicate, which is cap-independent.
        """
        for i in range(600):
            _insert_node(conn, "p", f"src/pkg/mod{i}.py")
        removed = prune_out_of_scope_symbols(conn, "p", tmp_path)
        assert removed == 0
        assert len(_paths(conn, "p")) == 600

    def test_is_idempotent(self, conn: sqlite3.Connection, tmp_path: Path) -> None:
        _insert_node(conn, "p", ".uv-cache/x/y.py")
        assert prune_out_of_scope_symbols(conn, "p", tmp_path) == 1
        assert prune_out_of_scope_symbols(conn, "p", tmp_path) == 0

    def test_scopes_to_project(self, conn: sqlite3.Connection, tmp_path: Path) -> None:
        _insert_node(conn, "p1", ".uv-cache/x/y.py")
        _insert_node(conn, "p2", ".uv-cache/x/y.py")
        prune_out_of_scope_symbols(conn, "p1", tmp_path)
        assert _paths(conn, "p2") == {".uv-cache/x/y.py"}

    def test_empty_graph_is_not_an_error(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        assert prune_out_of_scope_symbols(conn, "p", tmp_path) == 0
