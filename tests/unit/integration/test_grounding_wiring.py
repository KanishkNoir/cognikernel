"""GroundingContext must actually be built and passed in production.

The gate's path-grounding branch is a no-op unless a caller supplies a context.
Adding the parameter is not the same as wiring it — the same mistake that left
the whole gate dead when it lived in persist_events.
"""
import sqlite3
from pathlib import Path

from cognikernel.integration.session import build_grounding_context


def _insert_node(conn: sqlite3.Connection, project_id: str, path: str) -> None:
    conn.execute(
        "INSERT INTO symbol_nodes "
        "(project_id, path, node_type, name, parent_name, signature, updated_at) "
        "VALUES (?, ?, 'function', ?, '', 'f()', 0)",
        (project_id, path, f"f_{path}"),
    )


class TestBuildGroundingContext:
    def test_includes_symbol_store_paths(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        _insert_node(conn, "p", "src/known.py")
        ctx = build_grounding_context(conn, "p", tmp_path)
        assert ctx.is_known("src/known.py")

    def test_includes_paths_discovered_on_disk(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "fresh.py").write_text("x = 1\n", encoding="utf-8")
        ctx = build_grounding_context(conn, "p", tmp_path)
        assert ctx.is_known("src/fresh.py")

    def test_unknown_path_is_not_known(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        _insert_node(conn, "p", "src/known.py")
        ctx = build_grounding_context(conn, "p", tmp_path)
        assert not ctx.is_known("src/nope.py")

    def test_near_miss_is_detected(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        _insert_node(conn, "p", "src/storage/connection.py")
        ctx = build_grounding_context(conn, "p", tmp_path)
        assert ctx.is_near_miss("rc/storage/connection.py")

    def test_empty_project_yields_empty_context_not_an_error(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        ctx = build_grounding_context(conn, "p", tmp_path)
        assert ctx.known_paths == frozenset()

    def test_never_raises_on_bad_root(self, conn: sqlite3.Connection) -> None:
        ctx = build_grounding_context(conn, "p", Path("/nonexistent/xyz123"))
        assert ctx.known_paths is not None


class TestEmptyContextIsNotEnforced:
    """A brand-new project has no inventory. Grounding against an empty set
    would downgrade every component event, so an empty context must be treated
    as 'cannot verify' rather than 'nothing is real'."""

    def test_empty_context_admits_component_events(
        self, conn: sqlite3.Connection, tmp_path: Path
    ) -> None:
        from cognikernel.model import Event
        from cognikernel.quality.gate import admit

        ctx = build_grounding_context(conn, "p", tmp_path)
        e = Event(
            project_id="p", session_id="s", event_type="COMPONENT_STATUS",
            payload={"description": "x", "path": "src/whatever.py"},
            content_hash="h", weight=1.0,
        )
        assert admit(e, ctx).action == "admit"
