"""SQLite CRUD for the symbol graph (symbol_nodes + symbol_edges + symbol_files)."""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognikernel.symbols.extractor import SymbolEdge, SymbolNode, SymbolUpdate

_log = logging.getLogger("cognikernel.symbols")


def apply_symbol_update(
    conn: sqlite3.Connection,
    update: "SymbolUpdate",
    *,
    project_path: str | None = None,
    session_id: str = "",
    last_action: str = "scan",
) -> None:
    """Upsert nodes/edges and delete stale paths. Idempotent.

    When `project_path` is provided, also upserts `symbol_files` rows so the
    PreToolUse hook's STEP 2 has authoritative file-level state. The hash and
    timestamp let the renderer present truthful "last refreshed" claims (B-2).
    Callers that just want the symbol_nodes/edges behavior can omit project_path.
    """
    from cognikernel.storage import symbol_files as sf

    # 1. Delete removed paths (nodes, outgoing edges, and incoming edges)
    for path in update.delete_paths:
        conn.execute(
            "DELETE FROM symbol_nodes WHERE project_id = ? AND path = ?",
            (update.project_id, path),
        )
        conn.execute(
            "DELETE FROM symbol_edges WHERE project_id = ? AND from_path = ?",
            (update.project_id, path),
        )
        conn.execute(
            "DELETE FROM symbol_edges WHERE project_id = ? AND to_path = ?",
            (update.project_id, path),
        )
        conn.execute(
            "DELETE FROM symbol_files WHERE project_id = ? AND path = ?",
            (update.project_id, path),
        )

    # 2. For each re-parsed path: delete existing nodes/edges first (fresh parse replaces all)
    upsert_paths = {n.path for n in update.upsert_nodes} | {e.from_path for e in update.upsert_edges}
    for path in upsert_paths:
        conn.execute(
            "DELETE FROM symbol_nodes WHERE project_id = ? AND path = ?",
            (update.project_id, path),
        )
        conn.execute(
            "DELETE FROM symbol_edges WHERE project_id = ? AND from_path = ?",
            (update.project_id, path),
        )

    # 3. Insert fresh nodes
    for node in update.upsert_nodes:
        conn.execute(
            """INSERT OR IGNORE INTO symbol_nodes
               (project_id, path, node_type, name, parent_name,
                signature, return_type, fields, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (node.project_id, node.path, node.node_type, node.name,
             node.parent_name, node.signature, node.return_type,
             node.fields, node.updated_at),
        )

    # 4. Insert fresh edges
    for edge in update.upsert_edges:
        conn.execute(
            """INSERT OR IGNORE INTO symbol_edges
               (project_id, from_path, to_path, edge_type, is_external)
               VALUES (?, ?, ?, ?, ?)""",
            (edge.project_id, edge.from_path, edge.to_path,
             edge.edge_type, 1 if edge.is_external else 0),
        )

    conn.commit()

    # 5. Symbol-files lifecycle (C1) — only when project_path is provided.
    if project_path is not None:
        now_ms = int(time.time() * 1000)
        for path in sorted(upsert_paths):
            symbol_count = sum(1 for n in update.upsert_nodes if n.path == path)
            abs_path = Path(project_path) / path
            content_sha = _sha256_of(abs_path) if abs_path.exists() else ""
            sf.upsert(
                conn,
                update.project_id,
                path,
                freshness="fresh",
                refreshed_at=now_ms,
                refreshed_in_session=session_id,
                last_action=last_action,
                content_sha256=content_sha,
                scan_status="scanned",
                symbol_count=symbol_count,
            )


def _sha256_of(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


def load_symbol_nodes(
    conn: sqlite3.Connection,
    project_id: str,
    paths: list[str] | None = None,
) -> list["SymbolNode"]:
    """Load symbol nodes for a project, optionally filtered to specific paths."""
    from cognikernel.symbols.extractor import SymbolNode

    if paths is not None:
        if not paths:
            return []
        placeholders = ",".join("?" * len(paths))
        rows = conn.execute(
            f"""SELECT project_id, path, node_type, name, parent_name,
                       signature, return_type, fields, updated_at
                FROM symbol_nodes
                WHERE project_id = ? AND path IN ({placeholders})
                ORDER BY id ASC""",
            [project_id, *paths],
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT project_id, path, node_type, name, parent_name,
                      signature, return_type, fields, updated_at
               FROM symbol_nodes
               WHERE project_id = ?
               ORDER BY id ASC""",
            (project_id,),
        ).fetchall()

    return [
        SymbolNode(
            project_id=r["project_id"],
            path=r["path"],
            node_type=r["node_type"],
            name=r["name"],
            parent_name=r["parent_name"],
            signature=r["signature"],
            return_type=r["return_type"],
            fields=r["fields"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


def load_symbol_edges(
    conn: sqlite3.Connection,
    project_id: str,
    from_paths: list[str] | None = None,
) -> list["SymbolEdge"]:
    """Load local import edges for a project (is_external=0 only)."""
    from cognikernel.symbols.extractor import SymbolEdge

    if from_paths is not None:
        if not from_paths:
            return []
        placeholders = ",".join("?" * len(from_paths))
        rows = conn.execute(
            f"""SELECT project_id, from_path, to_path, edge_type, is_external
                FROM symbol_edges
                WHERE project_id = ? AND is_external = 0 AND from_path IN ({placeholders})""",
            [project_id, *from_paths],
        ).fetchall()
    else:
        rows = conn.execute(
            """SELECT project_id, from_path, to_path, edge_type, is_external
               FROM symbol_edges
               WHERE project_id = ? AND is_external = 0""",
            (project_id,),
        ).fetchall()

    return [
        SymbolEdge(
            project_id=r["project_id"],
            from_path=r["from_path"],
            to_path=r["to_path"],
            edge_type=r["edge_type"],
            is_external=bool(r["is_external"]),
        )
        for r in rows
    ]


def prune_out_of_scope_symbols(
    conn,
    project_id: str,
    project_root,
) -> int:
    """Delete symbol rows for paths that should never have been scanned.

    Completes an invalidation rule that was always incomplete: build_symbol_update
    emits delete_paths only for git-DELETED files, so a path that left scope —
    vendored, cached, or gitignored — stayed in the graph forever, competing for
    PageRank and the skeleton token budget. One real store carried 4,655 such
    nodes (94% of its graph).

    This invalidates a DERIVED cache, not memory. The symbol graph is rebuilt by
    scanning disk, and any path pruned here reappears if it re-enters scope. No
    decision, constraint, or abandonment is touched.

    Keys on the EXCLUSION predicate (skip-dir component or gitignore match),
    never on set-difference against a discovery run: discovery caps at
    _MAX_FILES, so a project with more in-scope files than the cap has
    legitimate modules missing from any single run, and "delete what wasn't
    discovered" would permanently purge them. The predicate is cap-independent
    and idempotent.

    Returns the number of paths removed. Never raises.
    """
    from cognikernel.symbols.extractor import (
        _SKIP_DIRS,
        _is_gitignored,
        _load_gitignore_globs,
    )

    try:
        globs = _load_gitignore_globs(Path(project_root))
        rows = conn.execute(
            "SELECT DISTINCT path FROM symbol_nodes WHERE project_id = ?",
            (project_id,),
        ).fetchall()

        doomed = [
            path for (path,) in rows
            if path and (
                any(part in _SKIP_DIRS for part in path.split("/"))
                or (globs and _is_gitignored(path, globs))
            )
        ]
        for path in doomed:
            conn.execute(
                "DELETE FROM symbol_nodes WHERE project_id = ? AND path = ?",
                (project_id, path),
            )
            conn.execute(
                "DELETE FROM symbol_edges WHERE project_id = ? AND from_path = ?",
                (project_id, path),
            )
            conn.execute(
                "DELETE FROM symbol_edges WHERE project_id = ? AND to_path = ?",
                (project_id, path),
            )
            conn.execute(
                "DELETE FROM symbol_files WHERE project_id = ? AND path = ?",
                (project_id, path),
            )
        if doomed:
            conn.commit()
        return len(doomed)
    except Exception as exc:
        _log.warning("symbol_prune.failed", extra={"error": str(exc)})
        return 0
