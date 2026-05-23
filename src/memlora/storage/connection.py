from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import stat
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from memlora.config import Config

_log = logging.getLogger("memlora.storage")

# Raw sqlite3.connect() must never be called outside this module.
# All connections go through get_connection(), which enforces PRAGMAs.


def hash_project_path(project_path: str | Path) -> str:
    """Return a stable 16-char hex identifier for a project root path."""
    normalized = str(Path(project_path).resolve())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def get_db_path(config: Config, project_id: str) -> Path:
    """Resolve the database file path for a project, creating parent dirs as needed."""
    config.projects_dir.mkdir(parents=True, exist_ok=True)
    return config.projects_dir / f"{project_id}.db"


@contextmanager
def get_connection(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    """Open a SQLite connection with required PRAGMAs applied.

    Use as a context manager; connection is closed on exit.
    V1 opens/closes per call. Upgrade to a pool if profiling shows contention.
    """
    _warn_if_wal_large(db_path)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        _apply_pragmas(conn)
        _set_db_permissions(db_path)
        yield conn
    finally:
        conn.close()


# ── internals ────────────────────────────────────────────────────────────────

def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA busy_timeout = 30000")  # must be first — protects WAL switch
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA wal_autocheckpoint = 1000")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA mmap_size = 268435456")
    conn.execute("PRAGMA cache_size = -64000")
    conn.execute("PRAGMA foreign_keys = ON")


def _set_db_permissions(db_path: Path) -> None:
    if not db_path.exists():
        return
    try:
        os.chmod(db_path, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    except OSError:
        pass  # Windows does not fully support POSIX permissions


def _warn_if_wal_large(
    db_path: Path,
    threshold_bytes: int = 100 * 1024 * 1024,
) -> None:
    wal_path = Path(str(db_path) + "-wal")
    if not wal_path.exists():
        return
    size = wal_path.stat().st_size
    if size > threshold_bytes:
        _log.warning(
            "WAL file exceeds 100 MB — a long-running read transaction may be "
            "blocking checkpoints",
            extra={"db_path": str(db_path), "wal_size_bytes": size},
        )
