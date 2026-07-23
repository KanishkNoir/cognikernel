from __future__ import annotations

import hashlib
import logging
import os
import re
import sqlite3
import stat
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from cognikernel.config import Config

_log = logging.getLogger("cognikernel.storage")
_WSL_DRIVE_RE = re.compile(r"^/(?:mnt|cygdrive)/([A-Za-z])(?:/(.*))?$")
_WINDOWS_DRIVE_RE = re.compile(r"^([A-Za-z]):(?:/(.*))?$")

# Raw sqlite3.connect() must never be called outside this module.
# All connections go through get_connection(), which enforces PRAGMAs.


def hash_project_path(project_path: str | Path) -> str:
    """Return a stable 16-char hex identifier for a project root path."""
    normalized = str(Path(project_path).resolve())
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def hash_project_identity(identity: str) -> str:
    """Return the DB id for an explicit logical project identity."""
    normalized = "identity:" + identity.strip()
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def normalized_project_path_key(project_path: str | Path) -> str:
    """Return a path key for comparing OS aliases without changing DB ids.

    `hash_project_path` remains the legacy DB filename rule. This comparator is
    deliberately separate so existing databases keep their names while callers
    can still recognize that `C:/repo` and `/mnt/c/repo` refer to the same
    Windows-backed checkout.
    """
    raw = str(project_path).strip().replace("\\", "/")
    while "//" in raw:
        raw = raw.replace("//", "/")
    raw = raw.rstrip("/") or raw

    m = _WSL_DRIVE_RE.match(raw)
    if m:
        rest = (m.group(2) or "").strip("/")
        return f"{m.group(1).lower()}:/{rest}".casefold()

    m = _WINDOWS_DRIVE_RE.match(raw)
    if m:
        rest = (m.group(2) or "").strip("/")
        return f"{m.group(1).lower()}:/{rest}".casefold()

    try:
        return Path(raw).expanduser().resolve().as_posix()
    except Exception:
        return raw


def project_paths_equivalent(left: str | Path, right: str | Path) -> bool:
    """Best-effort equivalence for native and mounted views of one checkout."""
    return normalized_project_path_key(left) == normalized_project_path_key(right)


def resolve_project_id(project_path: str | Path, config: Config) -> str:
    """Resolve the project DB id, honoring explicit identities and path aliases.

    Compatibility rule: when the legacy path-hash DB exists, keep using it.
    Otherwise, scan known project DBs for a meta.project_path that is equivalent
    to the requested path (for example Windows `C:/...` vs WSL `/mnt/c/...`).
    A configured `project_identity` is the opt-in escape hatch for genuinely
    different checkout paths that should share one logical memory store.
    """
    if config.project_identity:
        return hash_project_identity(config.project_identity)

    legacy_id = hash_project_path(project_path)
    if (config.projects_dir / f"{legacy_id}.db").exists():
        return legacy_id

    equivalent = _find_equivalent_project_id(project_path, config)
    return equivalent or legacy_id


def get_db_path(config: Config, project_id: str) -> Path:
    """Resolve the database file path for a project, creating parent dirs as needed."""
    config.projects_dir.mkdir(parents=True, exist_ok=True)
    return config.projects_dir / f"{project_id}.db"


def _find_equivalent_project_id(project_path: str | Path, config: Config) -> str | None:
    """Scan known project DBs for an equivalent recorded project_path.

    Speed: this sits on the session_capture hook fast path whenever the
    legacy-DB short-circuit misses (every capture of a NEW project), and
    opening each store costs ~25ms — over the hook budget on machines with
    many projects. A sidecar JSON index (path key per DB, validated by
    mtime+size) lets the scan open each DB once per machine instead of once
    per call. The index is advisory: corruption or staleness just causes a
    re-read of the affected DB; failures never affect resolution.
    """
    projects_dir = config.projects_dir
    if not projects_dir.exists():
        return None
    import json as _json
    index_path = projects_dir / "_path_keys.json"
    try:
        index = _json.loads(index_path.read_text(encoding="utf-8"))
        if not isinstance(index, dict):
            index = {}
    except Exception:
        index = {}

    wanted = normalized_project_path_key(project_path)
    dirty = False
    result: str | None = None
    for db_file in sorted(projects_dir.glob("*.db")):
        try:
            st = db_file.stat()
            cached = index.get(db_file.name)
            if cached and cached.get("mtime") == st.st_mtime_ns and cached.get("size") == st.st_size:
                key = cached.get("key")
            else:
                conn = sqlite3.connect(str(db_file), timeout=1.0)
                try:
                    row = conn.execute(
                        "SELECT value FROM meta WHERE key='project_path'"
                    ).fetchone()
                finally:
                    conn.close()
                key = normalized_project_path_key(row[0]) if row and row[0] else None
                index[db_file.name] = {"mtime": st.st_mtime_ns, "size": st.st_size, "key": key}
                dirty = True
            if key is not None and key == wanted:
                result = db_file.stem
                break
        except Exception:
            continue

    if dirty:
        try:
            index_path.write_text(_json.dumps(index), encoding="utf-8")
        except Exception:
            pass  # advisory cache — never fail resolution over it
    return result


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
