from __future__ import annotations

import sqlite3
from pathlib import Path

from memlora.config import EXPECTED_PROJECTION_VERSION, EXPECTED_SCHEMA_VERSION

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply all pending schema migrations and validate the projection version.

    Safe to call on every startup — idempotent by design.
    """
    _bootstrap_meta(conn)
    _run_schema_migrations(conn)
    _check_projection_version(conn)


# ── internals ────────────────────────────────────────────────────────────────

def _bootstrap_meta(conn: sqlite3.Connection) -> None:
    """Ensure the meta table exists with seed rows before any migrations run.

    This is the chicken-and-egg bootstrap: we need meta to know which
    migrations to run, but meta is also created inside migration 001.

    Reads the sqlite_master catalog first (zero write cost) to detect the
    fully-bootstrapped case (meta table exists + both rows present). This
    makes `run_migrations` safe to call concurrently with the background
    process-jobs worker — doctor/show won't fight the worker for the write
    lock when nothing actually needs writing.
    """
    # Fast path: meta table and both seed rows already exist — no writes needed.
    try:
        meta_exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if meta_exists:
            rows = conn.execute(
                "SELECT key FROM meta WHERE key IN ('schema_version','projection_version')"
            ).fetchall()
            if len(rows) >= 2:
                return  # fully bootstrapped, skip all writes
    except Exception:
        pass  # if the read fails, fall through to the full bootstrap

    # Slow path: table or seed rows are missing — do the writes.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '0')"
    )
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('projection_version', ?)",
        (str(EXPECTED_PROJECTION_VERSION),),
    )
    conn.commit()


def _run_schema_migrations(conn: sqlite3.Connection) -> None:
    current = int(
        conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
    )

    # Fast path: already up-to-date — no writes, no file I/O, no lock contention.
    if current >= EXPECTED_SCHEMA_VERSION:
        return

    for migration_file in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        version = int(migration_file.stem.split("_")[0])
        if version <= current:
            continue

        sql = migration_file.read_text(encoding="utf-8")
        # executescript issues an implicit COMMIT first, then runs all statements.
        conn.executescript(sql)
        # Record the new version in a separate transaction after the script commits.
        conn.execute(
            "UPDATE meta SET value = ? WHERE key = 'schema_version'",
            (str(version),),
        )
        conn.commit()


def _check_projection_version(conn: sqlite3.Connection) -> None:
    stored = int(
        conn.execute(
            "SELECT value FROM meta WHERE key = 'projection_version'"
        ).fetchone()[0]
    )
    if stored == EXPECTED_PROJECTION_VERSION:
        return

    # Projection schema changed — wipe cached projections and rebuild from events.
    _reset_all_projections(conn)
    conn.execute(
        "UPDATE meta SET value = ? WHERE key = 'projection_version'",
        (str(EXPECTED_PROJECTION_VERSION),),
    )
    conn.commit()


def _reset_all_projections(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM state_projections")
    conn.commit()
