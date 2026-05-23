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
    CREATE TABLE IF NOT EXISTS makes both calls safe.
    """
    conn.execute(
        "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', '0')"
    )
    # Fresh databases start at the current projection version — no rebuild needed.
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
