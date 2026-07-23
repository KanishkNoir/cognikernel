from __future__ import annotations

import sqlite3
from pathlib import Path

from cognikernel.config import EXPECTED_PROJECTION_VERSION, EXPECTED_SCHEMA_VERSION

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def run_migrations(conn: sqlite3.Connection) -> None:
    """Apply all pending schema migrations and validate the projection version.

    Safe to call on every startup — idempotent by design.
    """
    _bootstrap_meta(conn)
    _run_schema_migrations(conn)
    _check_projection_version(conn)
    # FTS5 lives outside the numbered chain: availability depends on the user's
    # SQLite build, and a missing extension must degrade (lexical axis absent),
    # never abort migrations. ensure_fts is one meta SELECT once established.
    try:
        from cognikernel.storage.fts import ensure_fts
        ensure_fts(conn)
    except Exception:
        pass


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

    # Migrations that rebuild a table with inbound foreign keys (e.g. widening a
    # CHECK on a parent like raw_evidence) must run with FK enforcement OFF — with
    # it ON, DROP TABLE on the parent performs an implicit cascading DELETE that
    # wipes child rows. PRAGMA foreign_keys is a no-op inside a transaction, so it
    # cannot live in the migration .sql; toggle it here, around the loop, while we
    # are in autocommit. Connections normally run FK ON (connection.py); restore it.
    fk_was_on = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    if fk_was_on:
        conn.execute("PRAGMA foreign_keys = OFF")
    try:
        _apply_pending(conn, current)
    finally:
        if fk_was_on:
            conn.execute("PRAGMA foreign_keys = ON")


def _apply_pending(conn: sqlite3.Connection, current: int) -> None:
    for migration_file in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        version = int(migration_file.stem.split("_")[0])
        if version <= current:
            continue

        sql = migration_file.read_text(encoding="utf-8")
        # Atomicity (audit P1): the migration body AND its version bump must commit
        # as ONE transaction. Previously the script ran (auto-committing each DDL)
        # and the version was bumped in a separate commit, so a crash mid-script —
        # e.g. after a table RENAME but before the rebuild — left a half-applied
        # schema still recorded at the OLD version. The next startup re-ran the
        # migration, the RENAME failed (the table was already gone), and the DB
        # never booted again. Wrapping in an explicit BEGIN/COMMIT makes it
        # all-or-nothing: executescript implicitly COMMITs any pending tx first,
        # then runs our script verbatim, so the BEGIN we own brackets the whole
        # unit. Migration .sql files must NOT contain their own BEGIN/COMMIT or
        # transaction-incompatible statements (PRAGMA/VACUUM) — none do today.
        script = (
            "BEGIN;\n"
            f"{sql}\n"
            f"UPDATE meta SET value = '{version}' WHERE key = 'schema_version';\n"
            "COMMIT;"
        )
        try:
            conn.executescript(script)
        except Exception:
            # On a mid-script failure the BEGIN transaction is left open; roll it
            # back so the partial schema change is discarded and the DB stays at
            # `current`, then re-raise so startup surfaces the bad migration.
            conn.rollback()
            raise


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
