"""Migration 021 crash-safety + concurrency (T-104 / #14).

`CONTRIBUTING.md` requires a new migration to survive a crash mid-script,
and the runner (`storage/migrations.py`) implements that by wrapping the
migration body AND its version bump in one explicit transaction. That
guarantee is asserted by the runner's own docstring/comments and exercised
GENERICALLY against a synthetic migration set in
`tests/unit/storage/test_migrations.py::TestAtomicity` — this file exercises
it specifically against 021's REAL SQL, and adds the one property nothing
else in the suite covers: two connections racing the SAME upgrade.

Four scenarios, matching the issue's acceptance criteria:
  1. Crash mid-script on 021's real body -> rolls back cleanly -> a normal
     re-run still reaches v21.
  2. Re-entrancy -> the fast path, no writes, on an already-migrated store.
  3. Concurrent open -> two connections racing a v20 store both reach v21
     with no `duplicate column name` and no corruption.
  4. A static guard over EVERY migration file, not just 021 -- fails if a
     future migration embeds its own BEGIN/COMMIT/PRAGMA/VACUUM.
"""
from __future__ import annotations

import re
import shutil
import sqlite3
import threading
from pathlib import Path

import pytest

from cognikernel.config import EXPECTED_SCHEMA_VERSION
from cognikernel.storage.connection import get_connection
from cognikernel.storage.migrations import _MIGRATIONS_DIR, run_migrations
import cognikernel.storage.migrations as migrations_module


# ── shared setup: a real v20 store, built from the REAL 001-020 files ────────
# Mirrors tests/unit/storage/test_migrations.py::TestMigration021BeliefHistory
# — a faithful v20 shape, not a synthetic stand-in, so these tests exercise
# 021's actual SQL rather than a shape that merely resembles it.

def _build_v20_dir(dest: Path) -> None:
    dest.mkdir(exist_ok=True)
    for f in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        version = int(f.stem.split("_")[0])
        if version <= 20:
            (dest / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")


def _build_v20_store(tmp_path: Path, name: str = "v20.db") -> Path:
    db_path = tmp_path / name
    v20_dir = tmp_path / f"migrations_v20_{name}"
    _build_v20_dir(v20_dir)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    original = migrations_module._MIGRATIONS_DIR
    migrations_module._MIGRATIONS_DIR = v20_dir
    try:
        run_migrations(conn)
    finally:
        migrations_module._MIGRATIONS_DIR = original
    conn.close()

    version = sqlite3.connect(str(db_path)).execute(
        "SELECT value FROM meta WHERE key = 'schema_version'"
    ).fetchone()[0]
    assert version == "20", f"fixture setup failed: store is at v{version}, not v20"
    return db_path


class TestCrashMidScript:
    """021's real body, force-failed partway through."""

    def _corrupt_021_dir(self, tmp_path: Path) -> Path:
        """A migrations dir with the real 001-020 files plus a MODIFIED 021
        that appends one broken trailing statement -- so the real ALTER
        TABLE statements all run (uncommitted) before the script fails,
        proving even a LATE failure discards everything, not just an early
        one."""
        corrupt_dir = tmp_path / "migrations_corrupt"
        _build_v20_dir(corrupt_dir)
        real_021 = (_MIGRATIONS_DIR / "021_belief_history.sql").read_text(encoding="utf-8")
        broken_021 = real_021 + "\nINSERT INTO does_not_exist_table (x) VALUES (1);\n"
        (corrupt_dir / "021_belief_history.sql").write_text(broken_021, encoding="utf-8")
        return corrupt_dir

    def test_crash_leaves_version_at_20_with_no_new_columns(self, tmp_path: Path) -> None:
        db_path = _build_v20_store(tmp_path)
        corrupt_dir = self._corrupt_021_dir(tmp_path)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        original = migrations_module._MIGRATIONS_DIR
        migrations_module._MIGRATIONS_DIR = corrupt_dir
        try:
            with pytest.raises(sqlite3.OperationalError, match="does_not_exist_table"):
                run_migrations(conn)
        finally:
            migrations_module._MIGRATIONS_DIR = original

        version = conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0]
        assert version == "20"

        cols = {r[1] for r in conn.execute("PRAGMA table_info(events)").fetchall()}
        new_cols = {"superseded_at", "archived_at", "captured_at_sha", "supersede_reason"}
        assert not (new_cols & cols), f"partial application survived: {new_cols & cols}"
        conn.close()

    def test_subsequent_normal_run_still_reaches_v21(self, tmp_path: Path) -> None:
        """The crash must not leave the connection or the migrations
        directory state poisoned for a later, real attempt."""
        db_path = _build_v20_store(tmp_path)
        corrupt_dir = self._corrupt_021_dir(tmp_path)

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        original = migrations_module._MIGRATIONS_DIR
        migrations_module._MIGRATIONS_DIR = corrupt_dir
        try:
            with pytest.raises(sqlite3.OperationalError):
                run_migrations(conn)
        finally:
            migrations_module._MIGRATIONS_DIR = original
        conn.close()

        # A fresh connection, the REAL (unmodified) migrations dir this time.
        with get_connection(db_path) as conn2:
            run_migrations(conn2)
            version = conn2.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()[0]
            cols = {r[1] for r in conn2.execute("PRAGMA table_info(events)").fetchall()}

        assert version == str(EXPECTED_SCHEMA_VERSION)
        assert {"superseded_at", "archived_at", "captured_at_sha", "supersede_reason"} <= cols


class TestReentrancy:
    def test_second_run_on_an_already_migrated_store_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "reentrant.db"
        with get_connection(db_path) as conn:
            run_migrations(conn)
            assert conn.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()[0] == str(EXPECTED_SCHEMA_VERSION)

            changes_before = conn.total_changes
            run_migrations(conn)  # second call — must be a true no-op
            changes_after = conn.total_changes

        assert changes_after == changes_before, (
            "a second run_migrations() call on an up-to-date store performed "
            "writes — the fast path (current >= EXPECTED_SCHEMA_VERSION) was "
            "not actually taken"
        )


class TestConcurrentOpen:
    """Two connections racing a v20 -> v21 upgrade.

    Found a real bug writing this test: reading `current` once and looping
    (the pre-existing design) lets two connections both observe v20, then
    race to apply the same ALTER TABLE statements — the loser's own DDL
    collides with the winner's already-committed columns and raises
    'duplicate column name'. Fixed in storage/migrations.py:_apply_pending
    by re-checking the ACTUAL schema_version after a failed apply and
    treating "someone else already finished this migration" as a benign
    no-op rather than a real failure.
    """

    def test_two_connections_racing_both_reach_v21_without_error(
        self, tmp_path: Path
    ) -> None:
        db_path = _build_v20_store(tmp_path, name="race.db")

        # Warm the store into WAL mode via the app's own connection path
        # FIRST, exactly as `cognikernel init` does exactly once before any
        # concurrent access to a project's store is possible. Racing the
        # first-ever WAL-mode switch itself is a real but separate,
        # pre-existing SQLite behavior (PRAGMA journal_mode = WAL needs an
        # exclusive lock to switch) — not what this test is about, and not
        # a scenario that occurs in production (a store is never opened
        # concurrently before its first, synchronous `init`).
        with get_connection(db_path):
            pass

        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def _race() -> None:
            try:
                with get_connection(db_path) as conn:
                    barrier.wait(timeout=10)
                    run_migrations(conn)
            except BaseException as exc:  # a racing connection must never crash silently
                errors.append(exc)

        t0 = threading.Thread(target=_race)
        t1 = threading.Thread(target=_race)
        t0.start()
        t1.start()
        t0.join(timeout=30)
        t1.join(timeout=30)

        assert not errors, f"racing connection(s) raised: {errors}"
        assert not t0.is_alive() and not t1.is_alive(), "a racing connection hung"

        with get_connection(db_path) as conn:
            version = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()[0]
            cols = [r[1] for r in conn.execute("PRAGMA table_info(events)").fetchall()]

        assert version == str(EXPECTED_SCHEMA_VERSION)
        dupes = {c for c in cols if cols.count(c) > 1}
        assert not dupes, f"duplicate columns from a double-applied migration: {dupes}"


class TestNoTransactionControlInMigrationFiles:
    """Guards the RULE, not just migration 021 — CONTRIBUTING.md requires
    every migration to be free of BEGIN/COMMIT/PRAGMA/VACUUM, since the
    runner supplies its own transaction wrapper and these are
    transaction-incompatible or would silently break the atomicity
    guarantee. Comments mentioning these words in prose (several existing
    files explain PRAGMA/commit behavior in English) must not false-positive
    — only an actual statement counts.
    """

    _FORBIDDEN = re.compile(r"^\s*(BEGIN|COMMIT|PRAGMA|VACUUM)\b", re.IGNORECASE)

    @staticmethod
    def _strip_line_comments(sql: str) -> str:
        return "\n".join(line.split("--", 1)[0] for line in sql.splitlines())

    def test_no_migration_file_contains_transaction_control(self) -> None:
        offenders: dict[str, list[str]] = {}
        for f in sorted(_MIGRATIONS_DIR.glob("*.sql")):
            stripped = self._strip_line_comments(f.read_text(encoding="utf-8"))
            for statement in stripped.split(";"):
                if self._FORBIDDEN.match(statement):
                    offenders.setdefault(f.name, []).append(statement.strip()[:60])

        assert not offenders, (
            f"migration file(s) contain BEGIN/COMMIT/PRAGMA/VACUUM as real "
            f"statements, not just in comments — the runner's own transaction "
            f"wrapper (BEGIN;...COMMIT;) would then double up or the "
            f"transaction-incompatible statement would break executescript: "
            f"{offenders}"
        )

    def test_the_guard_actually_catches_a_real_violation(self) -> None:
        """A meta-test for the meta-test (mirrors the spirit of
        test_architecture_contracts.py's own guard against a silently
        broken contract): prove the regex/comment-stripping logic above
        would ACTUALLY flag a real violation, not just happen to find none
        in the current file set."""
        sql_with_real_pragma = (
            "-- a comment mentioning PRAGMA and COMMIT does not count\n"
            "ALTER TABLE events ADD COLUMN x TEXT;\n"
            "PRAGMA foreign_keys = OFF;\n"
        )
        stripped = self._strip_line_comments(sql_with_real_pragma)
        hits = [s for s in stripped.split(";") if self._FORBIDDEN.match(s)]
        assert hits, "the guard failed to catch a real PRAGMA statement"
