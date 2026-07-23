import sqlite3
from pathlib import Path

import pytest

from cognikernel.storage.connection import (
    get_connection,
    get_db_path,
    hash_project_identity,
    hash_project_path,
    normalized_project_path_key,
    project_paths_equivalent,
    resolve_project_id,
)
from cognikernel.config import Config


class TestPragmas:
    def test_wal_journal_mode(self, tmp_db: Path) -> None:
        with get_connection(tmp_db) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_synchronous_normal(self, tmp_db: Path) -> None:
        with get_connection(tmp_db) as conn:
            # 0=OFF 1=NORMAL 2=FULL 3=EXTRA
            sync = conn.execute("PRAGMA synchronous").fetchone()[0]
        assert sync == 1

    def test_foreign_keys_on(self, tmp_db: Path) -> None:
        with get_connection(tmp_db) as conn:
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1

    def test_temp_store_memory(self, tmp_db: Path) -> None:
        with get_connection(tmp_db) as conn:
            # 0=DEFAULT 1=FILE 2=MEMORY
            ts = conn.execute("PRAGMA temp_store").fetchone()[0]
        assert ts == 2

    def test_busy_timeout(self, tmp_db: Path) -> None:
        with get_connection(tmp_db) as conn:
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == 30000


class TestConnectionLifecycle:
    def test_row_factory_is_set(self, tmp_db: Path) -> None:
        with get_connection(tmp_db) as conn:
            row = conn.execute("SELECT 1 AS val").fetchone()
        assert row["val"] == 1

    def test_connection_closed_after_context(self, tmp_path: Path) -> None:
        db_path = tmp_path / "lifecycle.db"
        with get_connection(db_path) as conn:
            pass
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("SELECT 1")

    def test_exception_still_closes_connection(self, tmp_path: Path) -> None:
        db_path = tmp_path / "exc.db"
        conn_ref = None
        with pytest.raises(RuntimeError):
            with get_connection(db_path) as conn:
                conn_ref = conn
                raise RuntimeError("simulated failure")
        assert conn_ref is not None
        with pytest.raises(sqlite3.ProgrammingError):
            conn_ref.execute("SELECT 1")


class TestHashProjectPath:
    def test_same_path_same_hash(self) -> None:
        h1 = hash_project_path("/home/user/myproject")
        h2 = hash_project_path("/home/user/myproject")
        assert h1 == h2

    def test_different_paths_different_hashes(self) -> None:
        h1 = hash_project_path("/home/user/project_a")
        h2 = hash_project_path("/home/user/project_b")
        assert h1 != h2

    def test_hash_is_hex_string(self) -> None:
        h = hash_project_path("/some/path")
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_length(self) -> None:
        h = hash_project_path("/some/path")
        assert len(h) == 16

    def test_windows_and_wsl_mount_paths_compare_equivalent(self) -> None:
        win = "C:\\Users\\Admin\\repo"
        wsl = "/mnt/c/Users/Admin/repo"
        assert project_paths_equivalent(win, wsl)
        assert normalized_project_path_key(win) == normalized_project_path_key(wsl)

    def test_explicit_identity_has_stable_hash(self) -> None:
        assert hash_project_identity("acme-api") == hash_project_identity("acme-api")
        assert hash_project_identity("acme-api") != hash_project_identity("other")

    def test_resolve_uses_explicit_identity(self, tmp_path: Path) -> None:
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel", project_identity="acme-api")
        assert resolve_project_id("/any/checkout", cfg) == hash_project_identity("acme-api")

    def test_resolve_finds_existing_db_by_path_alias(self, tmp_path: Path) -> None:
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        cfg.projects_dir.mkdir(parents=True)
        db_path = cfg.projects_dir / "existing12345678.db"
        with get_connection(db_path) as conn:
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute(
                "INSERT INTO meta (key, value) VALUES ('project_path', ?)",
                ("C:\\Users\\Admin\\repo",),
            )
            conn.commit()

        assert resolve_project_id("/mnt/c/Users/Admin/repo", cfg) == "existing12345678"


class TestGetDbPath:
    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        config = Config(cognikernel_dir=tmp_path / "cognikernel")
        db_path = get_db_path(config, "abc123")
        assert db_path.parent.exists()

    def test_returns_db_extension(self, tmp_path: Path) -> None:
        config = Config(cognikernel_dir=tmp_path / "cognikernel")
        db_path = get_db_path(config, "abc123")
        assert db_path.suffix == ".db"

    def test_project_id_in_filename(self, tmp_path: Path) -> None:
        config = Config(cognikernel_dir=tmp_path / "cognikernel")
        db_path = get_db_path(config, "myproject99")
        assert "myproject99" in db_path.name
