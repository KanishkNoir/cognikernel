import sqlite3
import subprocess
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


class TestRepoRootAnchoring:
    """#33: project identity is the repo root, not the cwd.

    An agent that runs `cd packages/toolbelt-core` mid-session was otherwise
    silently starting a SECOND memory store for the same project. Measured on
    one real session: 8 captures, 7 of which landed in a subpackage store,
    leaving the project's own store holding 22 of 284 transcript lines — while
    every capture reported success.
    """

    @staticmethod
    def _repo(tmp_path: Path) -> Path:
        root = tmp_path / "repo"
        (root / "packages" / "core").mkdir(parents=True)
        try:
            subprocess.run(
                ["git", "init", "-q", str(root)],
                check=True,
                capture_output=True,
            )
        except FileNotFoundError:
            pytest.skip("git is required for repo-root anchoring tests")
        return root

    def test_subdirectory_resolves_to_the_root_store_when_one_exists(
        self, tmp_path: Path
    ) -> None:
        """The #33 case exactly: the project has a store, the agent cd's into
        a subpackage, and the capture must go to the project's store."""
        root = self._repo(tmp_path)
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        cfg.projects_dir.mkdir(parents=True)
        root_id = hash_project_path(root.resolve())
        (cfg.projects_dir / f"{root_id}.db").touch()

        assert resolve_project_id(root / "packages" / "core", cfg) == root_id

    def test_unrelated_projects_under_an_umbrella_repo_stay_separate(
        self, tmp_path: Path
    ) -> None:
        """The mirror-image bug this must NOT introduce.

        `git rev-parse --show-toplevel` walks up as far as it takes, so a user
        whose ~/code or dotfiles directory is itself a repo would have every
        unrelated project beneath it collapse into one store. That is worse
        than #33: #33 split one project's memory, this would MERGE two
        projects' decisions.

        Anchoring therefore only ever redirects into a root store that already
        exists — it never creates one. Nobody has worked at the umbrella root
        here, so there is no store there, and the two projects stay apart.
        """
        umbrella = tmp_path / "code"
        (umbrella / "projA").mkdir(parents=True)
        (umbrella / "projB").mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(umbrella)], check=True,
                       capture_output=True)
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")

        a = resolve_project_id(umbrella / "projA", cfg)
        b = resolve_project_id(umbrella / "projB", cfg)
        assert a != b
        assert a == hash_project_path(umbrella / "projA")

    def test_a_new_store_is_created_at_the_path_not_the_root(
        self, tmp_path: Path
    ) -> None:
        """With no store at the root, resolution is unchanged from before —
        the cwd's own hash. Anchoring redirects; it does not relocate."""
        root = self._repo(tmp_path)
        sub = root / "packages" / "core"
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        assert resolve_project_id(sub, cfg) == hash_project_path(sub)

    def test_an_existing_subdirectory_store_still_wins(self, tmp_path: Path) -> None:
        """Back-compat, and the reason root anchoring is not unconditional.
        Someone whose only store IS a subdirectory store must not appear to
        lose their memory on upgrade."""
        root = self._repo(tmp_path)
        sub = root / "packages" / "core"
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        cfg.projects_dir.mkdir(parents=True)
        sub_id = hash_project_path(sub)
        (cfg.projects_dir / f"{sub_id}.db").touch()

        assert resolve_project_id(sub, cfg) == sub_id

    def test_the_root_store_wins_over_a_subdirectory_store(self, tmp_path: Path) -> None:
        """The split-healing case: once the project's own store exists, a
        capture from a subdirectory goes THERE, so the fork stops. This is why
        the root check runs before the exact-path check — the reverse order
        would keep feeding whichever store the bug created first."""
        root = self._repo(tmp_path)
        sub = root / "packages" / "core"
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        cfg.projects_dir.mkdir(parents=True)
        root_id = hash_project_path(root.resolve())
        (cfg.projects_dir / f"{root_id}.db").touch()
        (cfg.projects_dir / f"{hash_project_path(sub)}.db").touch()

        assert resolve_project_id(sub, cfg) == root_id

    def test_explicit_identity_still_outranks_the_root(self, tmp_path: Path) -> None:
        root = self._repo(tmp_path)
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel",
                     project_identity="acme-api")
        assert (resolve_project_id(root / "packages" / "core", cfg)
                == hash_project_identity("acme-api"))

    def test_a_non_repo_path_is_unchanged(self, tmp_path: Path) -> None:
        """No work tree is the ordinary case for a project that simply is not
        a repo, and must behave exactly as before."""
        plain = tmp_path / "not_a_repo"
        plain.mkdir()
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        assert resolve_project_id(plain, cfg) == hash_project_path(plain)

    def test_two_sibling_repos_stay_separate(self, tmp_path: Path) -> None:
        """Anchoring must collapse subdirectories INTO a project, never
        collapse distinct projects into each other."""
        a, b = tmp_path / "a", tmp_path / "b"
        for r in (a, b):
            r.mkdir()
            subprocess.run(["git", "init", "-q", str(r)], check=True,
                           capture_output=True)
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        assert resolve_project_id(a, cfg) != resolve_project_id(b, cfg)

    def test_an_unreadable_tree_falls_back_to_the_path(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Fail-open: a filesystem error while walking up must not break
        resolution. Identity is on the capture path, so a permission error or
        an odd mount has to degrade to the old behaviour, not raise."""
        import cognikernel.storage.connection as conn_mod

        def boom(*_a, **_k):
            raise PermissionError("no access")

        monkeypatch.setattr(conn_mod.Path, "exists", boom)
        conn_mod._ROOT_CACHE.clear()
        try:
            plain = tmp_path / "somewhere"
            # project_root directly: patching Path.exists globally would also
            # break the store-existence checks in resolve_project_id, which
            # would prove nothing about the walk.
            assert conn_mod.project_root(plain) == plain.resolve()
        finally:
            conn_mod._ROOT_CACHE.clear()

    def test_a_dot_git_FILE_counts_as_a_root(self, tmp_path: Path) -> None:
        """Worktrees and submodules record `.git` as a FILE pointing elsewhere,
        not a directory. Matching only directories would silently treat every
        worktree as rootless."""
        import cognikernel.storage.connection as conn_mod

        root = tmp_path / "wt"
        (root / "pkg").mkdir(parents=True)
        (root / ".git").write_text(
            "gitdir: /elsewhere/.git/worktrees/wt", encoding="utf-8"
        )
        conn_mod._ROOT_CACHE.clear()
        try:
            assert conn_mod.project_root(root / "pkg") == root.resolve()
        finally:
            conn_mod._ROOT_CACHE.clear()

    def test_the_walk_stops_at_the_nearest_root(self, tmp_path: Path) -> None:
        """A repo inside a repo (a submodule checkout) belongs to the INNER
        one — the walk must stop at the first `.git` it meets, not run to the
        outermost."""
        import cognikernel.storage.connection as conn_mod

        outer = tmp_path / "outer"
        inner = outer / "vendor" / "inner"
        (inner / "src").mkdir(parents=True)
        (outer / ".git").mkdir()
        (inner / ".git").mkdir()
        conn_mod._ROOT_CACHE.clear()
        try:
            assert conn_mod.project_root(inner / "src") == inner.resolve()
        finally:
            conn_mod._ROOT_CACHE.clear()


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
