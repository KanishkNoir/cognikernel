"""Unit tests for the grep result cache."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from memlora.config import Config
from memlora.integration.session import init_project
from memlora.storage.connection import get_connection, get_db_path, hash_project_path
from memlora.storage.grep_cache import (
    cache_key,
    invalidate_project_cache,
    lookup_grep_result,
    store_grep_result,
)


@pytest.fixture
def db_conn(tmp_path: Path):
    cfg = Config(memlora_dir=tmp_path / "memlora")
    project_path = tmp_path / "proj"
    project_path.mkdir()
    init_project(project_path, config=cfg)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(cfg, project_id)
    with get_connection(db_path) as conn:
        yield conn, project_id


class TestCacheKey:
    def test_same_inputs_same_key(self) -> None:
        k1 = cache_key("class Foo", "src/", "*.py")
        k2 = cache_key("class Foo", "src/", "*.py")
        assert k1 == k2

    def test_different_pattern_different_key(self) -> None:
        assert cache_key("class Foo", "", "") != cache_key("class Bar", "", "")

    def test_different_path_different_key(self) -> None:
        assert cache_key("foo", "src/", "") != cache_key("foo", "lib/", "")

    def test_different_glob_different_key(self) -> None:
        assert cache_key("foo", "", "*.py") != cache_key("foo", "", "*.ts")


class TestLookupGrep:
    def test_miss_returns_none(self, db_conn) -> None:
        conn, project_id = db_conn
        result = lookup_grep_result(conn, project_id, "missing", "", "")
        assert result is None

    def test_hit_returns_cached_text(self, db_conn) -> None:
        conn, project_id = db_conn
        store_grep_result(conn, project_id, "def foo", "src/", "*.py", "src/models.py:5:def foo():")
        result = lookup_grep_result(conn, project_id, "def foo", "src/", "*.py")
        assert result == "src/models.py:5:def foo():"

    def test_hit_increments_hit_count(self, db_conn) -> None:
        conn, project_id = db_conn
        store_grep_result(conn, project_id, "pattern", "", "", "match line")
        lookup_grep_result(conn, project_id, "pattern", "", "")
        lookup_grep_result(conn, project_id, "pattern", "", "")
        hit_count = conn.execute(
            "SELECT hit_count FROM grep_cache WHERE project_id = ? AND pattern = 'pattern'",
            (project_id,),
        ).fetchone()[0]
        assert hit_count == 2

    def test_empty_result_cached(self, db_conn) -> None:
        conn, project_id = db_conn
        store_grep_result(conn, project_id, "rare", "", "", "")
        result = lookup_grep_result(conn, project_id, "rare", "", "")
        assert result == ""

    def test_different_project_isolated(self, db_conn, tmp_path: Path) -> None:
        conn, project_id = db_conn
        store_grep_result(conn, project_id, "shared", "", "", "result A")
        # Different project_id → miss
        result = lookup_grep_result(conn, "other-project-id", "shared", "", "")
        assert result is None


class TestStoreGrep:
    def test_upsert_replaces_old_result(self, db_conn) -> None:
        conn, project_id = db_conn
        store_grep_result(conn, project_id, "foo", "", "", "old result")
        store_grep_result(conn, project_id, "foo", "", "", "new result")
        result = lookup_grep_result(conn, project_id, "foo", "", "")
        assert result == "new result"

    def test_upsert_resets_hit_count(self, db_conn) -> None:
        conn, project_id = db_conn
        store_grep_result(conn, project_id, "foo", "", "", "first")
        lookup_grep_result(conn, project_id, "foo", "", "")
        lookup_grep_result(conn, project_id, "foo", "", "")
        store_grep_result(conn, project_id, "foo", "", "", "updated")  # re-store resets count
        hit_count = conn.execute(
            "SELECT hit_count FROM grep_cache WHERE project_id = ? AND pattern = 'foo'",
            (project_id,),
        ).fetchone()[0]
        assert hit_count == 0

    def test_sets_cached_at(self, db_conn) -> None:
        conn, project_id = db_conn
        before = int(time.time() * 1000)
        store_grep_result(conn, project_id, "ts", "", "", "result")
        after = int(time.time() * 1000)
        ts = conn.execute(
            "SELECT cached_at FROM grep_cache WHERE project_id = ? AND pattern = 'ts'",
            (project_id,),
        ).fetchone()[0]
        assert before <= ts <= after

    def test_independent_cache_key_separation(self, db_conn) -> None:
        conn, project_id = db_conn
        store_grep_result(conn, project_id, "foo", "src/", "*.py", "python result")
        store_grep_result(conn, project_id, "foo", "src/", "*.ts", "ts result")
        assert lookup_grep_result(conn, project_id, "foo", "src/", "*.py") == "python result"
        assert lookup_grep_result(conn, project_id, "foo", "src/", "*.ts") == "ts result"


class TestInvalidateCache:
    def test_clears_all_entries_for_project(self, db_conn) -> None:
        conn, project_id = db_conn
        store_grep_result(conn, project_id, "a", "", "", "r1")
        store_grep_result(conn, project_id, "b", "", "", "r2")
        store_grep_result(conn, project_id, "c", "x/", "", "r3")
        count = invalidate_project_cache(conn, project_id)
        assert count == 3
        assert lookup_grep_result(conn, project_id, "a", "", "") is None

    def test_only_affects_target_project(self, db_conn) -> None:
        conn, project_id = db_conn
        store_grep_result(conn, project_id, "pat", "", "", "result")
        # Invalidate a different project
        invalidate_project_cache(conn, "other-id")
        # Original project still has its entry
        assert lookup_grep_result(conn, project_id, "pat", "", "") == "result"

    def test_empty_cache_returns_zero(self, db_conn) -> None:
        conn, project_id = db_conn
        count = invalidate_project_cache(conn, project_id)
        assert count == 0


class TestPathAwareInvalidation:
    def test_path_match_invalidates_row(self, db_conn) -> None:
        conn, project_id = db_conn
        # Row with path_filter matching the changed file
        store_grep_result(conn, project_id, "def foo", "src/models.py", "", "match")
        count = invalidate_project_cache(conn, project_id, changed_path="src/models.py")
        assert count == 1
        assert lookup_grep_result(conn, project_id, "def foo", "src/models.py", "") is None

    def test_path_mismatch_preserves_row(self, db_conn) -> None:
        conn, project_id = db_conn
        # Changed file is config.py but grep was for models.py
        store_grep_result(conn, project_id, "def foo", "src/models.py", "", "match")
        count = invalidate_project_cache(conn, project_id, changed_path="src/config.py")
        assert count == 0
        assert lookup_grep_result(conn, project_id, "def foo", "src/models.py", "") == "match"

    def test_glob_match_invalidates_row(self, db_conn) -> None:
        conn, project_id = db_conn
        # Row with glob_filter *.py; changed file is a .py file
        store_grep_result(conn, project_id, "class Foo", "", "*.py", "match")
        count = invalidate_project_cache(conn, project_id, changed_path="src/models.py")
        assert count == 1

    def test_glob_mismatch_preserves_row(self, db_conn) -> None:
        conn, project_id = db_conn
        # Row with glob_filter *.ts; changed file is a .py file
        store_grep_result(conn, project_id, "interface Foo", "", "*.ts", "match")
        count = invalidate_project_cache(conn, project_id, changed_path="src/models.py")
        assert count == 0
        assert lookup_grep_result(conn, project_id, "interface Foo", "", "*.ts") == "match"

    def test_no_filter_row_always_invalidated(self, db_conn) -> None:
        # A whole-project grep (no path, no glob) is always invalidated on any file change
        conn, project_id = db_conn
        store_grep_result(conn, project_id, "TODO", "", "", "matches")
        count = invalidate_project_cache(conn, project_id, changed_path="src/anything.py")
        assert count == 1
        assert lookup_grep_result(conn, project_id, "TODO", "", "") is None

    def test_mixed_rows_selective_invalidation(self, db_conn) -> None:
        conn, project_id = db_conn
        # Whole-project grep → always invalidated
        store_grep_result(conn, project_id, "TODO", "", "", "whole project")
        # Path-specific grep matching changed file → invalidated
        store_grep_result(conn, project_id, "def foo", "src/models.py", "", "models")
        # Path-specific grep NOT matching → preserved
        store_grep_result(conn, project_id, "class Bar", "src/views.py", "", "views")
        # Glob matching changed file → invalidated
        store_grep_result(conn, project_id, "import", "", "*.py", "all py")

        count = invalidate_project_cache(conn, project_id, changed_path="src/models.py")
        assert count == 3  # TODO, def foo, import (*.py)

        # views.py grep is still cached
        assert lookup_grep_result(conn, project_id, "class Bar", "src/views.py", "") == "views"

    def test_directory_path_filter_invalidated_by_file_inside(self, db_conn) -> None:
        conn, project_id = db_conn
        # Grep scoped to a directory (path_filter = "src/") should be invalidated
        # when any file inside that directory changes
        store_grep_result(conn, project_id, "def foo", "src/", "", "match")
        count = invalidate_project_cache(conn, project_id, changed_path="src/models.py")
        assert count == 1
