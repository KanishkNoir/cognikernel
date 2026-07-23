"""Tests for cognikernel.utils.paths — canonical path normalization (Stage C2)."""
from __future__ import annotations

from pathlib import Path

import pytest

from cognikernel.utils.paths import canonicalize_path, is_bare_basename


# ── empty inputs ─────────────────────────────────────────────────────────────


class TestEmptyInputs:
    def test_empty_string(self) -> None:
        assert canonicalize_path("") == ""

    def test_whitespace_only(self) -> None:
        assert canonicalize_path("   ") == ""

    def test_none(self) -> None:
        assert canonicalize_path(None) == ""

    def test_just_dot(self) -> None:
        assert canonicalize_path(".") == ""


# ── relative paths without project_root ──────────────────────────────────────


class TestRelativeNoRoot:
    def test_bare_basename_kept(self) -> None:
        assert canonicalize_path("env.py") == "env.py"

    def test_directory_path_kept(self) -> None:
        assert canonicalize_path("alembic/env.py") == "alembic/env.py"

    def test_deep_path(self) -> None:
        assert canonicalize_path("a/b/c/d.py") == "a/b/c/d.py"

    def test_backslashes_normalized(self) -> None:
        assert canonicalize_path("alembic\\env.py") == "alembic/env.py"

    def test_mixed_separators(self) -> None:
        assert canonicalize_path("alembic\\versions/0001.py") == "alembic/versions/0001.py"

    def test_leading_dot_slash_stripped(self) -> None:
        assert canonicalize_path("./app/main.py") == "app/main.py"

    def test_double_leading_dot_slash_stripped(self) -> None:
        assert canonicalize_path("./././app.py") == "app.py"

    def test_double_slash_collapsed(self) -> None:
        assert canonicalize_path("app//main.py") == "app/main.py"

    def test_trailing_slash_stripped(self) -> None:
        assert canonicalize_path("app/main.py/") == "app/main.py"

    def test_whitespace_trimmed(self) -> None:
        assert canonicalize_path("  app/main.py  ") == "app/main.py"


# ── parent-directory escape attempts (rejected) ──────────────────────────────


class TestEscapeRejected:
    def test_leading_dot_dot(self) -> None:
        assert canonicalize_path("../escape.py") == ""

    def test_embedded_dot_dot(self) -> None:
        assert canonicalize_path("app/../escape.py") == ""

    def test_bare_dot_dot(self) -> None:
        assert canonicalize_path("..") == ""

    def test_dot_dot_slash_in_middle(self) -> None:
        assert canonicalize_path("a/../../b.py") == ""


# ── absolute paths without project_root → '' ─────────────────────────────────


class TestAbsoluteNoRoot:
    def test_posix_absolute(self) -> None:
        assert canonicalize_path("/abs/no/root.py") == ""

    def test_windows_absolute(self) -> None:
        assert canonicalize_path("C:/abs/no/root.py") == ""

    def test_windows_with_backslashes(self) -> None:
        assert canonicalize_path("C:\\abs\\no\\root.py") == ""


# ── absolute paths with project_root ────────────────────────────────────────


class TestAbsoluteWithRoot:
    def test_inside_project_root(self, tmp_path: Path) -> None:
        target = tmp_path / "app" / "main.py"
        target.parent.mkdir(parents=True)
        target.write_text("x", encoding="utf-8")

        assert canonicalize_path(str(target), str(tmp_path)) == "app/main.py"

    def test_inside_project_root_with_backslashes(self, tmp_path: Path) -> None:
        target = tmp_path / "app" / "main.py"
        target.parent.mkdir(parents=True)
        target.write_text("x", encoding="utf-8")

        result = canonicalize_path(str(target).replace("/", "\\"), str(tmp_path))
        assert result == "app/main.py"

    def test_outside_project_root(self, tmp_path: Path) -> None:
        elsewhere = tmp_path.parent / "elsewhere.py"
        elsewhere.write_text("x", encoding="utf-8")

        assert canonicalize_path(str(elsewhere), str(tmp_path)) == ""

    def test_nonexistent_path_inside_root_uses_lexical_fallback(
        self, tmp_path: Path,
    ) -> None:
        """Even when target doesn't exist on disk, we can still relativize lexically."""
        fake = tmp_path / "not_yet_written.py"
        # Don't create the file
        assert canonicalize_path(str(fake), str(tmp_path)) == "not_yet_written.py"

    def test_relative_input_passes_through_with_root_given(
        self, tmp_path: Path,
    ) -> None:
        """A relative input is treated as already-canonical regardless of root."""
        assert canonicalize_path("app/main.py", str(tmp_path)) == "app/main.py"


# ── is_bare_basename ─────────────────────────────────────────────────────────


class TestIsBareBasename:
    def test_simple_filename(self) -> None:
        assert is_bare_basename("env.py") is True

    def test_path_with_one_dir(self) -> None:
        assert is_bare_basename("alembic/env.py") is False

    def test_deep_path(self) -> None:
        assert is_bare_basename("a/b/c.py") is False

    def test_empty_string(self) -> None:
        # Empty is a special signal; not a bare basename.
        assert is_bare_basename("") is False

    def test_no_extension(self) -> None:
        assert is_bare_basename("Makefile") is True

    def test_path_starting_with_slash_unusual(self) -> None:
        """A leading-slash string would not normally reach this function
        (canonicalize strips/rejects them) but exercise the predicate."""
        assert is_bare_basename("/etc/passwd") is False


# ── integration: canonicalize then is_bare_basename ──────────────────────────


class TestRoundTrip:
    def test_canonicalize_then_drop_bare(self) -> None:
        """The typical projection-time flow: canonicalize, then check bare."""
        # Full path survives.
        canonical = canonicalize_path("alembic\\env.py")
        assert canonical == "alembic/env.py"
        assert is_bare_basename(canonical) is False

        # Bare survives canonicalization but is bare.
        canonical2 = canonicalize_path("env.py")
        assert canonical2 == "env.py"
        assert is_bare_basename(canonical2) is True

        # Empty (rejected absolute) — caller doesn't even check is_bare.
        rejected = canonicalize_path("/etc/passwd")
        assert rejected == ""
