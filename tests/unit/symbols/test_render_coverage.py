"""Tests for the B-2 / B-3 skeleton renderer additions.

Coverage header sourced from symbol_files.CoverageStats, and per-Python-file
import hints derived from the path.
"""
from __future__ import annotations

import pytest

from memlora.storage.symbol_files import CoverageStats, RefreshInfo
from memlora.symbols.projection import SkeletonClass, SkeletonEntry, SkeletonMethod
from memlora.symbols.render import (
    _path_to_module,
    _render_import_hint,
    render_skeleton_section,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _entry(
    path: str = "app/main.py",
    imports=None,
    classes=None,
    functions=None,
) -> SkeletonEntry:
    return SkeletonEntry(
        path=path,
        imports=imports or [],
        classes=classes or [],
        functions=functions or [],
    )


def _cls(name: str, methods=None) -> SkeletonClass:
    return SkeletonClass(name=name, bases="", fields="", methods=methods or [])


def _fn(name: str) -> SkeletonMethod:
    return SkeletonMethod(name=name, signature="()", return_type="")


# ── B-2: coverage header ─────────────────────────────────────────────────────


class TestCoverageHeader:
    def test_coverage_line_includes_all_counts(self) -> None:
        cov = CoverageStats(scanned=17, with_symbols=14, parse_errors=2, ignored=1, pending=0)
        out = render_skeleton_section([_entry()], coverage=cov)
        assert "17 files scanned" in out
        assert "14 with public symbols listed" in out
        assert "2 parse errors" in out
        assert "1 ignored" in out

    def test_coverage_line_omits_zero_categories(self) -> None:
        """Empty categories shouldn't clutter the header."""
        cov = CoverageStats(scanned=10, with_symbols=8, parse_errors=0, ignored=0, pending=0)
        out = render_skeleton_section([_entry()], coverage=cov)
        assert "parse errors" not in out
        assert "ignored" not in out

    def test_no_coverage_argument_gives_bare_header(self) -> None:
        """Back-compat: callers that don't supply coverage get the simple header."""
        out = render_skeleton_section([_entry()])
        # The header is the first line; nothing after it before the entries.
        first_two = out.split("\n", 2)[:2]
        assert first_two[0] == "### Codebase skeleton"
        # Second line is the blank separator before the first entry's block.
        # No coverage prose smuggled in.
        assert "scanned" not in first_two[1]
        assert "Coverage" not in first_two[1]


class TestRefreshLine:
    def test_refresh_line_renders_session_action_path(self) -> None:
        refresh = RefreshInfo(
            path="app/core/security.py",
            refreshed_in_session="abcdef1234567890",
            last_action="Edit",
            refreshed_at=1_700_000_000_000,
        )
        out = render_skeleton_section([_entry()], refresh=refresh)
        assert "Last refreshed: session abcdef123456, after Edit of app/core/security.py" in out

    def test_refresh_line_omitted_when_never_refreshed(self) -> None:
        refresh = RefreshInfo(
            path="", refreshed_in_session="", last_action="", refreshed_at=0,
        )
        out = render_skeleton_section([_entry()], refresh=refresh)
        assert "Last refreshed" not in out

    def test_refresh_handles_short_session_id(self) -> None:
        refresh = RefreshInfo(
            path="x.py", refreshed_in_session="abc", last_action="Write",
            refreshed_at=1_700_000_000_000,
        )
        out = render_skeleton_section([_entry()], refresh=refresh)
        assert "session abc, after Write of x.py" in out


# ── B-3: import hints ────────────────────────────────────────────────────────


class TestImportHints:
    def test_python_entry_with_public_class_gets_hint(self) -> None:
        entry = _entry(path="app/core/security.py", classes=[_cls("Argon2Hasher")])
        out = render_skeleton_section([entry])
        assert "Import: from app.core.security import Argon2Hasher" in out

    def test_python_entry_with_public_function_gets_hint(self) -> None:
        entry = _entry(path="app/core/security.py", functions=[_fn("hash_password")])
        out = render_skeleton_section([entry])
        assert "Import: from app.core.security import hash_password" in out

    def test_classes_and_functions_combined_in_hint(self) -> None:
        entry = _entry(
            path="app/core/security.py",
            classes=[_cls("Hasher")],
            functions=[_fn("hash_password"), _fn("verify_password")],
        )
        out = render_skeleton_section([entry])
        # Classes come first, then functions; order matches public_names assembly.
        assert "Import: from app.core.security import Hasher, hash_password, verify_password" in out

    def test_private_symbols_excluded_from_hint(self) -> None:
        entry = _entry(
            path="app/core/util.py",
            classes=[_cls("_PrivateThing"), _cls("PublicThing")],
            functions=[_fn("_helper"), _fn("public_fn")],
        )
        out = render_skeleton_section([entry])
        assert "Import: from app.core.util import PublicThing, public_fn" in out
        assert "_PrivateThing" not in out.split("Import:")[1].split("\n")[0]
        assert "_helper" not in out.split("Import:")[1].split("\n")[0]

    def test_non_python_files_have_no_import_hint(self) -> None:
        entry = _entry(
            path="frontend/src/App.tsx",
            classes=[_cls("App")],
        )
        out = render_skeleton_section([entry])
        assert "Import:" not in out

    def test_python_file_with_no_public_symbols_omits_hint(self) -> None:
        entry = _entry(path="app/__init__.py")  # no classes / functions
        out = render_skeleton_section([entry])
        assert "Import:" not in out

    def test_python_file_with_only_private_symbols_omits_hint(self) -> None:
        entry = _entry(path="app/_internal.py", functions=[_fn("_helper")])
        out = render_skeleton_section([entry])
        assert "Import:" not in out


# ── _path_to_module helper ───────────────────────────────────────────────────


class TestPathToModule:
    def test_simple_module(self) -> None:
        assert _path_to_module("app/main.py") == "app.main"

    def test_nested_module(self) -> None:
        assert _path_to_module("app/core/security.py") == "app.core.security"

    def test_top_level_file(self) -> None:
        assert _path_to_module("main.py") == "main"

    def test_init_py_collapses_to_package(self) -> None:
        assert _path_to_module("app/core/__init__.py") == "app.core"

    def test_root_init_py_empty(self) -> None:
        """A bare top-level `__init__.py` (no package above) → no module name."""
        assert _path_to_module("__init__.py") == ""

    def test_non_python_returns_empty(self) -> None:
        assert _path_to_module("frontend/App.tsx") == ""

    def test_invalid_identifier_segment_returns_empty(self) -> None:
        """Defense: `foo-bar` is not a valid Python identifier → no module."""
        assert _path_to_module("foo-bar/m.py") == ""

    def test_digit_leading_segment_returns_empty(self) -> None:
        assert _path_to_module("123/m.py") == ""


# ── full B-2 + B-3 combined rendering ────────────────────────────────────────


class TestCombinedBehavior:
    def test_coverage_refresh_and_import_hints_all_present(self) -> None:
        cov = CoverageStats(scanned=3, with_symbols=2, parse_errors=1, ignored=0, pending=0)
        refresh = RefreshInfo(
            path="app/main.py", refreshed_in_session="sess-12345678",
            last_action="Write", refreshed_at=1_700_000_000_000,
        )
        entry = _entry(
            path="app/main.py",
            classes=[_cls("FastAPI")],
            functions=[_fn("lifespan")],
        )

        out = render_skeleton_section([entry], coverage=cov, refresh=refresh)

        # Headline elements all visible.
        assert "### Codebase skeleton" in out
        assert "3 files scanned" in out
        assert "1 parse errors" in out
        # Session id is truncated to 12 chars in the header for compactness.
        assert "Last refreshed: session sess-1234567" in out
        assert "Import: from app.main import FastAPI, lifespan" in out
