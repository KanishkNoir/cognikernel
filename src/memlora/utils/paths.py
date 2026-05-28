"""Canonical-path utility used across extraction, projection, and hook layers.

A single source of truth for the question "given some path string, what is the
forward-slash relative path I should store/index it under?" — eliminates the
C4 duplicate-keys bug (`env.py` and `alembic/env.py` both indexed) by funneling
every caller through the same normalization rules.

Rules (in order):
  1. Empty / whitespace input              → ''  (caller must check)
  2. Backslashes                           → forward slashes
  3. Repeated slashes (`a//b`)             → single slash (`a/b`)
  4. Leading `./` and `.//`                → stripped
  5. Trailing slash                        → stripped
  6. When `project_root` is provided:
       - Absolute paths under project_root → relativized
       - Absolute paths outside            → ''
  7. When `project_root` is None:
       - Absolute paths                    → ''  (cannot resolve; caller error)
       - Relative paths                    → returned normalized

The empty string is the universal failure signal. Callers branch on `if path:`.

There is no filesystem I/O here: `canonicalize_path` is a pure function so
extraction code can call it without touching the disk.
"""
from __future__ import annotations

import posixpath
from pathlib import Path, PurePath


def canonicalize_path(
    raw: str | None,
    project_root: str | Path | None = None,
) -> str:
    """Return a canonical forward-slash relative path, or '' on failure.

    See module docstring for the full rule set.
    """
    if not raw:
        return ""

    s = str(raw).strip()
    if not s:
        return ""

    # Normalize separators first so subsequent posixpath ops behave consistently.
    s = s.replace("\\", "/")
    # Collapse repeated slashes without using posixpath.normpath (which also
    # touches "..", and we want to preserve those as a signal of caller error).
    while "//" in s:
        s = s.replace("//", "/")

    if s == ".":
        return ""

    # Determine "absoluteness" using the forward-slash form so Windows drive
    # letters (C:/...) and POSIX absolute paths (/foo/...) both resolve.
    is_absolute = _looks_absolute(s)

    if is_absolute:
        if project_root is None:
            return ""
        rel = _relativize_under(s, str(project_root))
        return rel  # may be '' if outside project_root

    # Relative path → strip leading "./" segments and trailing slash, return.
    while s.startswith("./"):
        s = s[2:]
    if s.endswith("/") and len(s) > 1:
        s = s.rstrip("/")
    if s == "":
        return ""
    # Reject ".." escape attempts — paths attempting to climb out of the project
    # are caller bugs.
    if s.startswith("../") or "/../" in s or s == "..":
        return ""
    return s


def is_bare_basename(path: str) -> bool:
    """Return True if `path` has no directory component (just a filename).

    These are dropped from `component_map` during projection rebuild — they're
    extractor noise that conflicts with the prefixed canonical form (C4).

    Empty input is NOT a bare basename — callers should check `path` first.
    """
    if not path:
        return False
    return "/" not in path


# ── internals ────────────────────────────────────────────────────────────────


def _looks_absolute(forward_slash_path: str) -> bool:
    """True for POSIX `/foo`, Windows `C:/foo`, or UNC `//server/share`."""
    if not forward_slash_path:
        return False
    if forward_slash_path.startswith("/"):
        return True
    # Drive letter (single-char) followed by colon: "C:foo" or "C:/foo".
    if len(forward_slash_path) >= 2 and forward_slash_path[1] == ":":
        return True
    return False


def _relativize_under(forward_path: str, project_root: str) -> str:
    """Return `forward_path` relative to `project_root`, or '' if outside.

    Uses Path.resolve() — handles symlinks and mixed-case drive letters on
    Windows. The PurePath fallback handles cases where resolution fails
    (non-existent paths) but the lexical comparison still makes sense.
    """
    try:
        abs_target = Path(forward_path).resolve()
        abs_root = Path(project_root).resolve()
        rel = abs_target.relative_to(abs_root)
        return str(rel).replace("\\", "/")
    except (ValueError, OSError):
        pass

    # Lexical fallback for paths that don't exist on disk yet (tests, etc.)
    target = PurePath(forward_path)
    root = PurePath(project_root.replace("\\", "/"))
    try:
        rel = target.relative_to(root)
        return str(rel).replace("\\", "/")
    except ValueError:
        return ""
