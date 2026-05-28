"""Render symbol skeleton entries into compact injection block text.

Phase B-2 adds a truthful coverage header sourced from `symbol_files`:
  Coverage: N scanned · M with public symbols listed · K parse errors · J ignored.
  Last refreshed: session <id>, after <Write|Edit|scan> of <path>.

Phase B-3 adds per-entry Python import hints derived from the file's path:
  app/core/security.py
    Import: from app.core.security import hash_password, verify_password
    .hash_password(plain:str)→str
    ...
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlora.storage.symbol_files import CoverageStats, RefreshInfo
    from memlora.symbols.projection import SkeletonEntry, SkeletonMethod


def render_skeleton_section(
    entries: list["SkeletonEntry"],
    *,
    coverage: "CoverageStats | None" = None,
    refresh: "RefreshInfo | None" = None,
) -> str:
    """Return '### Codebase skeleton\\n...' or '' if entries is empty.

    When `coverage` / `refresh` are provided (typically from
    `memlora.storage.symbol_files`), the header is augmented with truthful
    scan statistics and a "last refreshed" provenance line. Otherwise the
    header is the bare title — back-compat with callers that don't have
    symbol_files data yet.
    """
    if not entries:
        return ""

    lines = ["### Codebase skeleton"]

    if coverage is not None:
        lines.append(_render_coverage_line(coverage))
    if refresh is not None and refresh.refreshed_at > 0:
        lines.append(_render_refresh_line(refresh))

    for entry in sorted(entries, key=lambda e: e.path):
        rendered = _render_entry(entry)
        if rendered.strip():
            lines.append("")
            lines.append(rendered)
    return "\n".join(lines)


# ── coverage / freshness header (B-2) ────────────────────────────────────────


def _render_coverage_line(coverage: "CoverageStats") -> str:
    """Truthful one-line header per symbol_files counts.

    Skips empty fragments so the line stays tight when only some categories
    are populated."""
    fragments = [f"{coverage.scanned} files scanned"]
    fragments.append(f"{coverage.with_symbols} with public symbols listed")
    if coverage.parse_errors:
        fragments.append(f"{coverage.parse_errors} parse errors")
    if coverage.ignored:
        fragments.append(f"{coverage.ignored} ignored")
    return f"Coverage: {' · '.join(fragments)}."


def _render_refresh_line(refresh: "RefreshInfo") -> str:
    sess_short = (refresh.refreshed_in_session or "")[:12] or "?"
    action = refresh.last_action or "scan"
    path = refresh.path or "?"
    return f"Last refreshed: session {sess_short}, after {action} of {path}."


# ── per-entry rendering ──────────────────────────────────────────────────────


def _render_entry(entry: "SkeletonEntry") -> str:
    """Render a single file's skeleton block (header → import hint → members)."""
    lines: list[str] = []
    lines.append(_render_file_header(entry))

    import_line = _render_import_hint(entry)
    if import_line:
        lines.append(import_line)

    for cls in entry.classes:
        if cls.bases:
            class_line = f"  {cls.name}({cls.bases})"
        else:
            class_line = f"  {cls.name}"
        if cls.fields:
            class_line += f": {cls.fields}"
        lines.append(class_line)

        if cls.methods:
            method_strs = [_render_method(m) for m in cls.methods]
            joined = " | ".join(method_strs)
            if len(joined) <= 100:
                lines.append(f"    {joined}")
            else:
                for ms in method_strs:
                    lines.append(f"    {ms}")

    for fn in entry.functions:
        lines.append(f"  {_render_method(fn)}")

    return "\n".join(lines)


def _render_file_header(entry: "SkeletonEntry") -> str:
    if entry.imports:
        return f"{entry.path} → {', '.join(entry.imports)}"
    return entry.path


def _render_method(m: "SkeletonMethod") -> str:
    ret = f"→{m.return_type}" if m.return_type else ""
    if m.route_info:
        return f"{m.route_info} .{m.name}{m.signature}{ret}"
    return f".{m.name}{m.signature}{ret}"


# ── import hints (B-3) ───────────────────────────────────────────────────────


def _render_import_hint(entry: "SkeletonEntry") -> str:
    """Return 'Import: from <module> import <names>' for Python files, '' otherwise.

    Public symbols only — classes + top-level functions whose name doesn't
    start with underscore. Files with no public symbols (e.g., empty
    `__init__.py`) get no hint.
    """
    if not entry.path.endswith(".py"):
        return ""

    module = _path_to_module(entry.path)
    if not module:
        return ""

    public_names: list[str] = []
    for cls in entry.classes:
        if not cls.name.startswith("_"):
            public_names.append(cls.name)
    for fn in entry.functions:
        if not fn.name.startswith("_"):
            public_names.append(fn.name)

    if not public_names:
        return ""

    return f"  Import: from {module} import {', '.join(public_names)}"


def _path_to_module(path: str) -> str:
    """`app/core/security.py` → `app.core.security`. Returns '' on failure."""
    if not path.endswith(".py"):
        return ""
    stem = path[:-3]  # strip .py
    if not stem or stem.endswith("/"):
        return ""
    # `__init__.py` files: collapse to the package name (drop trailing `__init__`).
    if stem == "__init__":
        # Top-level __init__.py — no package above to name.
        return ""
    if stem.endswith("/__init__"):
        stem = stem[: -len("/__init__")]
        if not stem:
            return ""
    parts = [seg for seg in stem.split("/") if seg]
    if not parts:
        return ""
    # Reject parts that aren't valid Python identifiers (defensive — caller
    # should already have canonicalized paths but extraction noise could leak).
    if not all(_is_valid_identifier(p) for p in parts):
        return ""
    return ".".join(parts)


def _is_valid_identifier(s: str) -> bool:
    return bool(s) and s.isidentifier()
