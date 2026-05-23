"""Render symbol skeleton entries into compact injection block text."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlora.symbols.projection import SkeletonEntry, SkeletonMethod


def render_skeleton_section(entries: list["SkeletonEntry"]) -> str:
    """Return '### Codebase skeleton\\n...' or '' if entries is empty."""
    if not entries:
        return ""
    lines = ["### Codebase skeleton"]
    for entry in sorted(entries, key=lambda e: e.path):
        rendered = _render_entry(entry)
        if rendered.strip():
            lines.append("")
            lines.append(rendered)
    return "\n".join(lines)


def _render_entry(entry: "SkeletonEntry") -> str:
    """Render a single file's skeleton block."""
    lines: list[str] = []
    lines.append(_render_file_header(entry))

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
