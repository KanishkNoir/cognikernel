"""Deterministic injection-block cost report — the meter for prefix shrinking.

The whole-session telemetry in `ingest.py` needs live Claude Code sessions to
produce data. This module measures the thing the architecture curriculum
actually changes — the rendered Session-context block — by tiktoken token
count, broken down by markdown section. It needs no live data, so before/after
diffs across Units 3/5/6 are fully reproducible.

`total_tokens` is the authoritative number (one tiktoken pass over the whole
block). The per-section breakdown is approximate: tiktoken is not additive
across split boundaries, so the section sum will not exactly equal the total.
Use sections to see *where* tokens go; use `total_tokens` to gate the budget.
"""
from __future__ import annotations

from typing import Any

from cognikernel.injection.template import count_tokens_accurate

_PREAMBLE = "(preamble)"


def section_token_report(block: str) -> dict[str, Any]:
    """Break a rendered block into sections by `##`/`###` headers, count tokens.

    Returns {"sections": {section_name: tokens}, "total_tokens": int}. Content
    before the first header is bucketed under `(preamble)`. Empty sections are
    omitted.
    """
    sections: list[tuple[str, list[str]]] = []
    current_name = _PREAMBLE
    current: list[str] = []

    for line in block.split("\n"):
        stripped = line.lstrip()
        if stripped.startswith("### ") or stripped.startswith("## "):
            sections.append((current_name, current))
            current_name = stripped.lstrip("#").strip()
            current = [line]
        else:
            current.append(line)
    sections.append((current_name, current))

    per_section: dict[str, int] = {}
    for name, body in sections:
        text = "\n".join(body)
        if not text.strip():
            continue
        # Later sections with a duplicate header name accumulate rather than clobber.
        per_section[name] = per_section.get(name, 0) + count_tokens_accurate(text)

    return {"sections": per_section, "total_tokens": count_tokens_accurate(block)}


def render_cost_report(project_path: str, config: Any = None) -> dict[str, Any]:
    """Render the live injection block for a project and report its token cost."""
    from cognikernel.integration.session import render_state

    block = render_state(project_path, config=config)
    report = section_token_report(block)
    report["project_path"] = str(project_path)
    return report


def diff_reports(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare two section_token_report outputs (e.g., pre/post a change).

    Returns total before/after/delta and a per-section delta (added sections
    count as +tokens, removed sections as −tokens).
    """
    b = before.get("sections", {})
    a = after.get("sections", {})
    names = sorted(set(b) | set(a))
    section_delta = {n: a.get(n, 0) - b.get(n, 0) for n in names}
    return {
        "total_before": before.get("total_tokens", 0),
        "total_after": after.get("total_tokens", 0),
        "total_delta": after.get("total_tokens", 0) - before.get("total_tokens", 0),
        "section_delta": section_delta,
    }
