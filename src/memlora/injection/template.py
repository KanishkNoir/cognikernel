"""Injection template engine — renders selected events into a system prompt block.

Eight canonical sections, fixed order:
  1. Header
  2. Hard constraints       (primacy zone — never token-cut)
  3. Active thread          (never token-cut)
  4. Most active files      (only rendered when Codebase skeleton is present)
  5. Do-not-retry graveyard (never token-cut)
  6. Component state        (reference material)
  7. Key decisions
  8. Codebase skeleton      (recency zone — Symbol Graph, AST-derived)
  9. Summary                (recency anchor)

Hard constraints and graveyard are sorted by content_hash (not weight) so that
the rendered prefix is stable across sessions, enabling Anthropic prompt-cache hits.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from memlora.config import SectionBudgets
    from memlora.storage.events import Event

_log = logging.getLogger("memlora.injection")


@dataclass
class InjectionContext:
    project_name: str
    session_number: int
    total_sessions: int
    state_version: int
    hard_constraints: list[Event]
    graveyard: list[Event]
    components: list[Event]
    decisions: list[Event]
    active_threads: list[Event]
    summary_text: str
    token_budget: int = 1000
    hot_files: list[tuple[str, int, str]] = field(default_factory=list)
    skeleton: list = field(default_factory=list)  # list[SkeletonEntry]
    ckl_mode: bool = False
    ckl_v2: bool = False
    section_budgets: SectionBudgets | None = None


def render_injection(ctx: InjectionContext) -> str:
    """Render the full block. Sections with no items are omitted.

    Section order:
      1. Header
      2. Hard constraints       ← primacy zone; never token-cut
      3. Active thread          ← never token-cut
      4. Most active files      ← only when skeleton is present (points Claude to skeleton)
      5. Do-not-retry graveyard ← never token-cut
      6. Component state        ← reference material
      7. Key decisions
      8. Codebase skeleton      ← recency zone; AST-derived, Symbol Graph
      9. Summary                ← recency anchor
    """
    from memlora.symbols.render import render_skeleton_section
    has_skeleton = bool(ctx.skeleton)
    sb = ctx.section_budgets

    hard = ctx.hard_constraints
    grave = ctx.graveyard
    comps = ctx.components
    decs = ctx.decisions

    if sb is not None:
        hard = _enforce_section_budget(
            hard,
            lambda items: _render_hard_constraints(
                items, ckl_mode=ctx.ckl_mode, ckl_v2=ctx.ckl_v2
            ),
            sb.hard_constraints,
        )
        grave = _enforce_section_budget(
            grave,
            lambda items: _render_graveyard(
                items, ckl_mode=ctx.ckl_mode, ckl_v2=ctx.ckl_v2
            ),
            sb.graveyard,
        )
        comps = _enforce_section_budget(comps, _render_components, sb.components)
        decs = _enforce_section_budget(
            decs,
            lambda items: _render_decisions(
                items, ckl_mode=ctx.ckl_mode, ckl_v2=ctx.ckl_v2
            ),
            sb.decisions,
        )

    sections = [
        _render_header(ctx),
        _render_hard_constraints(hard, ckl_mode=ctx.ckl_mode, ckl_v2=ctx.ckl_v2),
        _render_active_thread(ctx.active_threads),
        _render_hot_files(ctx.hot_files, has_skeleton=has_skeleton),
        _render_graveyard(grave, ckl_mode=ctx.ckl_mode, ckl_v2=ctx.ckl_v2),
        _render_components(comps),
        _render_decisions(decs, ckl_mode=ctx.ckl_mode, ckl_v2=ctx.ckl_v2),
        render_skeleton_section(ctx.skeleton),
        _render_summary(ctx.summary_text),
    ]
    return "\n\n".join(s for s in sections if s)


def _enforce_section_budget(
    events: list[Event],
    render_fn: Callable[[list[Event]], str],
    budget: int,
) -> list[Event]:
    """Drop lowest-weight events until render_fn(remaining) fits in budget tokens.

    Never drops the last remaining event — even if a single event exceeds the
    section budget, the event is kept and the global backstop is left to handle
    the overflow. Returns the surviving subset of events.
    """
    remaining = list(events)
    while remaining and count_tokens_accurate(render_fn(remaining)) > budget:
        if len(remaining) == 1:
            return remaining
        # Sort ascending and pop the first (lowest-weight) event
        remaining.sort(key=lambda e: e.weight)
        remaining.pop(0)
    return remaining


# ── section renderers ─────────────────────────────────────────────────────────

def _render_header(ctx: InjectionContext) -> str:
    from memlora.injection.ckl import CKL_LEGEND
    header = (
        f"## Session context [auto-generated — do not edit]\n"
        f"project: {ctx.project_name} · session {ctx.session_number} "
        f"of {ctx.total_sessions} · state v{ctx.state_version}"
    )
    if ctx.skeleton:
        header += (
            "\nBefore using Read/Glob/Grep, check Codebase skeleton below — "
            "classes, methods, and imports listed without re-reading files."
        )
    if ctx.ckl_mode:
        header += f"\n{CKL_LEGEND}"
    if ctx.ckl_v2:
        from memlora.injection.ckl import CKL_OPS_LEGEND
        header += f"\n{CKL_OPS_LEGEND}"
    return header


def _render_hard_constraints(
    constraints: list[Event], ckl_mode: bool = False, ckl_v2: bool = False
) -> str:
    if not constraints:
        return ""
    # Stable sort by content_hash → deterministic order → prompt-cache hits
    ordered = sorted(constraints, key=lambda c: c.content_hash)
    if ckl_mode:
        from memlora.injection.ckl import render_event_ckl
        lines = ["### Hard constraints — never violate"]
        for c in ordered:
            lines.append(render_event_ckl(c, "CSTR", v2=ckl_v2))
        return "\n".join(lines)
    lines = ["### Hard constraints — never violate"]
    for c in ordered:
        desc = c.payload.get("description", "")
        rationale = c.payload.get("rationale", "")
        if rationale:
            lines.append(f"- {desc} — {rationale}")
        else:
            lines.append(f"- {desc}")
    return "\n".join(lines)


def _render_graveyard(
    items: list[Event], ckl_mode: bool = False, ckl_v2: bool = False
) -> str:
    if not items:
        return ""
    ordered = sorted(items, key=lambda e: e.content_hash)
    if ckl_mode:
        from memlora.injection.ckl import render_event_ckl
        lines = ["### Do not retry — confirmed failures"]
        for item in ordered:
            lines.append(render_event_ckl(item, "DEAD", v2=ckl_v2))
        return "\n".join(lines)
    lines = ["### Do not retry — confirmed failures"]
    for item in ordered:
        approach = item.payload.get("description", "")
        reason = item.payload.get("rationale", item.payload.get("reason", ""))
        if reason:
            lines.append(f"- {approach} -> {reason}")
        else:
            lines.append(f"- {approach}")
    return "\n".join(lines)


def _render_components(components: list[Event]) -> str:
    if not components:
        return ""
    lines = ["### Component state"]
    for c in components:
        path = c.payload.get("path", "")
        status = c.payload.get("status", c.payload.get("change_type", "modified")).upper()
        intent = c.payload.get("intent", "")
        if intent:
            lines.append(f"- {path} · {status} — {intent}")
        else:
            lines.append(f"- {path} · {status}")
    return "\n".join(lines)


def _render_decisions(
    decisions: list[Event], ckl_mode: bool = False, ckl_v2: bool = False
) -> str:
    if not decisions:
        return ""
    if ckl_mode:
        from memlora.injection.ckl import render_event_ckl
        lines = ["### Key decisions"]
        for d in decisions:
            lines.append(render_event_ckl(d, "DEC", v2=ckl_v2))
        return "\n".join(lines)
    lines = ["### Key decisions"]
    for i, d in enumerate(decisions, 1):
        desc = d.payload.get("description", "")
        rationale = d.payload.get("rationale", "")
        sess = d.session_id
        if rationale:
            lines.append(f"{i}. {desc} — {rationale} (session {sess})")
        else:
            lines.append(f"{i}. {desc} (session {sess})")
    return "\n".join(lines)


def _render_active_thread(threads: list[Event]) -> str:
    if not threads:
        return ""
    thread = threads[0]
    desc = thread.payload.get("description", "")
    state = thread.payload.get("state", "")
    next_steps = thread.payload.get("next_steps", "")
    lines = ["### Active thread", f"Working on: {desc}"]
    if state:
        lines.append(f"Current state: {state}")
    if next_steps:
        lines.append(f"Next: {next_steps}")
    return "\n".join(lines)


def _render_hot_files(files: list[tuple[str, int, str]], has_skeleton: bool = False) -> str:
    if not files or not has_skeleton:
        return ""
    lines = ["### Most active files — structure in Codebase skeleton below"]
    for path, mentions, _ in files:
        lines.append(f"- {path} · {mentions}x")
    return "\n".join(lines)


def _render_summary(summary_text: str) -> str:
    if not summary_text:
        return ""
    return f"### Summary\n{summary_text}"


# ── summary generation ────────────────────────────────────────────────────────

def generate_summary(ctx: InjectionContext) -> str:
    """Deterministic NL summary — no LLM call, zero latency."""
    parts: list[str] = []

    languages: set[str] = set()
    has_package_json = False
    for c in ctx.components:
        path = c.payload.get("path", "")
        if path.endswith((".ts", ".tsx")):
            languages.add("TypeScript")
        elif path.endswith(".py"):
            languages.add("Python")
        elif path.endswith((".js", ".jsx", ".mjs")):
            languages.add("JavaScript")
        elif path.endswith(".go"):
            languages.add("Go")
        elif path.endswith(".rs"):
            languages.add("Rust")
        if path == "package.json" or path.endswith("/package.json"):
            has_package_json = True

    frameworks: set[str] = set()
    if has_package_json:
        frameworks.add("Node")

    if languages or frameworks:
        lang_str = "/".join(sorted(languages))
        fw_str = "/".join(sorted(frameworks))
        if lang_str and fw_str:
            parts.append(f"{lang_str}/{fw_str} project.")
        elif lang_str:
            parts.append(f"{lang_str} project.")
        else:
            parts.append(f"{fw_str} project.")

    if ctx.active_threads:
        desc = ctx.active_threads[0].payload.get("description", "")
        if desc:
            parts.append(f"Currently {desc}.")

    in_flux = [
        c.payload.get("path", "") for c in ctx.components
        if c.payload.get("status") == "in_flux"
    ]
    if in_flux:
        parts.append(f"Do not ship until {in_flux[0]} is stable.")

    return " ".join(parts) if parts else "Project state is being established."


# ── token counting ────────────────────────────────────────────────────────────

def count_tokens_accurate(text: str) -> int:
    """Token count via tiktoken (cl100k_base); falls back to len/4 if unavailable."""
    try:
        import tiktoken
        encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    except Exception:
        return max(1, len(text) // 4)


# ── budget enforcement ────────────────────────────────────────────────────────

def render_with_budget_enforcement(ctx: InjectionContext) -> str:
    """Render and apply backstop drop loop if accurate token count exceeds budget.

    Drop order (never drops hard constraints, graveyard, or active thread):
      1. Ranked decisions — pop lowest-weight (list is weight-desc, so pop tail)
      2. Stable components
      3. Skeleton entries — pop lowest symbol-count file
    """
    ctx = copy.copy(ctx)
    ctx.decisions = list(ctx.decisions)
    ctx.components = list(ctx.components)
    ctx.active_threads = list(ctx.active_threads)  # protected — never dropped
    ctx.skeleton = list(ctx.skeleton)

    block = render_injection(ctx)
    actual = count_tokens_accurate(block)

    if actual <= ctx.token_budget:
        return block

    # Global backstop activated — section budgets (if set) were insufficient.
    if ctx.section_budgets is not None:
        _log.warning(
            "injection.backstop_activated",
            extra={"actual_tokens": actual, "budget": ctx.token_budget},
        )

    while actual > ctx.token_budget and ctx.decisions:
        ctx.decisions.pop()
        block = render_injection(ctx)
        actual = count_tokens_accurate(block)

    while actual > ctx.token_budget and ctx.components:
        stable_idx = next(
            (i for i, c in enumerate(ctx.components)
             if c.payload.get("status") == "stable"),
            None,
        )
        if stable_idx is None:
            break
        ctx.components.pop(stable_idx)
        block = render_injection(ctx)
        actual = count_tokens_accurate(block)

    while actual > ctx.token_budget and ctx.skeleton:
        min_idx = min(
            range(len(ctx.skeleton)),
            key=lambda i: len(ctx.skeleton[i].classes) + len(ctx.skeleton[i].functions),
        )
        ctx.skeleton.pop(min_idx)
        block = render_injection(ctx)
        actual = count_tokens_accurate(block)

    return block
