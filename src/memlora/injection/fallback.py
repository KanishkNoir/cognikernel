"""Fallback rendering for absent or corrupted projection state.

Three distinct failure modes, three distinct messages:
  - None ctx          → project not initialised
  - Empty projection  → first session, state accumulating
  - Corrupted storage → tell user to run memlora doctor
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from memlora.injection.template import InjectionContext


class ProjectionCorruptedError(Exception):
    """Raised when a projection cannot be loaded due to storage corruption."""


def render_or_fallback(ctx: Optional[InjectionContext]) -> str:
    """Render a full block or an appropriate fallback. Never returns empty string."""
    from memlora.injection.template import render_with_budget_enforcement

    if ctx is None:
        return _fallback_uninitialized()

    if not any([
        ctx.hard_constraints,
        ctx.graveyard,
        ctx.components,
        ctx.decisions,
        ctx.active_threads,
    ]):
        return _fallback_empty_projection(ctx)

    try:
        return render_with_budget_enforcement(ctx)
    except ProjectionCorruptedError:
        return _fallback_corrupted(ctx)


def _fallback_uninitialized() -> str:
    return (
        "## Session context [auto-generated]\n"
        "MemLoRA is not initialized for this project. "
        "Run `memlora init` in the project root to enable session memory."
    )


def _fallback_empty_projection(ctx: InjectionContext) -> str:
    return (
        f"## Session context [auto-generated]\n"
        f"project: {ctx.project_name} · session 1\n\n"
        f"This is the first session for this project. "
        f"State will accumulate as you work — by session 3 or 4, "
        f"you'll see decisions, constraints, and component status here."
    )


def _fallback_corrupted(ctx: InjectionContext) -> str:
    return (
        f"## Session context [auto-generated]\n"
        f"project: {ctx.project_name}\n\n"
        f"Session state could not be loaded due to a storage error. "
        f"Run `memlora doctor` to diagnose. Continuing without context."
    )
