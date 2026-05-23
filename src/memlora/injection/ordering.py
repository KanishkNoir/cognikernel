"""Partition a flat compressed event list into injection section buckets."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlora.storage.events import Event
    from memlora.injection.template import InjectionContext

_HARD_TYPES      = frozenset({"CONSTRAINT_HARD"})
_GRAVEYARD_TYPES = frozenset({"APPROACH_ABANDONED_DO_NOT_RETRY"})
_COMPONENT_TYPES = frozenset({"COMPONENT_STATUS"})
_DECISION_TYPES  = frozenset({"DECISION", "CONSTRAINT_SOFT", "APPROACH_ABANDONED"})
_THREAD_TYPES    = frozenset({"THREAD_OPEN"})
# THREAD_CLOSE: intentionally excluded — low base weight, historical noise


def partition_events(events: list[Event]) -> dict[str, list[Event]]:
    """Split a flat event list into the five section buckets.

    THREAD_CLOSE events are silently dropped — they have very low weight and
    represent completed work that doesn't need to be injected.
    """
    buckets: dict[str, list[Event]] = {
        "hard_constraints": [],
        "graveyard": [],
        "components": [],
        "decisions": [],
        "active_threads": [],
    }
    for e in events:
        if e.event_type in _HARD_TYPES:
            buckets["hard_constraints"].append(e)
        elif e.event_type in _GRAVEYARD_TYPES:
            buckets["graveyard"].append(e)
        elif e.event_type in _COMPONENT_TYPES:
            buckets["components"].append(e)
        elif e.event_type in _DECISION_TYPES:
            buckets["decisions"].append(e)
        elif e.event_type in _THREAD_TYPES:
            buckets["active_threads"].append(e)
    return buckets


def make_injection_context(
    events: list[Event],
    project_name: str,
    session_number: int,
    total_sessions: int,
    state_version: int,
    token_budget: int = 2000,
) -> InjectionContext:
    """Build an InjectionContext from a compressed event list and project metadata."""
    from memlora.injection.template import InjectionContext, generate_summary

    buckets = partition_events(events)
    ctx = InjectionContext(
        project_name=project_name,
        session_number=session_number,
        total_sessions=total_sessions,
        state_version=state_version,
        hard_constraints=buckets["hard_constraints"],
        graveyard=buckets["graveyard"],
        components=buckets["components"],
        decisions=buckets["decisions"],
        active_threads=buckets["active_threads"],
        summary_text="",
        token_budget=token_budget,
    )
    ctx.summary_text = generate_summary(ctx)
    return ctx
