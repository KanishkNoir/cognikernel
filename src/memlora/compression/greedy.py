"""Greedy knapsack fill for token budget management."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from memlora.compression.token_count import estimate_tokens

if TYPE_CHECKING:
    from memlora.storage.events import Event

_MANDATORY_TYPES: frozenset[str] = frozenset({
    "CONSTRAINT_HARD",
    "APPROACH_ABANDONED_DO_NOT_RETRY",
})
# Authority values whose events are budget-exempt regardless of event_type.
# A user-stated fact — above all the active thread — must never be evicted by
# an assistant musing that merely accreted more dedup weight (the Tier-1.5
# mis-ranking bug). Mirror of memlora.extraction.authority.USER_STATED, kept as
# a bare string so the SessionStart render path avoids importing the heavy
# extraction package.
_MANDATORY_AUTHORITIES: frozenset[str] = frozenset({"user_stated"})
_MANDATORY_TOKEN_LIMIT: int = 500
_COMPONENT_TOKEN_LIMIT: int = 150
_COMPONENT_MAX_COUNT: int = 5


def greedy_fill(events: list["Event"], budget_tokens: int) -> list["Event"]:
    """Select events greedily by weight to maximize value within the token budget.

    Phase 1: include all mandatory events (CONSTRAINT_HARD,
    APPROACH_ABANDONED_DO_NOT_RETRY). If they alone exceed 200 tokens,
    compress them aggressively before proceeding.

    Phase 2: sort remaining candidates by weight descending and fill greedily.
    The loop never breaks early — a small high-value item later in the list
    may still fit after a large item fails.
    """
    non_archived = [e for e in events if not e.archived]

    # Phase 1 — mandatory (budget-exempt guardrail). Protect both the mandatory
    # event types and any user-stated event (authority gate) so the user's own
    # statements — notably the active thread — survive budget pressure.
    mandatory = [
        e for e in non_archived
        if e.event_type in _MANDATORY_TYPES
        or e.payload.get("authority") in _MANDATORY_AUTHORITIES
    ]
    # Capture original ids BEFORE _compress_mandatory returns shallow copies so
    # Phase 2 correctly excludes them even after the list is replaced.
    mandatory_ids = {id(e) for e in mandatory}
    mandatory_tokens = sum(estimate_tokens(e) for e in mandatory)

    if mandatory_tokens > _MANDATORY_TOKEN_LIMIT:
        mandatory = _compress_mandatory(mandatory, _MANDATORY_TOKEN_LIMIT)
        mandatory_tokens = sum(estimate_tokens(e) for e in mandatory)

    # Deduplicate by path — keep highest-weight event per unique path so a file
    # that transitioned from REFERENCED to MODIFIED doesn't appear twice.
    _seen_paths: dict[str, "Event"] = {}
    for e in non_archived:
        if e.event_type != "COMPONENT_STATUS" or id(e) in mandatory_ids:
            continue
        path = e.payload.get("path", "")
        if path not in _seen_paths or e.weight > _seen_paths[path].weight:
            _seen_paths[path] = e
    component_candidates = sorted(
        _seen_paths.values(),
        key=lambda e: e.weight,
        reverse=True,
    )
    guaranteed_components: list[Event] = []
    comp_tokens = 0
    for e in component_candidates[:_COMPONENT_MAX_COUNT]:
        cost = estimate_tokens(e)
        if comp_tokens + cost <= _COMPONENT_TOKEN_LIMIT:
            guaranteed_components.append(e)
            comp_tokens += cost

    selected: list[Event] = list(mandatory) + guaranteed_components
    used = mandatory_tokens + comp_tokens
    guaranteed_ids = {id(e) for e in guaranteed_components}

    # Phase 2 — greedy fill from remaining budget
    candidates = [
        e for e in non_archived
        if id(e) not in mandatory_ids and id(e) not in guaranteed_ids
    ]
    candidates.sort(key=lambda e: e.weight, reverse=True)

    for event in candidates:
        cost = estimate_tokens(event)
        if used + cost <= budget_tokens:
            selected.append(event)
            used += cost

    return selected


def compress_field_level(
    events: list["Event"],
    target_tokens: int,
) -> list["Event"]:
    """Field-level compression applied after greedy fill when budget is tight.

    Compression stages (never applied to mandatory types at stage 1/2,
    never applied to CONSTRAINT_HARD at stage 2):
      1. Trim ``affected_files`` to first 3 entries.
      2. Truncate ``rationale`` to 80 characters.

    Never drops ``description`` and never compresses mandatory event types.
    """
    events = [copy.copy(e) for e in events]
    for e in events:
        e.payload = dict(e.payload)

    if sum(estimate_tokens(e) for e in events) <= target_tokens:
        return events

    # Stage 1: trim affected_files (skip mandatory types)
    for e in events:
        if e.event_type in _MANDATORY_TYPES:
            continue
        files = e.payload.get("affected_files", [])
        if len(files) > 3:
            e.payload["affected_files"] = files[:3]

    if sum(estimate_tokens(e) for e in events) <= target_tokens:
        return events

    # Stage 2: truncate rationale (skip CONSTRAINT_HARD)
    for e in events:
        if e.event_type == "CONSTRAINT_HARD":
            continue
        rationale = e.payload.get("rationale", "")
        if len(rationale) > 80:
            e.payload["rationale"] = rationale[:77] + "..."

    return events


# ── internal helpers ──────────────────────────────────────────────────────────

def _compress_mandatory(
    mandatory: list["Event"],
    target_tokens: int,
) -> list["Event"]:
    """Aggressively shrink mandatory items when they exceed the token limit."""
    mandatory = [copy.copy(e) for e in mandatory]
    for e in mandatory:
        e.payload = dict(e.payload)

    # Truncate rationale first
    for e in mandatory:
        rationale = e.payload.get("rationale", "")
        if len(rationale) > 60:
            e.payload["rationale"] = rationale[:57] + "..."

    if sum(estimate_tokens(e) for e in mandatory) <= target_tokens:
        return mandatory

    # Still over budget: DROP lowest-weight mandatory events — NEVER truncate a
    # description. A clipped constraint ("...are sub-second transient...") destroys
    # signal and is actively misleading; show fewer constraints in full instead.
    # With well-formed extraction this branch is rarely hit — it triggers only when
    # the store is flooded with low-value/over-captured constraints.
    mandatory.sort(key=lambda e: e.weight, reverse=True)
    kept: list = []
    used = 0
    for e in mandatory:
        cost = estimate_tokens(e)
        if used + cost <= target_tokens:
            kept.append(e)
            used += cost
    if not kept and mandatory:  # always keep at least the highest-weight one, whole
        kept = [mandatory[0]]
    return kept
