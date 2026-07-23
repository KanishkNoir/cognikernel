"""Greedy knapsack fill for token budget management."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from cognikernel.compression.token_count import estimate_tokens

if TYPE_CHECKING:
    from cognikernel.model import Event

_MANDATORY_TYPES: frozenset[str] = frozenset({
    "CONSTRAINT_HARD",
    "APPROACH_ABANDONED_DO_NOT_RETRY",
})
# Authority values whose events are budget-exempt regardless of event_type.
# A user-stated fact — above all the active thread — must never be evicted by
# an assistant musing that merely accreted more dedup weight (the Tier-1.5
# mis-ranking bug). Mirror of cognikernel.extraction.authority.USER_STATED, kept as
# a bare string so the SessionStart render path avoids importing the heavy
# extraction package.
_MANDATORY_AUTHORITIES: frozenset[str] = frozenset({"user_stated"})
_MANDATORY_TOKEN_LIMIT: int = 500
_COMPONENT_TOKEN_LIMIT: int = 150
_COMPONENT_MAX_COUNT: int = 5

# R8 — authority precedence for the mandatory drop-to-fit: a user-stated fact outranks
# an assistant-decided one, so when budget-exempt constraints overflow we keep the
# user's real constraints and drop over-captured assistant prose first. Mirrors
# extraction.authority constants (kept as bare strings to avoid importing extraction).
_AUTHORITY_RANK: dict[str, int] = {
    "user_stated": 3,
    "assistant_decided": 2,
    "llm": 2,
    "assistant_answer_to_user_question": 1,
    "inferred_from_code": 0,
}


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

    # J7.2: internal zone caps scale with the configured budget so a larger
    # block grows every zone proportionally — at 5000 tokens the mandatory
    # zone must not stay pinned at the 1500-budget's 500.
    scale = max(budget_tokens, 1) / 1500.0
    mandatory_limit = int(_MANDATORY_TOKEN_LIMIT * scale)
    component_limit = int(_COMPONENT_TOKEN_LIMIT * scale)

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

    if mandatory_tokens > mandatory_limit:
        mandatory = _compress_mandatory(mandatory, mandatory_limit)
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
        if comp_tokens + cost <= component_limit:
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


# ── internal helpers ──────────────────────────────────────────────────────────

def _compress_mandatory(
    mandatory: list["Event"],
    target_tokens: int,
) -> list["Event"]:
    """Fit mandatory items into the token limit by DROPPING whole low-priority events.

    Lossless render contract: never truncate a field (a clipped constraint destroys
    signal). When budget-exempt constraints overflow, drop whole events worst-first by
    (authority, weight) so a user-stated constraint outlives an over-captured assistant
    one. Rarely hit with well-formed extraction. The events store is untouched — this
    only shapes the rendered block.
    """
    mandatory = [copy.copy(e) for e in mandatory]
    for e in mandatory:
        e.payload = dict(e.payload)

    if sum(estimate_tokens(e) for e in mandatory) <= target_tokens:
        return mandatory

    # R8: keep highest (authority, weight) first; drop the rest whole.
    mandatory.sort(key=lambda e: (_AUTHORITY_RANK.get(e.payload.get("authority", ""), 2), e.weight), reverse=True)
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
