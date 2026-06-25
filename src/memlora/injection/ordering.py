"""Partition a flat compressed event list into injection section buckets."""
from __future__ import annotations

from typing import TYPE_CHECKING

from memlora.extraction.authority import (
    ASSISTANT_ANSWER_TO_QUESTION,
    ASSISTANT_DECIDED,
    CONFIRMING_AUTHORITIES,
    INFERRED_FROM_CODE,
    LLM,
    USER_STATED,
    normalize_subject,
)
from memlora.storage.sections import (
    COMPONENT_TYPES as _COMPONENT_TYPES,
    DECISION_TYPES as _DECISION_TYPES,
    GRAVEYARD_TYPES as _GRAVEYARD_TYPES,
    HARD_TYPES as _HARD_TYPES,
    THREAD_TYPES as _THREAD_TYPES,
)
from memlora.utils.paths import is_bare_basename

if TYPE_CHECKING:
    from memlora.storage.events import Event
    from memlora.injection.template import InjectionContext

# Authority priority used when ranking THREAD_OPEN events for the singular
# "Active thread" slot. Lower wins. The renderer takes threads[0], so a stable
# sort here puts the user's most recent directive at the top regardless of
# weight inflation from repeated assistant musings.
_THREAD_AUTHORITY_PRIORITY = {
    USER_STATED: 0,
    ASSISTANT_ANSWER_TO_QUESTION: 1,
    LLM: 2,
    ASSISTANT_DECIDED: 3,
    INFERRED_FROM_CODE: 4,
}
_THREAD_AUTHORITY_FALLBACK = 5  # missing or unknown authority


def _thread_sort_key(event) -> tuple[int, float]:
    """Sort key for the active_threads bucket: (authority_priority, -weight)."""
    authority = (event.payload.get("authority") or "")
    return (
        _THREAD_AUTHORITY_PRIORITY.get(authority, _THREAD_AUTHORITY_FALLBACK),
        -float(event.weight or 0.0),
    )


def partition_events(events: list[Event]) -> dict[str, list[Event]]:
    """Split a flat event list into the six section buckets.

    Routing depends on (event_type, authority):
      - Events with authority=assistant_answer_to_user_question are routed
        to `pending_confirmations` unless a confirming-authority event with
        the same normalized subject also exists (then the co-capture is
        suppressed because the fact has been independently asserted).
      - Everything else routes by event_type as before.

    THREAD_CLOSE events are silently dropped — they have very low weight and
    represent completed work that doesn't need to be injected.
    """
    buckets: dict[str, list[Event]] = {
        "hard_constraints": [],
        "graveyard": [],
        "components": [],
        "decisions": [],
        "active_threads": [],
        "pending_confirmations": [],
    }

    # Pass 1 — index confirming-authority subjects for suppression.
    confirmed_subjects: set[str] = set()
    for e in events:
        auth = e.payload.get("authority", "")
        if auth in CONFIRMING_AUTHORITIES:
            subj = _subject_of(e)
            norm = normalize_subject(subj) if subj else ""
            if norm:
                confirmed_subjects.add(norm)

    # Pass 2 — route each event into its bucket.
    for e in events:
        auth = e.payload.get("authority", "")
        if auth == ASSISTANT_ANSWER_TO_QUESTION:
            subj = _subject_of(e)
            norm = normalize_subject(subj) if subj else ""
            if norm and norm in confirmed_subjects:
                continue  # suppressed by a confirming event
            buckets["pending_confirmations"].append(e)
            continue

        if e.event_type in _HARD_TYPES:
            buckets["hard_constraints"].append(e)
        elif e.event_type in _GRAVEYARD_TYPES:
            buckets["graveyard"].append(e)
        elif e.event_type in _COMPONENT_TYPES:
            # Defensive: skip bare-basename COMPONENT_STATUS rows. New
            # extractions reject these at insertion (file_mentions.py), but
            # projects whose DB predates that fix may still contain them.
            path = e.payload.get("path", "")
            if path and is_bare_basename(path):
                continue
            buckets["components"].append(e)
        elif e.event_type in _DECISION_TYPES:
            buckets["decisions"].append(e)
        elif e.event_type in _THREAD_TYPES:
            buckets["active_threads"].append(e)

    # Rank active threads so the renderer's threads[0] pick lands on the
    # user's directive, not on whichever assistant musing accreted the most
    # weight. Stable sort preserves insertion order within the same priority.
    buckets["active_threads"].sort(key=_thread_sort_key)
    return buckets


def _subject_of(event) -> str:
    """Best-effort subject extraction from an event payload.

    Looks in order at:
      - payload['subject']        (pattern events have this directly)
      - payload['triple']['subject']  (events that went through augment_with_triple)
      - payload['path']           (component events)
    """
    p = event.payload
    s = p.get("subject", "")
    if s:
        return s
    triple = p.get("triple")
    if isinstance(triple, dict):
        ts = triple.get("subject", "") or triple.get("object", "")
        if ts:
            return ts
    return p.get("path", "")


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
        pending_confirmations=buckets.get("pending_confirmations", []),
    )
    ctx.summary_text = generate_summary(ctx)
    return ctx
