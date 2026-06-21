"""Core domain model — the dependency-free event type shared by every layer.

`Event` is the central record every stage produces, ranks, merges, and renders.
It is a pure dataclass with no storage/DB coupling, so it lives in this base
module rather than in `storage.events` (where it used to). Keeping the model
below the layered packages lets compression/delta/extraction/injection type-hint
it without importing "up" into storage — the layering violation the architecture
audit surfaced. `storage.events` re-exports it for backward compatibility, so
existing `from memlora.storage.events import Event` call sites are unaffected.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

VALID_EVENT_TYPES: frozenset[str] = frozenset({
    "DECISION",
    "CONSTRAINT_HARD",
    "CONSTRAINT_SOFT",
    "COMPONENT_STATUS",
    "APPROACH_ABANDONED",
    "APPROACH_ABANDONED_DO_NOT_RETRY",
    "THREAD_OPEN",
    "THREAD_CLOSE",
})


@dataclass
class Event:
    project_id: str
    session_id: str
    event_type: str
    payload: dict[str, Any]
    content_hash: str
    evidence_id: int | None = None
    weight: float = 1.0
    mention_count: int = 1
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))
    id: int | None = None
    superseded_by: int | None = None
    archived: bool = False
    last_mentioned_session: int = 0
    # J2: normalized topic axis ("alias default"); None = not derived yet,
    # '' = derivation found no key. Computed at merge time (the single mint
    # choke point) and lazily backfilled for pre-016 rows.
    decision_key: str | None = None

    def __post_init__(self) -> None:
        if self.event_type not in VALID_EVENT_TYPES:
            raise ValueError(
                f"Unknown event_type {self.event_type!r}. "
                f"Valid types: {sorted(VALID_EVENT_TYPES)}"
            )
