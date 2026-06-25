"""Single source of truth for event-type → injection-section routing.

Both the projection builder (`storage.projections.rebuild_projection`) and the
render-time partitioner (`injection.ordering.partition_events`) route events
into the same buckets. Historically each module defined its own copy of these
frozensets and they drifted. Import from here so there is exactly one routing
table.
"""
from __future__ import annotations

HARD_TYPES: frozenset[str] = frozenset({"CONSTRAINT_HARD"})
GRAVEYARD_TYPES: frozenset[str] = frozenset({"APPROACH_ABANDONED_DO_NOT_RETRY"})
COMPONENT_TYPES: frozenset[str] = frozenset({"COMPONENT_STATUS"})
DECISION_TYPES: frozenset[str] = frozenset({"DECISION", "CONSTRAINT_SOFT", "APPROACH_ABANDONED"})
THREAD_TYPES: frozenset[str] = frozenset({"THREAD_OPEN"})
# THREAD_CLOSE is intentionally unrouted — low base weight, historical noise.
