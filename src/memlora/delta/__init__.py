"""Stage 5 — Delta merge and versioning.

Public API:
  execute_merge         — full six-step merge in one transaction
  merge_event           — insert-or-update a single event (with commit)
  find_superseded       — gated supersession finder (temporal + authority +
                          provenance, with optional semantic axis); the merge path
  detect_supersession   — ungated lexical-only finder (legacy primitive)
  apply_supersession    — mark events as superseded_by in the DB
  events_overlap        — OR of Jaccard + Levenshtein overlap detection
  jaccard_similarity    — token-set overlap in [0, 1]
  levenshtein_normalized — edit distance in [0.0 (identical) … 1.0 (different)]
  cascade_component_status — emit needs_review for blocked/abandoned dependents
  apply_decay_pass      — standalone decay + archive with idempotency guard
  DECAY_FACTOR          — 0.92 per-session multiplicative factor (~8-session half-life)
  ARCHIVE_THRESHOLD     — 0.05 floor below which events are archived
"""
from memlora.delta.cascade import cascade_component_status
from memlora.delta.decay import ARCHIVE_THRESHOLD, DECAY_FACTOR, apply_decay_pass
from memlora.delta.merge import execute_merge, merge_event
from memlora.delta.supersede import (
    JACCARD_THRESHOLD,
    LEVENSHTEIN_THRESHOLD,
    apply_supersession,
    detect_supersession,
    events_overlap,
    find_superseded,
    jaccard_similarity,
    levenshtein_normalized,
)

__all__ = [
    "ARCHIVE_THRESHOLD",
    "DECAY_FACTOR",
    "JACCARD_THRESHOLD",
    "LEVENSHTEIN_THRESHOLD",
    "apply_decay_pass",
    "apply_supersession",
    "cascade_component_status",
    "detect_supersession",
    "events_overlap",
    "execute_merge",
    "find_superseded",
    "jaccard_similarity",
    "levenshtein_normalized",
    "merge_event",
]
