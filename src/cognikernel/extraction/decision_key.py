"""Decision keys (J2.2) — the normalized topic axis a choice-family event is about.

The implementation moved to ``cognikernel.utils.decision_key`` (a dependency-free
base) so the delta merge can mint keys without importing the extraction stage —
breaking the extraction<->delta layering cycle. Re-exported here so existing
``from cognikernel.extraction.decision_key import ...`` call sites (e.g.
``storage.projections.backfill_keys``) are unaffected.
"""
from __future__ import annotations

from cognikernel.utils.decision_key import (  # noqa: F401  (re-export)
    CHOICE_FAMILY,
    backfill_keys,
    derive_decision_key,
    normalize_key,
)

__all__ = ["CHOICE_FAMILY", "backfill_keys", "derive_decision_key", "normalize_key"]
