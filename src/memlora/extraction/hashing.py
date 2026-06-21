"""Content hash computation for event deduplication.

The implementation moved to ``memlora.utils.hashing`` (a dependency-free base)
so the delta merge can hash without importing the extraction stage. Re-exported
here so existing ``from memlora.extraction.hashing import ...`` call sites — and
``memlora.extraction.__init__`` — are unaffected.
"""
from __future__ import annotations

from memlora.utils.hashing import compute_content_hash, normalize_for_hash

__all__ = ["compute_content_hash", "normalize_for_hash"]
