"""Token estimation for compression budget management."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlora.storage.events import Event

_CHARS_PER_TOKEN = 4


def estimate_tokens(event: "Event") -> int:
    """Estimate token cost using the len/4 approximation.

    Serializes the event's visible fields (type + key payload fields) and
    divides by 4. Returns at least 1.
    """
    parts: list[str] = [event.event_type]
    for key in ("description", "rationale", "path"):
        val = event.payload.get(key, "")
        if val:
            parts.append(str(val))
    files = event.payload.get("affected_files", [])
    if files:
        parts.append(", ".join(str(f) for f in files[:5]))
    text = " | ".join(parts)
    return max(1, len(text) // _CHARS_PER_TOKEN)
