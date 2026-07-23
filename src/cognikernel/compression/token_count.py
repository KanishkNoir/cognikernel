"""Token counting — the single counter used by both selection and enforcement.

`count_tokens` is the one canonical implementation. In a default install it is
the len/4 heuristic: tiktoken is NOT a dependency (install the `tokens` extra —
or tiktoken directly — to get exact `cl100k_base` counts; the encoder is then
loaded once and cached). What matters for correctness is that `greedy_fill`
(via `estimate_tokens`) and the renderer (via
`injection.template.count_tokens_accurate`, which delegates here) agree on ONE
unit — previously selection used len/4 while enforcement used tiktoken.
Budgets (config.token_budget et al.) are calibrated against this counter as
deployed, i.e. the heuristic unless tiktoken is present.
"""
from __future__ import annotations

import functools
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cognikernel.model import Event

_CHARS_PER_TOKEN = 4


@functools.lru_cache(maxsize=1)
def _encoder():
    """Load and cache the tiktoken encoder once (None if tiktoken is missing)."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Canonical token count for a string. tiktoken when available, else len/4."""
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_tokens(event: "Event") -> int:
    """Estimate an event's token cost using the canonical counter.

    Serializes the event's visible fields (type + key payload fields) and counts
    with `count_tokens`. Returns at least 1. This is an approximation of the
    event's rendered cost (it omits markdown wrapping), but it now shares the
    same counter as the renderer, so selection and enforcement use one unit.
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
    return max(1, count_tokens(text))
