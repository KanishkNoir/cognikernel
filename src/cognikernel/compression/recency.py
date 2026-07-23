"""Hyperbolic recency decay for event ranking."""
from __future__ import annotations

_DEFAULT_ALPHA: float = 0.15


def recency_factor(sessions_since_last_mention: int, alpha: float = _DEFAULT_ALPHA) -> float:
    """Hyperbolic decay: 1 / (1 + α·t).

    Preserves long-term architectural memory better than exponential decay —
    hard constraints from session 1 remain relevant at session 50 (≈0.12),
    whereas exponential decay would drop them near zero.
    """
    return 1.0 / (1.0 + alpha * max(0, sessions_since_last_mention))
