"""Composite weight formula for event ranking.

weight = base × recency × repetition × centrality × activity × type_multiplier × quality

Each factor is multiplicative — deficiency in any one suppresses the total.
`quality` applies the demotes a claim's markers carry (quality.detectors); the
stored weight is not an input, so a demote that only lowered it did nothing here.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from cognikernel.compression.centrality import centrality_factor
from cognikernel.compression.recency import recency_factor
from cognikernel.quality.detectors import quality_factor

if TYPE_CHECKING:
    from cognikernel.model import Event

BASE_WEIGHT: dict[str, float] = {
    "CONSTRAINT_HARD": 1.0,
    "APPROACH_ABANDONED_DO_NOT_RETRY": 1.0,
    "DECISION": 0.7,
    "CONSTRAINT_SOFT": 0.5,
    "COMPONENT_STATUS": 0.4,
    "APPROACH_ABANDONED": 0.4,
    "THREAD_OPEN": 0.6,
    "THREAD_CLOSE": 0.2,
}

TYPE_MULTIPLIER: dict[str, float] = {
    "CONSTRAINT_HARD": 1.5,
    "APPROACH_ABANDONED_DO_NOT_RETRY": 1.4,
    "DECISION": 1.0,
    "CONSTRAINT_SOFT": 0.9,
    "COMPONENT_STATUS": 1.1,
    "APPROACH_ABANDONED": 0.7,
    "THREAD_OPEN": 1.2,
    "THREAD_CLOSE": 0.4,
}

_ACTIVITY_BOOST: dict[str, float] = {
    "in_flux": 2.0,
    "blocked": 1.8,
    "needs_review": 1.5,
    "new": 1.4,
    "stable": 0.7,
    "complete": 0.5,
    "abandoned": 0.3,
    "unknown": 1.0,
}


def repetition_factor(mention_count: int) -> float:
    """Logarithmic growth — saturates to prevent high-mention domination.

    mention_count=1 → 1.0, =2 → 1.21, =5 → 1.48, =20 → 1.90
    """
    return 1.0 + 0.3 * math.log(max(1, mention_count))


def activity_factor(
    file_paths: list[str],
    component_map: dict[str, dict],
) -> float:
    """Status-based boost; returns the max boost across all affected files.

    in_flux (2.0) is the strongest multiplier — forces active work to dominate
    the injection block. Stable/complete files are actively suppressed.
    """
    if not file_paths:
        return 1.0
    boosts = [
        _ACTIVITY_BOOST.get(component_map.get(p, {}).get("status", "unknown"), 1.0)
        for p in file_paths
    ]
    return max(boosts)


def weight_factors(
    event: "Event",
    component_map: dict[str, dict],
    centrality_map: dict[str, float],
    current_session: int = 0,
) -> dict[str, float]:
    """The seven named factors of an event's ranking weight, in formula order.

    `cognikernel why` shows these (§14: "show the factorisation, not the
    number"), so compute_weight is their product and there is one formula.
    """
    affected_files: list[str] = event.payload.get("affected_files", [])
    return {
        "base": BASE_WEIGHT.get(event.event_type, 0.5),
        "recency": recency_factor(max(0, current_session - event.last_mentioned_session)),
        "repetition": repetition_factor(event.mention_count),
        "centrality": centrality_factor(affected_files, centrality_map),
        "activity": activity_factor(affected_files, component_map),
        "type": TYPE_MULTIPLIER.get(event.event_type, 1.0),
        "quality": quality_factor(event.payload),
    }


def compute_weight(
    event: "Event",
    component_map: dict[str, dict],
    centrality_map: dict[str, float],
    current_session: int = 0,
) -> float:
    """Compute the full ranking weight for a single event."""
    f = weight_factors(event, component_map, centrality_map, current_session)
    return (f["base"] * f["recency"] * f["repetition"] * f["centrality"] * f["activity"] * f["type"]
            * f["quality"])
