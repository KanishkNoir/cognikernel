"""Hard vs. soft constraint classification heuristics.

Constraint events start life with the signal_type from the trie (CONSTRAINT_HARD
or CONSTRAINT_SOFT). This module re-scores them using linguistic and contextual
signals and may reclassify in either direction.

The 0.85 threshold should be calibrated against a labeled corpus once the eval
harness (Stage 7) is in place.
"""
from __future__ import annotations

import re

from cognikernel.storage.events import Event

HARD_THRESHOLD: float = 0.85

# Explicit deontic / prohibition signal (#39). A genuine hard constraint names
# its force ("must", "never", "do not", "no X"); a bare imperative ("Use
# Postgres", "Set retries to 5") names none. Word-boundary matched, deliberately
# inclusive on negations so real user prohibitions are never demoted.
_DEONTIC_RE = re.compile(
    r"\b(must|mandatory|required|requirement|non-negotiable|obligatory|"
    r"always|never|cannot|can't|do not|don't|shall not|must not|"
    r"prohibit\w*|forbid\w*|disallow\w*|no|not)\b"
)

REQUIREMENT_MARKERS: frozenset[str] = frozenset(
    {
        "requirement", "required", "must", "mandatory",
        "non-negotiable", "always", "never", "obligatory",
    }
)
DOMAIN_MARKERS: frozenset[str] = frozenset(
    {
        "production", "deploy", "security", "auth", "authentication",
        "compliance", "hipaa", "gdpr", "sox", "pci", "regulatory",
    }
)
HEDGE_MARKERS: frozenset[str] = frozenset(
    {
        "probably", "might", "perhaps", "consider",
        "maybe", "could", "potentially", "possibly", "seems",
    }
)

# Transcript marker that forces hard classification regardless of score.
_HARD_OVERRIDE = "<!-- cognikernel:hard -->"
_SOFT_OVERRIDE = "<!-- cognikernel:soft -->"


def classify_event(event: Event) -> Event:
    """Reclassify constraint events; pass all other event types through unchanged."""
    if event.event_type not in ("CONSTRAINT_HARD", "CONSTRAINT_SOFT"):
        return event

    payload = event.payload
    description = payload.get("description", "")

    # Explicit overrides take precedence over heuristics.
    if _HARD_OVERRIDE in description:
        event.event_type = "CONSTRAINT_HARD"
        return event
    if _SOFT_OVERRIDE in description:
        event.event_type = "CONSTRAINT_SOFT"
        return event

    final_type = classify_constraint(
        confidence=float(payload.get("confidence", 0.5)),
        source_role=str(payload.get("source_role", "user")),
        description=description,
    )
    event.event_type = final_type
    return event


def classify_constraint(
    confidence: float,
    source_role: str,
    description: str,
    mention_count: int = 1,
) -> str:
    """Return 'CONSTRAINT_HARD' or 'CONSTRAINT_SOFT' based on scoring heuristics."""
    score = confidence

    # Repetition signal — the same constraint mentioned multiple times.
    if mention_count >= 2:
        score += 0.3

    # Source authority — user statements are more trustworthy than model speculation.
    if source_role == "user":
        score += 0.2
    elif source_role == "assistant":
        score -= 0.1

    d = description.lower()

    if any(m in d for m in REQUIREMENT_MARKERS):
        score += 0.2

    if any(m in d for m in DOMAIN_MARKERS):
        score += 0.15

    # Hedging language suggests the speaker is unsure — lower confidence in the rule.
    if any(m in d for m in HEDGE_MARKERS):
        score -= 0.3

    # Assistant source-role gate: only highest-confidence signals (must-not,
    # must-never, mandatory, cannot — all 1.0 in SIGNAL_DICTIONARY) survive as
    # CONSTRAINT_HARD when spoken by the assistant. Lower-confidence signals
    # (e.g. bare "never" at 0.9) get capped below the 0.85 threshold.
    if source_role == "assistant" and confidence < 1.0:
        score = min(score, 0.80)

    # User source-role gate (#39): a bare user imperative ("Use Postgres", "Set
    # retries to 5") collects the +0.2 user-authority bump and can clear the hard
    # threshold with no deontic signal at all — minting an ordinary decision as a
    # budget-exempt mandatory constraint. Symmetric to the assistant gate above:
    # a user constraint with no explicit deontic/prohibition marker is capped
    # below HARD. Genuine user prohibitions ("never use floats", "do not …",
    # "no X") carry a marker via _DEONTIC_RE and are unaffected.
    if source_role == "user" and confidence < 1.0 and not _DEONTIC_RE.search(d):
        score = min(score, 0.80)

    return "CONSTRAINT_HARD" if score >= HARD_THRESHOLD else "CONSTRAINT_SOFT"
