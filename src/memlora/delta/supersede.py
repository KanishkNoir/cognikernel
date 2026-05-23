"""Constraint supersession — keyword-overlap detection without embeddings.

Two complementary metrics:
  Jaccard similarity   — catches different word choice for the same concept
  Levenshtein (norm.)  — catches slight phrasing variations of the same sentence

OR rule: either metric triggering marks events as overlapping.
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlora.storage.events import Event

_SUPERSESSION_TYPES: frozenset[str] = frozenset({
    "CONSTRAINT_HARD",
    "CONSTRAINT_SOFT",
    "DECISION",
    "APPROACH_ABANDONED",
    "APPROACH_ABANDONED_DO_NOT_RETRY",
})

JACCARD_THRESHOLD: float = 0.6
LEVENSHTEIN_THRESHOLD: float = 0.15

STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "we", "our", "can", "will",
    "are", "was", "be", "been", "has", "have", "had", "do", "does",
    "did", "no", "its", "use", "used", "using", "so", "that",
    "this", "with", "from", "as", "by", "if", "up", "out", "any",
})


def normalize_for_overlap(text: str) -> set[str]:
    """Lowercase, strip punctuation, remove stopwords and tokens ≤ 2 chars."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return {t for t in text.split() if len(t) > 2 and t not in STOPWORDS}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard on normalised token sets: |A ∩ B| / |A ∪ B|."""
    a = normalize_for_overlap(text_a)
    b = normalize_for_overlap(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def levenshtein_normalized(a: str, b: str) -> float:
    """Normalised edit distance in [0.0 (identical) … 1.0 (totally different)]."""
    a = a.lower().strip()
    b = b.lower().strip()
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    m, n = len(a), len(b)
    if m < n:
        a, b, m, n = b, a, n, m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (a[i - 1] != b[j - 1]),
            )
        prev = curr
    return prev[n] / m


def events_overlap(event_a: Event, event_b: Event) -> bool:
    """Return True if two events express the same concept in different words."""
    if event_a.event_type != event_b.event_type:
        return False
    desc_a = event_a.payload.get("description", "")
    desc_b = event_b.payload.get("description", "")
    return (
        jaccard_similarity(desc_a, desc_b) >= JACCARD_THRESHOLD
        or levenshtein_normalized(desc_a, desc_b) <= LEVENSHTEIN_THRESHOLD
    )


def detect_supersession(
    conn: sqlite3.Connection,
    new_event: Event,
) -> list[int]:
    """Return IDs of existing events superseded by new_event.

    Queries same-type, non-superseded events then applies overlap detection.
    Returns empty list for event types that do not support supersession.
    """
    if new_event.event_type not in _SUPERSESSION_TYPES:
        return []

    rows = conn.execute(
        """
        SELECT id, payload FROM events
        WHERE project_id    = ?
          AND event_type    = ?
          AND archived      = 0
          AND superseded_by IS NULL
          AND content_hash != ?
        """,
        (new_event.project_id, new_event.event_type, new_event.content_hash),
    ).fetchall()

    new_desc = new_event.payload.get("description", "")
    superseded_ids: list[int] = []
    for row in rows:
        cand_desc = json.loads(row["payload"]).get("description", "")
        if (
            jaccard_similarity(new_desc, cand_desc) >= JACCARD_THRESHOLD
            or levenshtein_normalized(new_desc, cand_desc) <= LEVENSHTEIN_THRESHOLD
        ):
            superseded_ids.append(row["id"])

    return superseded_ids


def apply_supersession(
    conn: sqlite3.Connection,
    new_event_id: int,
    superseded_ids: list[int],
) -> int:
    """Mark each superseded event as replaced by new_event_id. Returns count updated."""
    for old_id in superseded_ids:
        conn.execute(
            "UPDATE events SET superseded_by = ? WHERE id = ?",
            (new_event_id, old_id),
        )
    return len(superseded_ids)
