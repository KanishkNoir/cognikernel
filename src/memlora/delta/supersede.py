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


def descriptions_overlap(desc_a: str, desc_b: str) -> bool:
    """Return True if two descriptions express the same concept.

    Exact semantics: ``jaccard >= JACCARD_THRESHOLD OR levenshtein <= LEVENSHTEIN_THRESHOLD``.

    De-quadratic pruning (no behavior change): Jaccard (O(tokens)) is checked
    first and short-circuits. If it fails, a length bound rules out Levenshtein
    cheaply — normalized edit distance is at least ``|len_a − len_b| / max(len)``,
    so when that lower bound already exceeds the threshold the O(m·n) Levenshtein
    computation is skipped entirely. This is exact: it never changes the result,
    only avoids work on length-disparate (hence non-matching) pairs, which is the
    common case during a merge scan.
    """
    if jaccard_similarity(desc_a, desc_b) >= JACCARD_THRESHOLD:
        return True
    la, lb = len(desc_a.strip()), len(desc_b.strip())
    if la and lb and abs(la - lb) / max(la, lb) > LEVENSHTEIN_THRESHOLD:
        return False  # Levenshtein lower bound already exceeds the threshold
    return levenshtein_normalized(desc_a, desc_b) <= LEVENSHTEIN_THRESHOLD


# ── cheap subject derivation (prototype) ─────────────────────────────────────
#
# Description-only overlap misses the canonical supersession case: a decision
# whose *choice* changes but whose *topic* stays the same — "use bcrypt for
# password hashing" → "use argon2id for password hashing" only share ~0.5
# Jaccard, below threshold. derive_subject pulls the topic (the noun phrase the
# decision is *about*, not the tool it picks) so those collapse.
#
# This lives here (not the extraction pipeline) on purpose: it is additive to
# supersession only, needs no payload/schema change, and keeps the heavy
# extraction package off the merge hot path. A topic source in the pipeline
# would let it also feed rendering/dedup-at-insert later.

# Decision verbs that introduce a choice, after which "for/to/as/in <topic>"
# names what the decision concerns.
_DECISION_VERB = (
    r"(?:use|using|adopt\w*|choos\w*|chose|switch\w*|stick\w*\s+with|"
    r"go\w*\s+with|prefer\w*|replac\w*|will\s+use|going\s+to\s+use|we'?ll\s+use)"
)
_TOPIC_RE = re.compile(
    rf"\b{_DECISION_VERB}\b[\w './+-]*?\b(?:for|to|as|in)\s+"
    r"(?P<topic>[a-z0-9][\w +./-]*?)\s*"
    r"(?:\binstead\b|\brather\b|\bbecause\b|\bsince\b|[.,;:]|$)",
    re.IGNORECASE,
)
# Prohibition / abandonment: the subject is the thing being rejected.
_PROHIBIT_RE = re.compile(
    r"\b(?:never\s+use|do\s+not\s+use|don'?t\s+use|avoid|abandon\w*|drop|reject\w*)\s+"
    r"(?P<thing>[a-z0-9][\w./+-]*)",
    re.IGNORECASE,
)
_LEADING_ARTICLE = re.compile(r"^(?:the|a|an|our|this|that|these|those)\s+", re.IGNORECASE)

# Subject match alone is too loose (generic topics like "the database" recur);
# require a minimum description token overlap so subject-keying only rescues
# genuinely-related decisions, not unrelated ones that share a generic topic.
SUBJECT_MATCH_MIN_JACCARD: float = 0.3


def _normalize_subject_str(text: str) -> str:
    s = re.sub(r"[^\w\s]", " ", text.lower())
    s = re.sub(r"\s+", " ", s).strip()
    s = _LEADING_ARTICLE.sub("", s)
    toks = [t for t in s.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(toks)


def derive_subject(description: str) -> str:
    """Best-effort topic of a decision/constraint, normalized; '' if none found.

    Examples:
      "We will use bcrypt for password hashing."            -> "password hashing"
      "use argon2id for password hashing instead of bcrypt" -> "password hashing"
      "Do not use Celery, we will never revisit it."        -> "celery"
    """
    if not description:
        return ""
    m = _TOPIC_RE.search(description)
    if m:
        topic = _normalize_subject_str(m.group("topic"))
        if topic:
            return topic
    m = _PROHIBIT_RE.search(description)
    if m:
        thing = _normalize_subject_str(m.group("thing"))
        if thing:
            return thing
    return ""


def subject_supersedes(desc_a: str, desc_b: str) -> bool:
    """True when two descriptions share a derived subject and are related enough.

    Gated by SUBJECT_MATCH_MIN_JACCARD so a shared *generic* topic alone does not
    force a supersession between unrelated decisions.
    """
    sa = derive_subject(desc_a)
    if not sa or sa != derive_subject(desc_b):
        return False
    return jaccard_similarity(desc_a, desc_b) >= SUBJECT_MATCH_MIN_JACCARD


def supersedes(desc_a: str, desc_b: str) -> bool:
    """Combined supersession predicate: textual overlap OR same-subject (gated)."""
    return descriptions_overlap(desc_a, desc_b) or subject_supersedes(desc_a, desc_b)


def events_overlap(event_a: Event, event_b: Event) -> bool:
    """Return True if two events express the same concept in different words."""
    if event_a.event_type != event_b.event_type:
        return False
    desc_a = event_a.payload.get("description", "")
    desc_b = event_b.payload.get("description", "")
    return supersedes(desc_a, desc_b)


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
        if supersedes(new_desc, cand_desc):
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
