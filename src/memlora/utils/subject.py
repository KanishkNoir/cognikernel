"""Subject/topic normalization primitives — shared text utilities.

These were split out of ``delta.supersede`` so both the delta merge (supersession)
and the extraction stage (decision_key derivation) can normalize the *topic* a
decision is about without importing across each other — the extraction<->delta
layering cycle the architecture audit surfaced. Pure functions, no memlora deps.

``derive_subject`` extracts the stable topic of a choice (independent of which
specific option was picked); ``STOPWORDS`` is the shared content-token filter.
``delta.supersede`` re-exports both for backward compatibility.
"""
from __future__ import annotations

import re

STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
    "of", "and", "or", "but", "not", "we", "our", "can", "will",
    "are", "was", "be", "been", "has", "have", "had", "do", "does",
    "did", "no", "its", "use", "used", "using", "so", "that",
    "this", "with", "from", "as", "by", "if", "up", "out", "any",
})

_DECISION_VERB = (
    r"(?:use|using|adopt\w*|choos\w*|chose|switch\w*|stick\w*\s+with|"
    r"go\w*\s+with|prefer\w*|replac\w*|will\s+use|going\s+to\s+use|we'?ll\s+use)"
)
# A decision verb anywhere in the sentence qualifies it as a choice (so "for X"
# in arbitrary prose isn't mistaken for a topic).
_VERB_RE = re.compile(rf"\b{_DECISION_VERB}\b", re.IGNORECASE)
_TERMINATOR = r"(?:\binstead\b|\brather\b|\bbecause\b|\bsince\b|[.,;:]|$)"
# PREFERRED topic: the purpose/role the choice serves ("… for password hashing",
# "… as the cache"). This is the *stable* topic — it survives when the choice
# changes ("to argon2id") — and it matches whether it precedes or follows the verb
# ("For password hashing, use X" and "use X for password hashing"). Choosing this
# over the to/in object fixes "switch from bcrypt to argon2id for password hashing"
# (topic = "password hashing", not "argon2id").
_FOR_TOPIC_RE = re.compile(
    rf"\b(?:for|as)\s+(?P<topic>[a-z0-9][\w +./-]*?)\s*{_TERMINATOR}",
    re.IGNORECASE,
)
# Fallback topic: the object right after the verb via to/in ("moved to Postgres").
_TO_TOPIC_RE = re.compile(
    rf"\b{_DECISION_VERB}\b[\w './+-]*?\b(?:to|in)\s+"
    rf"(?P<topic>[a-z0-9][\w +./-]*?)\s*{_TERMINATOR}",
    re.IGNORECASE,
)
_PROHIBIT_RE = re.compile(
    r"\b(?:never\s+use|do\s+not\s+use|don'?t\s+use|avoid|abandon\w*|drop|reject\w*)\s+"
    r"(?P<thing>[a-z0-9][\w./+-]*)",
    re.IGNORECASE,
)
_LEADING_ARTICLE = re.compile(
    r"^(?:the|a|an|our|this|that|these|those)\s+", re.IGNORECASE
)


def _normalize_subject_str(text: str) -> str:
    s = re.sub(r"[^\w\s]", " ", text.lower())
    s = re.sub(r"\s+", " ", s).strip()
    s = _LEADING_ARTICLE.sub("", s)
    toks = [t for t in s.split() if t not in STOPWORDS and len(t) > 2]
    return " ".join(toks)


def derive_subject(description: str) -> str:
    """Best-effort topic of a decision/constraint, normalized; '' if none found.

    Extracts what the decision is *about* (the noun phrase following a choice verb
    + for/to/as/in), independent of which specific choice was made. Prohibition
    patterns extract the thing being rejected.
    """
    if not description:
        return ""
    # Require a decision verb so "for X" in arbitrary prose isn't read as a topic.
    if _VERB_RE.search(description):
        # Prefer the purpose topic (for/as) over the choice object (to/in).
        for rx in (_FOR_TOPIC_RE, _TO_TOPIC_RE):
            m = rx.search(description)
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
