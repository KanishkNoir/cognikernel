"""Content hash computation for event deduplication.

The hash encodes only stable identity fields: event_type and a normalized
key phrase from the description. Rationale, weight, and timestamps are
excluded so that re-statements of the same idea hash identically.
"""
from __future__ import annotations

import hashlib
import json
import re

_WHITESPACE = re.compile(r"\s+")
_PUNCTUATION = re.compile(r'[,.;:!?\'"()\[\]{}]')

# Lightweight contraction normalization — no NLTK required.
_CONTRACTIONS: list[tuple[str, str]] = [
    ("we will", "we'll"),
    ("cannot", "can't"),
    ("do not", "don't"),
    ("will not", "won't"),
    ("is not", "isn't"),
    ("are not", "aren't"),
    ("does not", "doesn't"),
    ("did not", "didn't"),
    ("have not", "haven't"),
    ("has not", "hasn't"),
    ("should not", "shouldn't"),
    ("would not", "wouldn't"),
    ("could not", "couldn't"),
    ("must not", "mustn't"),
]


def compute_content_hash(event_type: str, description: str) -> str:
    """Return SHA-256 hex digest for deduplication keyed on event_type + description."""
    canonical = {
        "event_type": event_type,
        "key_phrase": normalize_for_hash(description),
    }
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def normalize_for_hash(text: str) -> str:
    """Aggressively normalize text so re-statements of the same idea share a hash."""
    text = text.lower()
    # Normalize contractions before stripping punctuation.
    for expanded, contracted in _CONTRACTIONS:
        text = text.replace(expanded, contracted)
    text = _PUNCTUATION.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text
