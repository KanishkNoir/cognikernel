"""Sliding-window context extraction.

For each trie match, expands a context window to capture the description
(the matching sentence) and the rationale (surrounding prose that explains
why the decision or constraint exists).
"""
from __future__ import annotations

import re

from memlora.extraction.sanitize import (
    is_question_description,
    sanitize_description,
    sanitize_rationale,
)
from memlora.extraction.tokenize import Sentence
from memlora.extraction.trie import TrieMatch
from memlora.storage.events import Event

# Sentences that start with workflow narration ("Let me…", "Now update…", "I'll…")
# are Claude talking to itself mid-implementation — not architectural decisions.
_NARRATION_PREFIXES = re.compile(
    r"^(?:now\s+(?:let\s+me|let'?s|i'?ll|implement|update|writ|add|read|check|look)"
    r"|let\s+me|let'?s|i'?ll\s|i\s+will\s|first[,\s]|starting\s+with"
    r"|next[,\s]|then\s+(?:let|i|we)|going\s+to\s)",
    re.IGNORECASE,
)

# References to CogniKernel tooling or CLAUDE.md instructions — meta-talk, not decisions.
_META_REFERENCES = re.compile(
    r"\b(?:get_session_state|mcp__|claude\.md|session\s+memory"
    r"|as\s+required\s+by|per\s+claude\.md|cognikernel|memlora)\b",
    re.IGNORECASE,
)

# Verbatim user-prompt fragments mistaken for decisions.
_USER_PROMPT_ECHO = re.compile(
    r"^(?:decide\s+on\b|implement\s+\w|explain\s+your\s+reasoning)",
    re.IGNORECASE,
)

# Assistant code-review and implementation-summary statements — these describe what
# the code does, not what it must do. They trigger signals ("never", "where possible")
# but are not architectural rules.
_IMPLEMENTATION_REVIEW = re.compile(
    r"^(?:"
    r"good\s*[—\-]"                                              # "Good — crud.py only uses..."
    r"|note\s+that\b"                                            # "Note that is_deleted is..."
    r"|making\s+all\b"                                           # "Making all five changes..."
    r"|all\s+\d+\s+tests?\s+(?:pass|fail)"                       # "All 18 tests pass..."
    r"|the\s+(?:[\w-]+\s+){1,4}(?:is|are|was|were)\s+"           # "The soft-delete sync is handled..."
    r"(?:handled|done|managed|triggered|implemented|synced|wired)"
    r")",
    re.IGNORECASE,
)

# "never" in a sentence where the grammatical subject is a code entity (third-person).
# These are descriptive ("routes never touch X") not normative ("we will never do X").
_DESCRIPTIVE_NEVER = re.compile(
    r"(?:it|the\s+\w+|routes?|triggers?|endpoints?|"
    r"crud(?:\.py)?|schemas?|models?|handlers?|middleware|"
    r"functions?|queries|columns?|tables?)\s+(?:\w+\s+){0,3}never\b",
    re.IGNORECASE,
)

# Connective words at the end of a preceding sentence indicate setup/reasoning.
_BACKWARD_CONNECTIVES = ("because", "since", "so that", "given that", "as a result")
_MAX_BACKWARD = 5
_MAX_FORWARD = 5
_DEFAULT_BEFORE = 2
_DEFAULT_AFTER = 2


def extract_window(
    sentences: list[Sentence], match_index: int, signal_type: str
) -> tuple[str, str]:
    """Return (description, rationale) for the match at match_index.

    description — text of the matching sentence.
    rationale   — surrounding context sentences joined into one string.
    """
    n = len(sentences)
    start = max(0, match_index - _DEFAULT_BEFORE)
    end = min(n, match_index + _DEFAULT_AFTER + 1)

    # Expand backward through connective sentences that set up the context.
    while start > 0 and (match_index - start) < _MAX_BACKWARD:
        prev = sentences[start - 1]
        prev_lower = prev.text.lower().rstrip(" .!?,;:")
        if any(prev_lower.endswith(c) for c in _BACKWARD_CONNECTIVES):
            start -= 1
        elif not prev.text.rstrip().endswith((".", "!", "?", ":")):
            start -= 1  # incomplete sentence — pull it in
        else:
            break

    # Expand forward through code blocks that follow the match.
    while end < n and sentences[end].is_code_block and (end - match_index) < _MAX_FORWARD:
        end += 1

    # Hard constraints on bullet lines are self-contained — narrow the window.
    if signal_type == "CONSTRAINT_HARD":
        if sentences[match_index].text.lstrip().startswith(("-", "*", "•")):
            start = match_index
            end = match_index + 1

    description = sentences[match_index].text.strip()
    rationale_parts = [
        s.text.strip()
        for s in sentences[start:end]
        if s.sentence_index != match_index and s.text.strip()
    ]
    rationale = " ".join(rationale_parts)

    return description, rationale


def extract_events_from_matches(
    sentences: list[Sentence],
    matches: list[TrieMatch],
    project_id: str,
    session_id: str,
) -> list[Event]:
    """Convert trie matches into un-hashed Event objects ready for classification."""
    events: list[Event] = []

    for match in matches:
        if match.sentence_index >= len(sentences):
            continue

        description, rationale = extract_window(
            sentences, match.sentence_index, match.signal_type
        )
        description = sanitize_description(description)
        rationale   = sanitize_rationale(rationale)

        if not description:
            continue

        source_role = sentences[match.sentence_index].role

        # Downgrade questions masquerading as constraints.
        event_type = match.signal_type
        confidence = match.confidence
        if event_type == "CONSTRAINT_HARD" and is_question_description(description):
            event_type = "CONSTRAINT_SOFT"
            confidence = min(confidence, 0.3)

        # Drop workflow narration and meta-talk regardless of event type.
        if _NARRATION_PREFIXES.match(description) or _META_REFERENCES.search(description):
            continue

        # Drop assistant code-review and implementation-summary sentences.
        if _IMPLEMENTATION_REVIEW.match(description):
            continue

        # Downgrade descriptive "never" hits — third-person subject describes what
        # code currently does, not a normative rule. e.g. "routes never touch X".
        if event_type == "CONSTRAINT_HARD" and match.matched_phrase == "never":
            if _DESCRIPTIVE_NEVER.search(description):
                event_type = "CONSTRAINT_SOFT"
                confidence = min(confidence, 0.3)

        # Drop DECISION events from non-assistant turns or echoed user prompts.
        if event_type == "DECISION":
            if source_role != "assistant" or _USER_PROMPT_ECHO.match(description):
                continue

        events.append(
            Event(
                project_id=project_id,
                session_id=session_id,
                event_type=event_type,
                payload={
                    "description": description,
                    "rationale": rationale,
                    "confidence": confidence,
                    "source_role": source_role,
                    "matched_phrase": match.matched_phrase,
                    "affected_files": [],
                },
                content_hash="",   # populated by hashing stage
                weight=confidence,
            )
        )

    return events
