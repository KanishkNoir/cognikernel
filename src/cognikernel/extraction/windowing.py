"""Sliding-window context extraction.

For each trie match, expands a context window to capture the description
(the matching sentence) and the rationale (surrounding prose that explains
why the decision or constraint exists).
"""
from __future__ import annotations

import re

from cognikernel.extraction.authority import (
    ASSISTANT_ANSWER_TO_QUESTION,
    default_authority_for_role,
)
from cognikernel.extraction.sanitize import (
    is_question_description,
    sanitize_description,
    sanitize_rationale,
)
from cognikernel.extraction.tokenize import Sentence
from cognikernel.extraction.trie import TrieMatch
from cognikernel.storage.events import Event

# Sentences that start with workflow narration ("Let me…", "Now update…", "I'll…")
# are Claude talking to itself mid-implementation — not architectural decisions.
_NARRATION_PREFIXES = re.compile(
    r"^(?:now\s+(?:let\s+me|let'?s|i'?ll|implement|update|writ|add|read|check|look)"
    r"|let\s+me|let'?s|i'?ll\s|i\s+will\s|first[,\s]|starting\s+with"
    r"|next[,\s]|then\s+(?:let|i|we)|going\s+to\s)",
    re.IGNORECASE,
)

# References to CogniKernel tooling or CLAUDE.md instructions — meta-talk, not decisions.
# F3: also catch "session context" and "already decided" — the assistant narrating
# *about* injected memory ("From the session context, this is already decided…").
_META_REFERENCES = re.compile(
    r"\b(?:get_session_state|mcp__|claude\.md|session\s+(?:memory|context)"
    r"|as\s+required\s+by|per\s+claude\.md|cognikernel|cognikernel|already\s+decided)\b",
    re.IGNORECASE,
)

# F3: status narration — momentary assistant chatter ("I need to check…", "Reading
# the body…", "Already fully implemented", "Nothing to do here."). These trip the
# high-frequency THREAD_* signals (need to / to do / implemented / done) but are
# not durable memory. First-person singular only — "We need to…" is a real thread.
_STATUS_NARRATION = re.compile(
    r"^(?:i\s+need\s+to|i\s+can(?:'?t|not)?\b|i'?m\b|i\s+have\b|i'?ve\b"
    r"|reading\b|already\b|nothing\b|checking\b|looking\b|here'?s\b)",
    re.IGNORECASE,
)

# F3: a quick proxy for "substance" — alphanumeric tokens of length >= 3. Used to
# drop empty fragments ("---") and one-word THREAD statuses ("Done.").
_CONTENT_WORD = re.compile(r"[a-z0-9]{3,}")


def _content_word_count(text: str) -> int:
    return len(_CONTENT_WORD.findall(text.lower()))


def _is_noise_description(description: str) -> bool:
    """Type-independent noise filter (F3 + earlier guards).

    Empty fragments, status narration, workflow narration, meta-talk about injected
    memory, and implementation-review sentences are momentary chatter, not durable
    memory. Shared by the trie path and the co-capture path so neither leaks noise.
    """
    if _content_word_count(description) == 0:
        return True
    if _STATUS_NARRATION.match(description):
        return True
    if _NARRATION_PREFIXES.match(description):
        return True
    if _META_REFERENCES.search(description):
        return True
    if _IMPLEMENTATION_REVIEW.match(description):
        return True
    return False

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
    r"|the\s+(?:[\w-]+\s+){1,4}(?:is|are|was|were)\s+(?:\w+\s+){0,3}"  # "The login endpoint is already fully implemented..."
    r"(?:handled|done|managed|triggered|implemented|synced|wired|complete|finished|ready)"
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

# Structural-label detector — section headers and category labels whose name
# re-uses a signal phrase (e.g. "Explicitly abandoned approaches:" hits the
# "explicitly abandoned" signal). The label NAMES a category; it is not an
# instance of the category. Must be checked on the RAW sentence text before
# sanitization strips the markdown markers that prove it's a label.
_LABEL_MAX_WORDS = 10
_LABEL_HEADING = re.compile(r"^#{1,6}\s+(.+)$")
_LABEL_BOLD_WRAP = re.compile(r"^\*\*(.+?)\*\*\s*:?\s*$")


def _is_structural_label(raw_text: str) -> bool:
    """Return True if the raw sentence is a section header / list label."""
    s = raw_text.strip()
    if not s:
        return False

    heading = _LABEL_HEADING.match(s)
    if heading:
        body = heading.group(1).rstrip(": ").strip()
    else:
        bold = _LABEL_BOLD_WRAP.match(s)
        if bold:
            body = bold.group(1).rstrip(": ").strip()
        elif s.endswith(":"):
            body = s[:-1].strip()
        else:
            return False

    word_count = len(body.split())
    return 0 < word_count <= _LABEL_MAX_WORDS


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

        # Drop matches landing in a section header / category label. Must run
        # on the raw description — sanitization strips the `##` and `**` markers
        # that are the structural evidence this sentence is a label, not content.
        if _is_structural_label(description):
            continue

        description = sanitize_description(description)
        rationale   = sanitize_rationale(rationale)

        if not description:
            continue

        # Type-independent noise filter (F3): empty fragments, status/workflow
        # narration, meta-talk, implementation-review. Shared with the co-capture path.
        if _is_noise_description(description):
            continue

        source_role = sentences[match.sentence_index].role

        # Downgrade questions masquerading as constraints.
        event_type = match.signal_type
        confidence = match.confidence
        if event_type == "CONSTRAINT_HARD" and is_question_description(description):
            event_type = "CONSTRAINT_SOFT"
            confidence = min(confidence, 0.3)

        # Downgrade descriptive "never" hits — third-person subject describes what
        # code currently does, not a normative rule. e.g. "routes never touch X".
        if event_type == "CONSTRAINT_HARD" and match.matched_phrase == "never":
            if _DESCRIPTIVE_NEVER.search(description):
                event_type = "CONSTRAINT_SOFT"
                confidence = min(confidence, 0.3)

        # F4: a DECISION stated in a USER turn is a first-class, highest-authority
        # decision (authority=user_stated via default_authority_for_role) — keep it.
        # Previously user-turn DECISIONs were blanket-dropped, which silently lost
        # user-stated decisions like "we're switching from bcrypt to argon2id" and
        # left supersession nothing to link. Still drop imperative prompt echoes
        # ("implement X", "decide on Y") and questions, which assert no decision.
        if event_type == "DECISION":
            if _USER_PROMPT_ECHO.match(description) or is_question_description(description):
                continue

        # F3: a THREAD event must name a work item with some substance — a one-word
        # status ("Done.") is narration, not a durable open/closed thread.
        if event_type in ("THREAD_OPEN", "THREAD_CLOSE") and _content_word_count(description) < 2:
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
                    "authority": default_authority_for_role(source_role),
                    "provenance": "trie",
                },
                content_hash="",   # populated by hashing stage
                weight=confidence,
            )
        )

    return events


# ── co-capture (A-4) ─────────────────────────────────────────────────────────


_MAX_ASSISTANT_COCAPTURE_SENTENCES = 2


def extract_co_captures(
    sentences: list[Sentence],
    matches: list[TrieMatch],
    project_id: str,
    session_id: str,
) -> list[Event]:
    """For each trie match on a USER sentence, capture the next assistant
    sentences and produce a co-capture Event.

    The co-capture event holds the assistant's response as its description with
    `authority = ASSISTANT_ANSWER_TO_QUESTION`. The renderer routes these to a
    separate `### Pending confirmation` section unless suppressed by a later
    user_stated / assistant_decided event with the same normalized subject.

    Design notes:
      - Only ONE co-capture per assistant turn even if multiple user matches
        precede it (deduped by (session_id, assistant turn start index)).
      - Assistant code blocks are skipped — they're implementation, not answers.
      - The original trie matches are unaffected; this is purely additive.
    """
    if not sentences:
        return []

    events: list[Event] = []
    seen_assistant_starts: set[int] = set()

    # Pre-compute trie-matched sentence indices for fast membership.
    user_match_indices: set[int] = set()
    for m in matches:
        if m.sentence_index >= len(sentences):
            continue
        if sentences[m.sentence_index].role == "user":
            user_match_indices.add(m.sentence_index)

    for user_idx in sorted(user_match_indices):
        # Walk forward to find the start of the next assistant turn.
        assistant_start = _find_next_assistant_start(sentences, user_idx)
        if assistant_start is None or assistant_start in seen_assistant_starts:
            continue
        seen_assistant_starts.add(assistant_start)

        captured = _capture_assistant_sentences(
            sentences, assistant_start, _MAX_ASSISTANT_COCAPTURE_SENTENCES,
        )
        if not captured.strip():
            continue

        description = sanitize_description(captured)
        if not description:
            continue

        # F3: co-captures bypass the trie-path filters, so apply the shared noise
        # filter here too — an assistant reply that just narrates about the session
        # context ("From the session context, this is already decided…") is not memory.
        if _is_noise_description(description):
            continue

        events.append(
            Event(
                project_id=project_id,
                session_id=session_id,
                # CONSTRAINT_SOFT keeps confidence honest — the user hasn't
                # confirmed yet, so the event should never gate the projection.
                event_type="CONSTRAINT_SOFT",
                payload={
                    "description": description,
                    "rationale": "",
                    "confidence": 0.5,
                    "source_role": "assistant",
                    "matched_phrase": "CO_CAPTURE",
                    "affected_files": [],
                    "authority": ASSISTANT_ANSWER_TO_QUESTION,
                    "answers_user_sentence_index": user_idx,
                    "provenance": "co_capture",
                },
                content_hash="",
                weight=0.5,
            )
        )

    return events


def _find_next_assistant_start(
    sentences: list[Sentence], from_index: int,
) -> int | None:
    """Return the index of the first assistant non-code sentence after
    `from_index`, or None if there isn't one within the same exchange."""
    for j in range(from_index + 1, len(sentences)):
        s = sentences[j]
        if s.role == "assistant" and not s.is_code_block:
            return j
        if s.role == "user" and j > from_index + 1:
            # Next user turn started before we found an assistant reply.
            return None
    return None


def _capture_assistant_sentences(
    sentences: list[Sentence], start_index: int, limit: int,
) -> str:
    """Concatenate up to `limit` consecutive assistant prose sentences."""
    parts: list[str] = []
    for j in range(start_index, len(sentences)):
        s = sentences[j]
        if s.role != "assistant":
            break
        if s.is_code_block:
            continue
        text = s.text.strip()
        if not text:
            continue
        parts.append(text)
        if len(parts) >= limit:
            break
    return " ".join(parts)
