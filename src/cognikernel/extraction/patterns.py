"""Pattern-with-capture engine — Phase A-3.

The Aho-Corasick trie in `trie.py` matches literal phrases efficiently but
cannot extract arguments. Pattern matching covers the other half of the
recall ceiling: phrases like "use {X}" and "no {X}, no {Y}" where the
*subject* is what we want, not the verb.

This module:
  1. Defines `Pattern` dataclass with regex + role filter + sentence-shape guard
  2. Provides `scan_patterns()` which iterates sentences and emits Events with
     a `subject` payload field (separate from `description`)
  3. Applies precision guards from the v2 plan so noisy hits (e.g. "use" inside
     a code-review explanation) don't masquerade as decisions

Patterns produce raw Events; downstream `classify_event` still applies the
existing source-role bonus, and `normalize_description` still strips leftover
prompt verbs. The trie and the pattern engine are complementary, not
competitive — both produce events into the same pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from cognikernel.extraction.authority import default_authority_for_role
from cognikernel.extraction.tokenize import Sentence
from cognikernel.storage.events import Event


# ── data model ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PatternMatch:
    sentence_index: int
    pattern_id: str
    event_type: str
    confidence: float
    subject: str       # captured group: the X in "use X" or "no X"


@dataclass(frozen=True)
class Pattern:
    pattern_id: str
    regex: re.Pattern
    event_type: str
    base_confidence: float
    role_filter: tuple[str, ...] = ()       # () = accept all roles
    shape_guard: Callable[[str], bool] | None = None

    def role_ok(self, role: str) -> bool:
        return not self.role_filter or role in self.role_filter

    def shape_ok(self, sentence_text: str) -> bool:
        return self.shape_guard is None or self.shape_guard(sentence_text)


# ── shape guards ─────────────────────────────────────────────────────────────


def _starts_with_use(text: str) -> bool:
    """`use {X}`, `we'll use {X}`, `let's use {X}` — imperative-style openings.

    Distinguishes 'Use PostgreSQL' (a directive) from 'I'll use that later'
    (narration). The guard requires the *sentence* to open with the directive.
    """
    s = text.lstrip().lower()
    return (
        s.startswith("use ")
        or s.startswith("we'll use ")
        or s.startswith("we will use ")
        or s.startswith("let's use ")
        or s.startswith("let us use ")
    )


def _not_a_question(text: str) -> bool:
    """Bare 'no X' clauses inside questions are not constraints."""
    return not text.rstrip().endswith("?")


def _starts_with_only(text: str) -> bool:
    """'Only X' as a directive — exclusivity rule, not a relative qualifier."""
    return text.lstrip().lower().startswith("only ")


def _has_parenthetical_not(text: str) -> bool:
    """Parenthetical negation `(not X)` — common stack-proposal shorthand
    e.g. 'production-ready SQL (not SQLite)'."""
    return "(not " in text.lower()


# ── patterns ─────────────────────────────────────────────────────────────────


# `use {X}` — single-token or short noun phrase after the verb.
# Subject = next 1-3 tokens, alphanumeric + dots/hyphens (library names, etc.)
_RE_USE = re.compile(
    r"^\s*(?:(?:we'll|we will|let's|let us)\s+)?use\s+"
    r"(?P<subject>[A-Za-z][\w.+/-]*(?:\s+[A-Za-z][\w.+/-]*){0,2})",
    re.IGNORECASE,
)

# `no {X}, no {Y}` — multi-negation, strong signal of explicit rejection.
# Subjects are bounded by `,` or `.` and the second word of a two-word subject
# cannot be a preposition — guards against "no Chakra in this project"
# capturing "Chakra in".
_PREPOSITION_BLOCKLIST = r"(?:in|for|with|at|by|on|to|of|from|into|via|under)"
_RE_NO_MULTI = re.compile(
    r"\bno\s+(?P<x>[A-Za-z][\w./+-]*"
    rf"(?:\s+(?!{_PREPOSITION_BLOCKLIST}\b)[A-Za-z][\w./+-]*)?)"
    r"\s*,\s*"
    r"no\s+(?P<y>[A-Za-z][\w./+-]*"
    rf"(?:\s+(?!{_PREPOSITION_BLOCKLIST}\b)[A-Za-z][\w./+-]*)?)",
    re.IGNORECASE,
)

# `no {X}` — single negation, weaker signal; require sentence-start to avoid
# matching narrative "...there's no time to..." style text.
_RE_NO_SINGLE = re.compile(
    r"^\s*no\s+(?P<subject>[A-Za-z][\w./+-]*(?:\s+[A-Za-z][\w./+-]*)?)",
    re.IGNORECASE,
)

# `(not {X})` — parenthetical negation common in stack proposals
# e.g. "production-ready SQL (not SQLite)".
_RE_PAREN_NOT = re.compile(
    r"\(\s*not\s+(?P<subject>[A-Za-z][\w./+-]*(?:\s+[A-Za-z][\w./+-]*)?)\s*\)",
    re.IGNORECASE,
)

# `only {X}` — exclusivity directive, must open the sentence.
_RE_ONLY = re.compile(
    r"^\s*only\s+(?P<subject>[A-Za-z][\w./+-]*(?:\s+[A-Za-z][\w./+-]*){0,2})",
    re.IGNORECASE,
)

# ── F6: declarative convention/config facts (no decision verb) ────────────────
# Conventions stated as plain facts ("URL prefix: /api/v1/", "JSON fields:
# camelCase", "Component library: shadcn/ui") carry no decision verb, so the trie
# misses them entirely — the D6/D7/D8 recall gaps. These patterns are keyed on
# UNAMBIGUOUS tokens (URL paths, casing keywords, a whitelist of stack labels) so
# recall rises without re-introducing the prose noise F3 removed. The full sentence
# is kept as the description, so the verbatim convention is recallable.

# D8 — API URL prefix. Two orders observed in real transcripts:
#   Form A (keyword → path): "URL prefix: /api/v1/", "routes mount under /api/v1".
_RE_URL_PREFIX = re.compile(
    r"\b(?:url\s+prefix|api\s+prefix|route\s+prefix|prefix(?:ed)?|versioned\s+under|"
    r"(?:routes?|endpoints?)\s+(?:\w+\s+){0,2}under|mount(?:ed)?\s+(?:under|at)|"
    r"namespaced?\s+under)\b[^A-Za-z0-9/\n]{0,8}(?P<subject>/[A-Za-z0-9][\w./-]*)",
    re.IGNORECASE,
)
#   Form B (path → keyword): "`/api/v1` as a mount prefix on the FastAPI app".
_RE_URL_PREFIX_REV = re.compile(
    r"(?P<subject>/[A-Za-z0-9][\w./-]*)[^A-Za-z0-9\n]{0,4}"
    r"(?:as\s+(?:a\s+|the\s+)?)?(?:mount\s+|url\s+|route\s+|api\s+)?prefix\b",
    re.IGNORECASE,
)

# D6 — casing convention: snake_case / camelCase / PascalCase / kebab-case.
_RE_CASING = re.compile(
    r"\b(?P<subject>snake_case|camelCase|PascalCase|kebab-case|SCREAMING_SNAKE_CASE)\b",
    re.IGNORECASE,
)

# D7 (+ stack) — "Label: value" config lines for a whitelist of stack categories.
_STACK_LABEL = (
    r"component\s+library|ui\s+library|ui\s+components?|framework|database|orm|"
    r"migrations?|state\s+management|styling|http\s+client|server\s+state|"
    r"client\s+state|forms?|build\s+tool|password\s+hashing|background\s+tasks?|"
    r"url\s+prefix|api\s+prefix|token\s+store|caching|pagination|jwt\s+secret"
)
_RE_CONFIG_LINE = re.compile(
    rf"^\s*[-*|]?\s*\*{{0,2}}\s*(?P<label>{_STACK_LABEL})\s*\*{{0,2}}\s*:\s*\*{{0,2}}\s*"
    r"(?P<subject>[A-Za-z0-9/][^\n|]{0,48}?)\s*(?:[.|]\s*|$)",
    re.IGNORECASE,
)


PATTERNS: tuple[Pattern, ...] = (
    Pattern(
        pattern_id="USE_X",
        regex=_RE_USE,
        event_type="DECISION",
        base_confidence=0.6,
        role_filter=("user", "assistant"),
        shape_guard=_starts_with_use,
    ),
    Pattern(
        pattern_id="NO_MULTI",
        regex=_RE_NO_MULTI,
        event_type="CONSTRAINT_HARD",
        base_confidence=0.85,
        role_filter=("user",),
        shape_guard=_not_a_question,
    ),
    Pattern(
        pattern_id="NO_SINGLE",
        regex=_RE_NO_SINGLE,
        event_type="CONSTRAINT_HARD",
        base_confidence=0.5,
        role_filter=("user",),
        shape_guard=_not_a_question,
    ),
    Pattern(
        pattern_id="PAREN_NOT",
        regex=_RE_PAREN_NOT,
        event_type="CONSTRAINT_HARD",
        base_confidence=0.7,
        role_filter=("user",),
        shape_guard=_has_parenthetical_not,
    ),
    Pattern(
        pattern_id="ONLY_X",
        regex=_RE_ONLY,
        event_type="CONSTRAINT_HARD",
        base_confidence=0.7,
        role_filter=("user",),
        shape_guard=_starts_with_only,
    ),
    # F6: declarative convention/config facts (stated by user or assistant).
    Pattern(
        pattern_id="URL_PREFIX",
        regex=_RE_URL_PREFIX,
        event_type="CONSTRAINT_SOFT",
        base_confidence=0.7,
        role_filter=("user", "assistant"),
        shape_guard=_not_a_question,
    ),
    Pattern(
        pattern_id="URL_PREFIX_REV",
        regex=_RE_URL_PREFIX_REV,
        event_type="CONSTRAINT_SOFT",
        base_confidence=0.7,
        role_filter=("user", "assistant"),
        shape_guard=_not_a_question,
    ),
    Pattern(
        pattern_id="CASING",
        regex=_RE_CASING,
        event_type="CONSTRAINT_SOFT",
        base_confidence=0.65,
        role_filter=("user", "assistant"),
        shape_guard=_not_a_question,
    ),
    Pattern(
        pattern_id="CONFIG_LINE",
        regex=_RE_CONFIG_LINE,
        event_type="DECISION",
        base_confidence=0.6,
        role_filter=("user", "assistant"),
        shape_guard=_not_a_question,
    ),
)


# ── scanner ──────────────────────────────────────────────────────────────────


def scan_patterns(
    sentences: list[Sentence],
    project_id: str,
    session_id: str,
) -> list[Event]:
    """Run every pattern against every (non-code) sentence and emit Events.

    Code-block sentences are skipped, mirroring the trie's behavior.

    Overlap suppression: NO_MULTI is a strict superset of NO_SINGLE for the
    same sentence — if NO_MULTI fires on sentence S, NO_SINGLE on the same
    sentence is suppressed so the multi-negation rejection isn't double-counted.
    """
    events: list[Event] = []

    for i, sentence in enumerate(sentences):
        if sentence.is_code_block:
            continue
        text = sentence.text
        if not text.strip():
            continue

        # Track whether NO_MULTI matched this sentence so NO_SINGLE can defer.
        no_multi_fired = False

        for pattern in PATTERNS:
            if pattern.pattern_id == "NO_SINGLE" and no_multi_fired:
                continue
            if not pattern.role_ok(sentence.role):
                continue
            if not pattern.shape_ok(text):
                continue

            fired_this_pattern = False
            for match in pattern.regex.finditer(text):
                subjects = _extract_subjects(pattern, match)
                for subject in subjects:
                    cleaned = _normalize_subject(subject)
                    if not _subject_is_meaningful(cleaned):
                        continue
                    events.append(
                        _build_event(
                            pattern=pattern,
                            sentence=sentence,
                            sentence_index=i,
                            subject=cleaned,
                            project_id=project_id,
                            session_id=session_id,
                        )
                    )
                    fired_this_pattern = True

            if pattern.pattern_id == "NO_MULTI" and fired_this_pattern:
                no_multi_fired = True

    return events


# ── helpers ──────────────────────────────────────────────────────────────────


def _extract_subjects(pattern: Pattern, match: re.Match) -> list[str]:
    """Extract one or more captured subjects from a regex match.

    Multi-negation patterns capture two; everything else captures one.
    """
    if pattern.pattern_id == "NO_MULTI":
        return [match.group("x").strip(), match.group("y").strip()]
    if "subject" in match.groupdict():
        return [match.group("subject").strip()]
    return []


# Trailing punctuation that callers don't want in the subject. Patterns
# match up to a sentence terminator so the terminator can leak into the
# capture; strip it here so downstream lookups don't see "PostgreSQL.".
_TRAILING_PUNCT = ".,;:!?*`"


def _normalize_subject(subject: str) -> str:
    """Strip trailing punctuation/whitespace from a captured subject."""
    return subject.strip().rstrip(_TRAILING_PUNCT).strip()


# Words that are common-English-noise rather than real subjects.
_STOPWORD_SUBJECTS = frozenset({
    "the", "a", "an", "of", "for", "to", "and", "or", "with", "by",
    "this", "that", "these", "those", "it", "them", "us",
    "way", "ways", "time", "times", "thing", "things",
})


def _subject_is_meaningful(subject: str) -> bool:
    """Drop subjects that are stopwords or single common words.

    Single short words like "the", "a", "way" are extraction noise — they
    indicate the regex captured the start of a clause rather than a real
    decision target.
    """
    s = subject.strip().lower()
    if not s:
        return False
    if s in _STOPWORD_SUBJECTS:
        return False
    # A multi-word subject with stopword leading char (e.g. "the new system")
    # is borderline — keep it for now since classifier confidence will handle.
    return True


def _build_event(
    *,
    pattern: Pattern,
    sentence: Sentence,
    sentence_index: int,
    subject: str,
    project_id: str,
    session_id: str,
) -> Event:
    """Convert a single pattern hit into a raw Event ready for normalization."""
    # The description carries the full sentence so downstream sanitization
    # and rendering have context. The subject lives separately in payload.
    return Event(
        project_id=project_id,
        session_id=session_id,
        event_type=pattern.event_type,
        payload={
            "description": sentence.text.strip(),
            "rationale": "",
            "subject": subject,
            "confidence": pattern.base_confidence,
            "source_role": sentence.role,
            "matched_phrase": pattern.pattern_id,
            "affected_files": [],
            "authority": default_authority_for_role(sentence.role),
            "provenance": "pattern",
        },
        content_hash="",   # populated by hashing stage
        weight=pattern.base_confidence,
    )
