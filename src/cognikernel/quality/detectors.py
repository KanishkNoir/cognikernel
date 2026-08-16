"""Pure defect detectors shared by the admission gate and the research audit.

One function per defect class from the spec taxonomy. Every detector is a pure
function of text (plus, where needed, the event type) — no I/O, no database, no
filesystem — so the audit script and the live gate cannot drift apart.

Detectors return None for "clean" and a DetectorHit for "flagged". They never
raise: a detector that cannot decide returns None, because the gate's failure
posture is to admit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DetectorHit:
    """A single defect finding. `rule_id` is the taxonomy id (D1..D7)."""
    rule_id: str
    note: str


# ── D7: subject-less statements ──────────────────────────────────────────────
#
# Tightened relative to the exploratory sweep regex, which flagged every leading
# "Do..." and so mislabelled well-formed imperative constraints. The signal we
# want is a *bare* pronoun subject — a pronoun followed directly by a verbal or
# adverbial, with no noun head to name the referent. "This migration is ..."
# keeps its referent and is therefore clean; "This only guarantees ..." does not.

_BARE_PRONOUN_SUBJECT = re.compile(
    r"^(?:this|that|these|those|it|they|he|she)\s+"
    r"(?:is|are|was|were|will|would|should|shall|must|can|could|may|might|"
    r"has|have|had|does|do|did|only|also|just|now|then|still|already|"
    r"means|makes|gives|guarantees|connects|requires|needs|lets|allows|"
    r"avoids|breaks|keeps|works|happens|applies|matters)\b",
    re.IGNORECASE,
)

_CONJUNCTION_OPENER = re.compile(
    r"^(?:and|but|so|then|also|however|therefore|thus|moreover|furthermore|"
    r"besides|otherwise|instead|meanwhile)\b[\s,]",
    re.IGNORECASE,
)


def detect_subject_less(text: str) -> DetectorHit | None:
    """D7 — the statement's subject exists only in unstated context."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    if _BARE_PRONOUN_SUBJECT.match(stripped):
        return DetectorHit("D7", "bare pronoun subject with no antecedent")
    if _CONJUNCTION_OPENER.match(stripped):
        return DetectorHit("D7", "discourse-connective opener")
    return None


# ── D2: junk stored as a constraint ──────────────────────────────────────────

_BOX_DRAWING = re.compile(r"[─-╿▀-▟]")

# Public alias — the render-time invariant check in injection/template.py
# reuses this rather than defining a second box-drawing pattern.
BOX_DRAWING_RE = _BOX_DRAWING

# A question is identified ONLY by a trailing '?'. Opener-based detection was
# tried and rejected: every candidate opener also begins well-formed
# declaratives in this corpus.
#   "Do not cache in Redis."                     imperative, not interrogative
#   "What must be redacted: message content, …"  label-list, not interrogative
# Both were flagged by an opener heuristic during verification. Requiring the
# '?' costs recall on unpunctuated questions and is the deliberate
# precision-first trade — it also matches the precedent already set by
# sanitize.is_question_description, which requires a question terminator and
# then excludes declaratives.
_CONSTRAINT_TYPES = frozenset({"CONSTRAINT_HARD", "CONSTRAINT_SOFT"})

_NON_ALPHA_MAX = 0.30


def _non_alpha_ratio(text: str) -> float:
    """Share of non-whitespace characters that are not letters."""
    body = [c for c in text if not c.isspace()]
    if not body:
        return 1.0
    return sum(1 for c in body if not c.isalpha()) / len(body)


def detect_junk_constraint(text: str, event_type: str) -> DetectorHit | None:
    """D2 — a constraint slot holding something that is not a proposition."""
    stripped = (text or "").strip()
    if not stripped or event_type not in _CONSTRAINT_TYPES:
        return None
    if _BOX_DRAWING.search(stripped):
        return DetectorHit("D2", "box-drawing/table artifact")
    if stripped.endswith("?"):
        return DetectorHit("D2", "interrogative stored as a constraint")
    if _non_alpha_ratio(stripped) > _NON_ALPHA_MAX:
        return DetectorHit("D2", "predominantly non-alphabetic")
    return None


# ── D4: harness boilerplate captured as memory ───────────────────────────────
#
# Phrases emitted by the agent harness (compaction summaries, resume banners),
# by CogniKernel's own injected block, or by the agent narrating a check of
# CogniKernel's own state (e.g. echoing CLAUDE.md/trust-header "greenfield"
# guidance back as if it were a project fact). None of these are project facts.

_BOILERPLATE = re.compile(
    r"read the full transcript at|continue the conversation from|"
    r"resume directly|do not acknowledge the summary|"
    r"do not recap what was happening|pick up the last task|"
    r"session context \[auto-generated|as if the break never happened|"
    r"no prior decisions stored|confirmed greenfield",
    re.IGNORECASE,
)


def detect_boilerplate(text: str) -> DetectorHit | None:
    """D4 — harness/compaction chatter, or CogniKernel's own injected block."""
    stripped = (text or "").strip()
    if not stripped:
        return None
    if _BOILERPLATE.search(stripped):
        return DetectorHit("D4", "harness or compaction boilerplate")
    return None


# ── D5: cross-type duplicates ────────────────────────────────────────────────
#
# Content-hash dedup is per (event_type, description), so the SAME fact stored
# under two types survives as two rows and renders in two sections. This key is
# type-independent: equal keys mean "same statement", whatever the type.

_NON_KEY_CHARS = re.compile(r"[^a-z0-9 ]")
_WHITESPACE = re.compile(r"\s+")


def normalized_key(text: str) -> str:
    """Type-independent identity key for a statement.

    Lowercase, strip punctuation, collapse whitespace. Equal keys across two
    different event types is exactly the D5 defect.
    """
    lowered = (text or "").lower()
    stripped = _NON_KEY_CHARS.sub(" ", lowered)
    return _WHITESPACE.sub(" ", stripped).strip()
