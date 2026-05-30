"""CKL V2 triple extraction — runs at event creation time (Phase B architecture).

Enriches event payloads with a structured triple (operator, subject, object)
when a reliable pattern is detected from the sanitized description. Events
without a recognised pattern render as CKL V1 prose at injection time.

Operator set (position-based, all have high semantic stability in LLM training):
  ¬   prohibition / must-not        (before object: ¬ SQL_DELETE)
  →   implication / dependency      (X → Y: "X uses Y")
  ←   derived-from / source-of      (reserved — not used by current patterns)
  ∅   null / rejected / empty       (subject ∅: "<approach> ∅")
  |   or / alternatives             (reserved — compositional future use)
  &   and / conjunction             (reserved — compositional future use)
  =   equals / is-defined-as        (reserved — compositional future use)
  :   type-annotation / property-of (reserved — compositional future use)

Pattern coverage:
  CONSTRAINT_HARD                   — negation patterns (never, must not, do not …)
  DECISION                          — implication patterns (uses, via, use X for Y …)
  APPROACH_ABANDONED_DO_NOT_RETRY   — null pattern (subject split on " — " separator)
  APPROACH_ABANDONED                — null pattern (same as above)

All other event types pass through unchanged (no triple added).
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlora.storage.events import Event

# ── operator constants ────────────────────────────────────────────────────────

OP_NEG  = "¬"
OP_IMPL = "→"
OP_FROM = "←"
OP_NULL = "∅"

# ── filler / negation patterns ────────────────────────────────────────────────

# Optional subject filler: "we", "the system", "this", "it", "you", "our X"
_FILLER_SUBJ = re.compile(
    r'^(?:(?:we|you|this|it|the\s+[\w-]+|our\s+[\w-]+)\s+)?'
    r'(?:(?:will|should|must|always|need\s+to|have\s+to)\s+)?',
    re.IGNORECASE,
)

# Negation verbs: strip from start of description after filler
_NEG_VERB = re.compile(
    r'^(?:never|not|do\s+not|don\'t|must\s+not|cannot|can\'t|avoid\s+(?:using?\s+)?|no)\s+',
    re.IGNORECASE,
)

# ── implication patterns ──────────────────────────────────────────────────────

# "X uses/via/implements/relies on Y"
# Subject is the first 1–3 word group; object is the rest (compacted).
# Word boundary \b prevents matching "use" inside "because", "abuse", etc.
_IMPL_VERB = re.compile(
    r'\b(?:uses?\s+|via\s+|implements?\s+|relies?\s+on\s+)',
    re.IGNORECASE,
)

# "use X for Y" or "utilize X for Y" → operator=→, subject=X, object=Y
_USE_FOR = re.compile(
    r'^(?:use\s+|utilize\s+)'
    r'(?P<source>[\w+/.-]+(?:\s+[\w+/.-]+){0,2})'
    r'\s+for\s+'
    r'(?P<target>.+?)$',
    re.IGNORECASE,
)

# "use X" (no "for") → operator=→, subject=X, no object
_USE_PLAIN = re.compile(
    r'^(?:use\s+|utilize\s+)(?P<source>[\w+/.-]+(?:\s+[\w+/.-]+){0,2})',
    re.IGNORECASE,
)

# Subject extractor: leading word-group before an implication verb
_LEADING_WORDS = re.compile(r'^([\w+/.-]+(?:\s+[\w+/.-]+){0,2})\s+')

# ── helpers ───────────────────────────────────────────────────────────────────

_COMPACT_BREAKS = re.compile(r'\s*(?:—|-{2,}|,|;|\.\s)')
_LEADING_ARTICLE = re.compile(r'^\s*(?:a|an|the)\s+', re.IGNORECASE)
# Strip inline markdown: **bold**, *italic*, `code`
_MD_INLINE = re.compile(r'\*{1,3}([^*]+)\*{1,3}|`([^`]+)`')
# Strip unclosed leading asterisks (e.g. "**Foo" where closing ** is absent after a split)
_MD_LEADING_STARS = re.compile(r'^\*+')

_OBJECT_CAP = 35
_SUBJECT_CAP = 25


def _strip_inline_md(text: str) -> str:
    """Remove markdown bold/italic/code markers, keeping the inner text.

    Handles both paired markers (**text**) and unclosed leading markers (**text)
    that occur when a split (e.g. on ' — ') removes the closing marker.
    """
    text = _MD_INLINE.sub(lambda m: (m.group(1) or m.group(2) or "").strip(), text)
    text = _MD_LEADING_STARS.sub("", text)
    return text


def _compact(text: str, cap: int) -> str:
    """Truncate prose to a short identifier-friendly label.

    Strips inline markdown, cuts at the first natural break (em-dash, comma,
    semicolon, sentence end), and strips leading articles.
    """
    text = _strip_inline_md(text.strip())
    text = _LEADING_ARTICLE.sub("", text)
    m = _COMPACT_BREAKS.search(text)
    if m:
        text = text[: m.start()].strip()
    return text[:cap].rstrip() if text else ""


# ── triple extraction by pattern ──────────────────────────────────────────────

def _negation_triple(description: str) -> dict | None:
    """Parse a prohibition.  Returns triple dict or None."""
    # Strip optional filler subject prefix
    s = _FILLER_SUBJ.sub("", description.strip())

    m = _NEG_VERB.match(s)
    if not m:
        return None

    obj_raw = s[m.end():].strip()
    if not obj_raw:
        return None

    obj = _compact(obj_raw, _OBJECT_CAP)
    if not obj:
        return None

    return {"operator": OP_NEG, "subject": "", "object": obj}


def _implication_triple(description: str) -> dict | None:
    """Parse an implication/dependency.  Returns triple dict or None."""
    desc = description.strip()

    # "use X for Y"
    m = _USE_FOR.match(desc)
    if m:
        subj = _compact(m.group("source"), _SUBJECT_CAP)
        obj  = _compact(m.group("target"), _OBJECT_CAP)
        if subj:
            return {"operator": OP_IMPL, "subject": subj, "object": obj}

    # "use X" (plain)
    m = _USE_PLAIN.match(desc)
    if m:
        subj = _compact(m.group("source"), _SUBJECT_CAP)
        if subj:
            return {"operator": OP_IMPL, "subject": subj, "object": ""}

    # "X uses/via/implements/relies on Y"
    parts = _IMPL_VERB.split(desc, maxsplit=1)
    if len(parts) == 2:
        subj_raw, obj_raw = parts
        # The left side must be a short word group (subject)
        sm = _LEADING_WORDS.match(subj_raw + " ")
        if sm:
            subj = _compact(sm.group(1), _SUBJECT_CAP)
            obj  = _compact(obj_raw, _OBJECT_CAP)
            if subj:
                return {"operator": OP_IMPL, "subject": subj, "object": obj}

    return None


def _null_triple(description: str) -> dict | None:
    """Parse a graveyard entry (rejected approach).  Returns triple dict or None."""
    desc = description.strip()

    # Try splitting on " — " or " - " first
    for sep in (" — ", " - "):
        if sep in desc:
            subject_part = desc[: desc.index(sep)].strip()
            subj = _compact(subject_part, _SUBJECT_CAP)
            if subj:
                return {"operator": OP_NULL, "subject": subj, "object": ""}

    # No separator — treat the whole short description as the abandoned approach
    if len(desc) <= 50:
        subj = _compact(desc, _SUBJECT_CAP)
        if subj:
            return {"operator": OP_NULL, "subject": subj, "object": ""}

    return None


# ── public API ────────────────────────────────────────────────────────────────

def extract_triple(description: str, event_type: str) -> dict | None:
    """Extract a CKL V2 triple from an event description.

    Returns a dict with keys ``operator``, ``subject``, ``object``, or None
    if no reliable pattern is found.  The dict is stored in the event payload
    under key ``"triple"`` and consumed at render time.
    """
    if not description:
        return None

    if event_type == "CONSTRAINT_HARD":
        return _negation_triple(description)

    if event_type == "DECISION":
        return _implication_triple(description)

    if event_type in ("APPROACH_ABANDONED_DO_NOT_RETRY", "APPROACH_ABANDONED"):
        return _null_triple(description)

    return None


def augment_with_triple(event: "Event") -> None:
    """In-place: add ``"triple"`` key to event.payload when a triple is found.

    No-op when extraction fails — the event retains V1 rendering.
    Idempotent: safe to call multiple times.
    """
    description = event.payload.get("description", "")
    triple = extract_triple(description, event.event_type)
    if triple is not None:
        event.payload["triple"] = triple
