"""Post-classification description normalization — Phase A-1 / A-2.

Two concerns live here, intentionally co-located:

  A-1 normalize_description()
      Strips prompt-verb prefixes that leak from the user's instruction into
      the captured description. The Arm-C diagnostic showed bullets like
      "Confirm the env var name and that it must never appear anywhere else"
      where "Confirm " is the user telling Claude what to do, not part of the
      constraint itself.

  A-2 smart_truncate()
      Replaces character-based truncation in sanitize.py. Cuts at sentence
      boundaries when possible, falls back to word boundaries, and appends
      an ellipsis when the cut is mid-thought. Never cuts mid-word.

Both helpers are pure — no DB, no filesystem.
"""
from __future__ import annotations

import re

# ── A-1: prompt-verb prefix stripping ────────────────────────────────────────


# Order matters: longer prefixes first so "Open a work thread:" wins over
# "Open " when both could match. Each entry MUST end in a space (or punctuation
# + space) so we don't accidentally cut into the surviving description.
_PROMPT_VERB_PREFIXES: tuple[str, ...] = (
    "Open a work thread: ",
    "Open a thread: ",
    "Note that ",
    "Make sure ",
    "Make sure that ",
    "Please make sure ",
    "Please confirm ",
    "Please remember ",
    "Please note ",
    "Confirm ",
    "Establish ",
    "Record ",
    "Lock in ",
    "Decide ",
    "Reminder: ",
    "Remember: ",
    "FYI: ",
)

# Multiple whitespace characters collapse to a single space.
_WHITESPACE = re.compile(r"\s+")


def normalize_description(text: str) -> str:
    """Strip prompt-verb prefixes, collapse whitespace, ensure terminal punctuation.

    Idempotent: re-applying produces the same string.
    Returns '' for falsy / pure-whitespace input.
    """
    if not text:
        return ""

    s = text.strip()
    if not s:
        return ""

    # Strip ONE prefix only — if the original had "Confirm Please remember X",
    # collapsing both is over-aggressive. The first match wins.
    for prefix in _PROMPT_VERB_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):].lstrip()
            break
        # Case-insensitive variant for sentence-start tolerance.
        lower_prefix = prefix.lower()
        if s.lower().startswith(lower_prefix):
            s = s[len(prefix):].lstrip()
            break

    # After stripping, the first character may now be lowercase. Capitalize it
    # so the sanitized form reads like a sentence.
    if s and s[0].islower():
        s = s[0].upper() + s[1:]

    s = _WHITESPACE.sub(" ", s).strip()
    if not s:
        return ""

    # Ensure terminal punctuation so injection bullets render consistently.
    if s[-1] not in ".!?":
        s = s + "."

    return s


# ── A-2: sentence-aware truncation ───────────────────────────────────────────


# Threshold: a sentence terminator counts as a "good" truncation point only if
# it falls past this fraction of the budget. Below the threshold we'd lose too
# much content and the bullet becomes meaningless.
_TERMINATOR_MIN_FRACTION = 0.5

_ELLIPSIS = "…"

# Characters that close a sentence.
_TERMINATORS = (".", "!", "?")


def smart_truncate(text: str, max_chars: int, *, ellipsis: str = _ELLIPSIS) -> str:
    """Truncate `text` to fit within `max_chars` without breaking words.

    Strategy (best to worst):
      1. Already fits           → return verbatim
      2. Sentence terminator in the back half of the budget → cut after it
      3. Word boundary in budget → cut there, append ellipsis
      4. Hard cut at budget    → append ellipsis (only when the text has no
                                  whitespace, e.g., a long unbreakable token)

    The ellipsis is always one user-perceived character (the `…` codepoint) so
    callers can budget against `max_chars` without separately accounting for it.
    `max_chars` < 4 is a degenerate case — we return text[:max_chars] with no
    ellipsis since the budget can't fit a meaningful truncation marker.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars < 4:
        return text[:max_chars]

    # Look for a sentence terminator in the slice that would fit.
    window = text[:max_chars]
    terminator_pos = max(window.rfind(t) for t in _TERMINATORS)
    min_acceptable_pos = int(max_chars * _TERMINATOR_MIN_FRACTION)
    if terminator_pos >= min_acceptable_pos:
        # Cut right after the terminator (inclusive) so the result reads as a
        # complete sentence. No ellipsis — the punctuation signals closure.
        return text[: terminator_pos + 1]

    # Word boundary cut. Reserve room for the ellipsis.
    budget_for_text = max_chars - len(ellipsis)
    word_cutoff = text[:budget_for_text + 1].rfind(" ")
    if word_cutoff > 0:
        return text[:word_cutoff].rstrip() + ellipsis

    # Hard cut — no whitespace in the budget window. Append ellipsis anyway so
    # the reader knows it's incomplete.
    return text[:budget_for_text] + ellipsis
