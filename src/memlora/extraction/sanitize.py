"""Post-extraction description and rationale sanitization.

Cleans raw windowed text before it reaches the DB:
- Strips markdown formatting (tables, fences, headers, bullets)
- Truncates to safe lengths
- Detects and downgrades pure-question descriptions
"""
from __future__ import annotations

import re

_MAX_DESC = 120
_MAX_RATIONALE = 120

# Block-level markdown patterns — applied per line, drop or strip the whole line.
_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$")
_TABLE_SEP  = re.compile(r"^\s*\|[-| :]+\|\s*$")
_CODE_FENCE = re.compile(r"^```.*$")
_HEADING    = re.compile(r"^#{1,6}\s+")
_BULLET     = re.compile(r"^[-*•]\s+")

# Inline markdown patterns — applied per kept line, keep text drop markers.
# Order in `_strip_inline_markdown` matters: links → bold → italic → code → strike.
_INLINE_LINK   = re.compile(r"\[([^\]\n]+)\]\([^)\n]+\)")
_INLINE_BOLD   = re.compile(r"\*\*([^*\n]+?)\*\*|__([^_\n]+?)__")
# Italic only fires when the marker is bordered by non-word chars so identifiers
# like `snake_case_var` and `*args` are preserved. The body must not start or
# end with whitespace (rules out emphasis-mid-word false positives).
_INLINE_STAR_I = re.compile(r"(?<![*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")
_INLINE_UND_I  = re.compile(r"(?<![_\w])_(?!\s)([^_\n]+?)(?<!\s)_(?![_\w])")
_INLINE_CODE   = re.compile(r"`([^`\n]+?)`")
_INLINE_STRIKE = re.compile(r"~~([^~\n]+?)~~")

# Role prefix and BOM stripping
_BOM = "﻿"
_ROLE_PREFIX = re.compile(
    r"^\s*﻿?\s*(?:User|Assistant|Human|Claude)\s*:\s*",
    re.IGNORECASE,
)

# Question heuristic: ends with "?" and has no declarative verb nearby
_QUESTION_END = re.compile(r"\?\s*$")
_DECLARATIVE  = re.compile(
    r"\b(is|are|was|were|will|must|should|cannot|never|always|use|choose|"
    r"decided|chose|switched|set|lock|require|mandate)\b",
    re.IGNORECASE,
)


def sanitize_description(text: str) -> str:
    """Return a clean, length-bounded description string.

    Length cap is applied via memlora.extraction.normalize.smart_truncate so
    the cut lands at a sentence or word boundary rather than mid-word (A-2).
    """
    from memlora.extraction.normalize import smart_truncate

    cleaned = _clean(text)
    return smart_truncate(cleaned, _MAX_DESC).rstrip()


def sanitize_rationale(text: str) -> str:
    """Return a clean, length-bounded rationale string.

    Same sentence/word-aware truncation as descriptions (A-2).
    """
    from memlora.extraction.normalize import smart_truncate

    cleaned = _clean(text)
    return smart_truncate(cleaned, _MAX_RATIONALE).rstrip()


def is_question_description(desc: str) -> bool:
    """Return True if the description looks like a user question, not a statement.

    Used to downgrade extracted events whose description is a question rather
    than a captured decision or constraint.
    """
    stripped = desc.strip()
    if not _QUESTION_END.search(stripped):
        return False
    if _DECLARATIVE.search(stripped):
        return False
    return True


# ── internals ─────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    # Strip BOM and leading role prefix before line-level processing
    text = text.lstrip(_BOM)
    text = _ROLE_PREFIX.sub("", text, count=1)
    lines = text.splitlines()
    kept: list[str] = []
    in_fence = False

    for line in lines:
        if _CODE_FENCE.match(line):
            if not in_fence:
                in_fence = True
                # Keep opening line as a 1-word hint (e.g. "```python")
                lang = line.strip().lstrip("`").strip()
                if lang:
                    kept.append(f"[code: {lang}]")
            else:
                in_fence = False
            continue
        if in_fence:
            continue
        if _TABLE_ROW.match(line) or _TABLE_SEP.match(line):
            continue
        line = _HEADING.sub("", line)
        line = _BULLET.sub("", line)
        line = _strip_inline_markdown(line)
        line = line.strip()
        if line:
            kept.append(line)

    return " ".join(kept)


def _strip_inline_markdown(line: str) -> str:
    """Remove inline markdown markers, keep their text content.

    Inline markdown (bold/italic/code/link/strike) is presentational; once the
    text is destined for re-injection into a future LLM context as plain prose,
    the markers are noise that fragments tokenization without adding meaning.
    """
    # Links first so the URL is dropped before italic/code patterns can match it.
    line = _INLINE_LINK.sub(r"\1", line)
    # Bold must run before italic — otherwise `*chose*` inside `**chose**` would
    # be consumed by the italic pattern and leave one stray `*` on each side.
    line = _INLINE_BOLD.sub(lambda m: m.group(1) or m.group(2), line)
    line = _INLINE_STAR_I.sub(r"\1", line)
    line = _INLINE_UND_I.sub(r"\1", line)
    line = _INLINE_CODE.sub(r"\1", line)
    line = _INLINE_STRIKE.sub(r"\1", line)
    return line
