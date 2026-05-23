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

# Markdown patterns to strip
_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$")
_TABLE_SEP  = re.compile(r"^\s*\|[-| :]+\|\s*$")
_CODE_FENCE = re.compile(r"^```.*$")
_HEADING    = re.compile(r"^#{1,6}\s+")
_BULLET     = re.compile(r"^[-*•]\s+")

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
    """Return a clean, length-bounded description string."""
    cleaned = _clean(text)
    return cleaned[:_MAX_DESC].rstrip()


def sanitize_rationale(text: str) -> str:
    """Return a clean, length-bounded rationale string."""
    cleaned = _clean(text)
    return cleaned[:_MAX_RATIONALE].rstrip()


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
        line = line.strip()
        if line:
            kept.append(line)

    return " ".join(kept)
