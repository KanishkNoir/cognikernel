"""Post-extraction description and rationale sanitization.

Cleans raw windowed text before it reaches the DB:
- Strips markdown formatting (tables, fences, headers, bullets)
- Truncates to safe lengths
- Detects and downgrades pure-question descriptions
"""
from __future__ import annotations

import re

# v1 A-2: descriptions are facts and are NEVER truncated — truncation discards
# the operative tail (a number / env var / model id). _HARD_DESC is only a
# blob-guard (mis-segmented code dump), and even then keep_whole_fact cuts solely
# at a complete sentence boundary, never mid-sentence. Rationale is context, not
# the fact, so it stays ellipsis-truncatable at a tight budget.
_HARD_DESC = 600
_MAX_RATIONALE = 120

# Block-level markdown patterns — applied per line, drop or strip the whole line.
_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$")
_TABLE_SEP  = re.compile(r"^\s*\|[-| :]+\|\s*$")

# Windowed table debris: extraction windows straddle a markdown table boundary
# and collapse header + separator + id cells onto the front of a content cell,
# all on ONE line ending in prose — so _TABLE_ROW (which needs a trailing `|`)
# never fires and the scaffolding leaks into the stored description
# ("| # | Invariant | |---|---| | O1 | Every status transition logs …").
# Signature: a separator run `|---|` OR `| |` empty-cell joints embedded mid-line.
_TABLE_SEP_FRAGMENT = re.compile(r"\|\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?")
_TABLE_DEBRIS_SIGNAL = re.compile(r"\|\s*:?-{2,}|\|\s*\|")
# A scaffolding cell: empty, a bare id/enumerator (#, C1, S12, O3, D5), or a
# short header word with no sentence content. NOT applied to substantive cells.
_SCAFFOLD_CELL = re.compile(
    r"^(?:#|[A-Z]{1,3}\d{1,3}|invariant|why|decision|rule|status|id|option|"
    r"consequence|notes?|value|key|field|column|type)$",
    re.IGNORECASE,
)
_CONTENT_WORD = re.compile(r"[A-Za-z0-9]{3,}")
_CODE_FENCE = re.compile(r"^```.*$")
_HEADING    = re.compile(r"^#{1,6}\s+")
_BULLET     = re.compile(r"^[-*•]\s+")
# J5.1: blockquote markers were previously untouched and leaked into stored
# descriptions ("> …impossible.").
_BLOCKQUOTE = re.compile(r"^\s*(?:>\s?)+")

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

    Facts are never truncated (v1 A-2) — keep_whole_fact only guards against a
    mis-segmented blob, and even then cuts at a complete sentence boundary, never
    mid-sentence, never with an ellipsis.
    """
    from memlora.extraction.normalize import keep_whole_fact

    cleaned = _clean(text)
    return keep_whole_fact(cleaned, _HARD_DESC).rstrip()


def sanitize_rationale(text: str) -> str:
    """Return a clean, length-bounded rationale string.

    Same sentence/word-aware truncation as descriptions (A-2).
    """
    from memlora.extraction.normalize import smart_truncate

    cleaned = _clean(text)
    return smart_truncate(cleaned, _MAX_RATIONALE).rstrip()


# J5.2 — context-dependent fragments. A sentence that only means something
# relative to its surrounding conversation ("The 2× multiplier only matters if
# _MAX_ATTEMPTS were raised above 2") is not a standalone fact; minted at full
# authority it pollutes the mandatory hard-constraints zone. The patterns
# require the anaphoric/conditional OPENER shape so genuine constraints with
# "only" semantics ("Only cache when temperature is 0", "Backoff applies only
# to 5xx") never match — the boundary is documented in the test table.
_FRAG_ANAPHORIC = re.compile(
    r"^(?:The|This|That|It|These|Those)\s+\S+(?:\s+\S+)?\s+"
    r"(?:only|just)\s+(?:matters|applies|works|exists|happens|fires|holds)\b",
    re.IGNORECASE,
)
_FRAG_CONDITIONAL = re.compile(
    r"\bonly\s+(?:matters|applies|fires|holds|happens|kicks\s+in)\s+(?:if|when)\b",
    re.IGNORECASE,
)
_FRAG_COUNTERFACTUAL = re.compile(
    r"\bwould\s+only\b|\bif\s+\S+(?:\s+\S+){0,4}\s+were\s+\w+",
    re.IGNORECASE,
)


def is_context_dependent_fragment(desc: str) -> bool:
    """True when a description is an aside that depends on unstated context.

    Used by the pipeline to demote weight and retype CONSTRAINT_HARD →
    CONSTRAINT_SOFT — a context-dependent sentence must never be budget-exempt
    mandatory. Mint-time only; raw evidence is untouched (lossless).
    """
    stripped = desc.strip()
    return bool(
        _FRAG_ANAPHORIC.search(stripped)
        or _FRAG_CONDITIONAL.search(stripped)
        or _FRAG_COUNTERFACTUAL.search(stripped)
    )


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
        line = _strip_table_scaffolding(line)
        line = _BLOCKQUOTE.sub("", line)
        line = _HEADING.sub("", line)
        line = _BULLET.sub("", line)
        line = _strip_inline_markdown(line)
        line = line.strip()
        if line:
            kept.append(line)

    return " ".join(kept)


def _strip_table_scaffolding(line: str) -> str:
    """Strip collapsed markdown-table debris from a mixed table+prose line.

    Only fires when the line carries an unambiguous table signal (a `|---|`
    separator run or `| |` empty-cell joint) — plain prose that merely contains
    a pipe (`str | None`, `A | B union`) has neither and is returned untouched.
    When it fires: drop separator runs, split on `|`, discard scaffolding cells
    (empty, bare ids like `S2`/`#`, short header words), and keep the
    substantive cells (>= 4 content words, i.e. real facts), rejoined as prose.
    """
    if "|" not in line or not _TABLE_DEBRIS_SIGNAL.search(line):
        return line
    without_sep = _TABLE_SEP_FRAGMENT.sub("|", line)
    cells = [c.strip() for c in without_sep.split("|")]
    kept = [
        c for c in cells
        if c and not _SCAFFOLD_CELL.match(c)
        and len(_CONTENT_WORD.findall(c)) >= _MIN_CONTENT_WORDS_TABLE
    ]
    if not kept:
        # Nothing substantive survived (a pure scaffolding line the block guard
        # missed) — drop to empty so the caller skips it.
        return ""
    joined = ". ".join(kept)
    # Avoid doubled terminators when a cell already ended in punctuation.
    return re.sub(r"([.!?])\.\s", r"\1 ", joined)


# 4 content words = a real fact cell, matching the pipeline's _MIN_CONTENT_WORDS
# floor; short id/header cells fall below it.
_MIN_CONTENT_WORDS_TABLE = 4


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
    # J5.1 residue pass: extraction windows can split a bold span so only one
    # side of the `**` pair lands in this line; the paired patterns above can't
    # see it and stored descriptions ended with artifacts like `impossible.**.`.
    # Trade-off: a literal `**kwargs` outside backticks loses its stars too —
    # acceptable, since the paired passes already mangle that case today.
    line = re.sub(r"\*{2,}", "", line)
    return line
