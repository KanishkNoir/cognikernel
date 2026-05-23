"""Sentence splitter for session transcripts.

Custom implementation (not spaCy/NLTK) tuned for developer chat transcripts:
  - Fenced code blocks are atomic sentences, never split internally.
  - Markdown bullets are individual sentences.
  - Role markers (Human: / Assistant:) delineate speaker blocks.
  - Sentence terminators (.!?) only split when followed by space+capital.
  - Common abbreviations (e.g., i.e., vs.) are not treated as sentence ends.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_FENCED_CODE = re.compile(r"```[\s\S]*?```", re.DOTALL)
_ROLE_HEADER = re.compile(
    r"^(Human|User|Assistant|Claude)\s*:\s*", re.MULTILINE | re.IGNORECASE
)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+\S")
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_ABBREVS = frozenset(
    {
        "e.g.", "i.e.", "vs.", "etc.", "mr.", "mrs.", "ms.", "dr.",
        "prof.", "sr.", "jr.", "inc.", "ltd.", "co.", "fig.", "approx.",
        "dept.", "est.", "avg.", "max.", "min.", "no.", "vol.",
    }
)


@dataclass
class Sentence:
    text: str
    start_offset: int
    end_offset: int
    role: str              # "user" | "assistant"
    is_code_block: bool
    sentence_index: int = field(default=0, compare=False)


def tokenize(transcript: str) -> list[Sentence]:
    """Split a session transcript into addressable Sentence units."""
    sentences: list[Sentence] = []

    # Locate all fenced code blocks once; they are atomic and never split.
    code_spans = [
        (m.start(), m.end(), m.group()) for m in _FENCED_CODE.finditer(transcript)
    ]

    # Split transcript into per-speaker regions.
    role_markers = list(_ROLE_HEADER.finditer(transcript))
    regions = _build_role_regions(transcript, role_markers)

    for role, r_start, r_end in regions:
        _process_region(transcript, role, r_start, r_end, code_spans, sentences)

    for idx, s in enumerate(sentences):
        s.sentence_index = idx

    return sentences


# ── region helpers ────────────────────────────────────────────────────────────

def _build_role_regions(
    transcript: str, markers: list
) -> list[tuple[str, int, int]]:
    if not markers:
        return [("user", 0, len(transcript))]

    regions: list[tuple[str, int, int]] = []

    if markers[0].start() > 0:
        regions.append(("user", 0, markers[0].start()))

    for i, m in enumerate(markers):
        label = m.group(1).lower()
        role = "assistant" if label in ("assistant", "claude") else "user"
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(transcript)
        regions.append((role, start, end))

    return regions


def _process_region(
    transcript: str,
    role: str,
    r_start: int,
    r_end: int,
    code_spans: list[tuple[int, int, str]],
    sentences: list[Sentence],
) -> None:
    # Code blocks that fall entirely within this region.
    region_code = sorted(
        (s, e, t) for s, e, t in code_spans if r_start <= s and e <= r_end
    )

    prose_start = r_start
    for code_start, code_end, code_text in region_code:
        if code_start > prose_start:
            _split_prose(transcript[prose_start:code_start], prose_start, role, sentences)
        sentences.append(
            Sentence(
                text=code_text.strip(),
                start_offset=code_start,
                end_offset=code_end,
                role=role,
                is_code_block=True,
            )
        )
        prose_start = code_end

    if prose_start < r_end:
        _split_prose(transcript[prose_start:r_end], prose_start, role, sentences)


# ── prose splitting ───────────────────────────────────────────────────────────

def _split_prose(
    text: str, base_offset: int, role: str, sentences: list[Sentence]
) -> None:
    accumulated: list[str] = []
    acc_start = base_offset
    offset = base_offset

    for line in text.split("\n"):
        line_bytes = len(line.encode("utf-8", errors="replace"))
        line_end = offset + line_bytes
        stripped = line.strip()

        if not stripped:
            if accumulated:
                _flush_prose(accumulated, acc_start, offset, role, sentences)
                accumulated = []
            offset = line_end + 1
            acc_start = offset
            continue

        if _BULLET.match(line):
            if accumulated:
                _flush_prose(accumulated, acc_start, offset, role, sentences)
                accumulated = []
                acc_start = offset
            sentences.append(
                Sentence(
                    text=stripped,
                    start_offset=offset,
                    end_offset=line_end,
                    role=role,
                    is_code_block=False,
                )
            )
            acc_start = line_end + 1
            offset = line_end + 1
            continue

        accumulated.append(stripped)
        offset = line_end + 1

    if accumulated:
        _flush_prose(accumulated, acc_start, offset, role, sentences)


def _flush_prose(
    lines: list[str], start: int, end: int, role: str, sentences: list[Sentence]
) -> None:
    text = " ".join(lines)
    parts = _split_on_terminators(text)
    approx_offset = start
    for part in parts:
        part = part.strip()
        if not part:
            continue
        part_end = min(approx_offset + len(part.encode("utf-8", errors="replace")), end)
        sentences.append(
            Sentence(
                text=part,
                start_offset=approx_offset,
                end_offset=part_end,
                role=role,
                is_code_block=False,
            )
        )
        approx_offset = part_end + 1


def _split_on_terminators(text: str) -> list[str]:
    """Split on .!? followed by space+capital, skipping known abbreviations."""
    split_points = list(_SENT_BOUNDARY.finditer(text))
    if not split_points:
        return [text]

    parts: list[str] = []
    prev = 0
    for m in split_points:
        # Check if the word just before the split is an abbreviation.
        prefix = text[prev : m.start()].rstrip()
        last_word = prefix.split()[-1].lower() if prefix.split() else ""
        if last_word in _ABBREVS or last_word + "." in _ABBREVS:
            continue
        parts.append(text[prev : m.start() + 1])  # include the terminator
        prev = m.end()

    parts.append(text[prev:])
    return [p for p in parts if p.strip()]
