"""Aho-Corasick trie scanner for multi-pattern signal extraction.

The automaton is built once at startup from signals.py and reused across
all extraction calls. Construction is ~milliseconds; traversal is ~microseconds
per sentence.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

import ahocorasick

from memlora.extraction.signals import SIGNAL_DICTIONARY
from memlora.extraction.tokenize import Sentence


@dataclass(frozen=True)
class TrieMatch:
    sentence_index: int
    matched_phrase: str
    signal_type: str
    confidence: float


class TrieScanner:
    """Multi-pattern scanner built from SIGNAL_DICTIONARY."""

    def __init__(self) -> None:
        self._automaton: ahocorasick.Automaton = self._build()

    @staticmethod
    def _build() -> ahocorasick.Automaton:
        automaton = ahocorasick.Automaton()
        for phrase, (signal_type, confidence) in SIGNAL_DICTIONARY.items():
            automaton.add_word(phrase.lower(), (phrase, signal_type, confidence))
        automaton.make_automaton()
        return automaton

    def scan(self, sentences: list[Sentence], transcript: str) -> list[TrieMatch]:
        """Scan the full transcript and return matches mapped to sentence indices.

        Code-block sentences are skipped — signals inside code are noise.
        Case-insensitive; word boundaries enforced via post-filter.
        """
        if not sentences:
            return []

        lower = transcript.lower()
        # Build sorted list of sentence start offsets for bisect lookup.
        starts = [s.start_offset for s in sentences]

        matches: list[TrieMatch] = []

        for end_idx, (phrase, signal_type, confidence) in self._automaton.iter(lower):
            start_idx = end_idx - len(phrase) + 1
            end_excl = end_idx + 1

            if not _word_boundary(lower, start_idx, end_excl):
                continue

            sent_idx = _sentence_for_offset(sentences, starts, start_idx)
            if sent_idx is None:
                continue

            if sentences[sent_idx].is_code_block:
                continue

            matches.append(
                TrieMatch(
                    sentence_index=sent_idx,
                    matched_phrase=phrase,
                    signal_type=signal_type,
                    confidence=confidence,
                )
            )

        return matches


# ── module-level singleton ────────────────────────────────────────────────────

_scanner: TrieScanner | None = None


def get_scanner() -> TrieScanner:
    """Return the shared TrieScanner, building it on first call."""
    global _scanner
    if _scanner is None:
        _scanner = TrieScanner()
    return _scanner


# ── helpers ───────────────────────────────────────────────────────────────────

def _word_boundary(text: str, start: int, end: int) -> bool:
    before_ok = start == 0 or not (text[start - 1].isalnum() or text[start - 1] == "_")
    after_ok = end >= len(text) or not (text[end].isalnum() or text[end] == "_")
    return before_ok and after_ok


def _sentence_for_offset(
    sentences: list[Sentence], starts: list[int], offset: int
) -> int | None:
    idx = bisect.bisect_right(starts, offset) - 1
    if idx < 0:
        return None
    s = sentences[idx]
    if s.start_offset <= offset <= s.end_offset:
        return idx
    return None
