"""Tests for the A-4 assistant-answer co-capture mechanism.

Covers:
  - User trie matches trigger co-capture from the next assistant turn
  - Co-capture events carry authority=assistant_answer_to_user_question
  - Multiple user matches before the same assistant turn dedupe
  - Assistant code blocks are skipped
  - Assistant trie matches do NOT trigger co-capture
"""
from __future__ import annotations

import pytest

from cognikernel.extraction.authority import ASSISTANT_ANSWER_TO_QUESTION
from cognikernel.extraction.tokenize import Sentence
from cognikernel.extraction.trie import TrieMatch
from cognikernel.extraction.windowing import extract_co_captures


def _sent(text: str, *, role: str, idx: int, is_code: bool = False) -> Sentence:
    s = Sentence(
        text=text, start_offset=idx * 100, end_offset=idx * 100 + len(text),
        role=role, is_code_block=is_code,
    )
    s.sentence_index = idx
    return s


def _match(idx: int, phrase: str = "must never", st: str = "CONSTRAINT_HARD") -> TrieMatch:
    return TrieMatch(sentence_index=idx, matched_phrase=phrase, signal_type=st, confidence=1.0)


# ── basic co-capture flow ────────────────────────────────────────────────────


class TestBasicCoCapture:
    def test_user_match_triggers_co_capture(self) -> None:
        sentences = [
            _sent("Confirm the env var must never appear elsewhere.", role="user", idx=0),
            _sent("JWT_SECRET_KEY lives in the env file only.", role="assistant", idx=1),
        ]
        events = extract_co_captures(sentences, [_match(0)], "p1", "s1")
        assert len(events) == 1
        e = events[0]
        assert e.payload["authority"] == ASSISTANT_ANSWER_TO_QUESTION
        assert "JWT_SECRET_KEY" in e.payload["description"]

    def test_assistant_match_does_not_trigger(self) -> None:
        sentences = [
            _sent("We must never expose secrets.", role="assistant", idx=0),
            _sent("Right.", role="user", idx=1),
        ]
        events = extract_co_captures(sentences, [_match(0)], "p1", "s1")
        assert events == []

    def test_dedup_when_multiple_user_matches_share_assistant_turn(self) -> None:
        """Two user trie hits before one assistant reply should produce ONE
        co-capture event, not two duplicates."""
        sentences = [
            _sent("First, must never X.", role="user", idx=0),
            _sent("Second, must never Y.", role="user", idx=1),
            _sent("Acknowledged: X and Y are forbidden.", role="assistant", idx=2),
        ]
        events = extract_co_captures(sentences, [_match(0), _match(1)], "p1", "s1")
        assert len(events) == 1


# ── filters & edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_no_assistant_reply_produces_nothing(self) -> None:
        sentences = [
            _sent("Must never use SQLite.", role="user", idx=0),
            _sent("More user talk.", role="user", idx=1),
        ]
        events = extract_co_captures(sentences, [_match(0)], "p1", "s1")
        assert events == []

    def test_assistant_code_block_skipped(self) -> None:
        """Code blocks aren't part of the conversational answer."""
        sentences = [
            _sent("Must never use SQLite.", role="user", idx=0),
            _sent("```python\nx = 1\n```", role="assistant", idx=1, is_code=True),
            _sent("Use PostgreSQL.", role="assistant", idx=2),
        ]
        events = extract_co_captures(sentences, [_match(0)], "p1", "s1")
        assert len(events) == 1
        assert "PostgreSQL" in events[0].payload["description"]

    def test_captures_up_to_two_sentences(self) -> None:
        """Limit: at most 2 consecutive assistant sentences."""
        sentences = [
            _sent("Must never X.", role="user", idx=0),
            _sent("One.", role="assistant", idx=1),
            _sent("Two.", role="assistant", idx=2),
            _sent("Three.", role="assistant", idx=3),
        ]
        events = extract_co_captures(sentences, [_match(0)], "p1", "s1")
        # First two sentences captured, third dropped.
        desc = events[0].payload["description"]
        assert "One" in desc
        assert "Two" in desc
        assert "Three" not in desc

    def test_empty_sentences_returns_empty(self) -> None:
        assert extract_co_captures([], [], "p1", "s1") == []

    def test_match_at_end_of_sentence_list(self) -> None:
        """User match with no following sentences — no co-capture."""
        sentences = [
            _sent("Must never X.", role="user", idx=0),
        ]
        events = extract_co_captures(sentences, [_match(0)], "p1", "s1")
        assert events == []


# ── full pipeline integration ────────────────────────────────────────────────


class TestPipelineIntegration:
    def test_co_capture_event_appears_in_extract_session(self) -> None:
        from cognikernel.extraction.pipeline import SessionMetadata, extract_session

        transcript = (
            "User:\nConfirm the env var name and that it must never appear anywhere else.\n\n"
            "Assistant:\nThe env var lives in JWT_SECRET_KEY only.\n"
        )
        meta = SessionMetadata(
            project_id="p1", session_id="s1",
            started_at=0, ended_at=0,
        )
        events = extract_session(transcript, meta)

        authorities = [e.payload.get("authority", "") for e in events]
        assert ASSISTANT_ANSWER_TO_QUESTION in authorities

        co = next(
            e for e in events
            if e.payload.get("authority") == ASSISTANT_ANSWER_TO_QUESTION
        )
        assert "JWT_SECRET_KEY" in co.payload["description"]
