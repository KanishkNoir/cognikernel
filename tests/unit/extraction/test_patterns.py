"""Tests for memlora.extraction.patterns — Phase A-3 pattern engine.

Locks the precision guards from the v2 plan:
  USE_X       : starts with "use" / "we'll use" / "let's use", role any
  NO_MULTI    : "no X, no Y" parallel, role=user, not in question
  NO_SINGLE   : "no X" at sentence start, role=user, not in question
  PAREN_NOT   : "(not X)" inside sentence, role=user
  ONLY_X      : starts with "only", role=user

Each test asserts a single behavior so failures point at the exact pattern.
"""
from __future__ import annotations

import pytest

from memlora.extraction.patterns import scan_patterns
from memlora.extraction.tokenize import Sentence


def _sent(
    text: str,
    *,
    role: str = "user",
    is_code: bool = False,
    idx: int = 0,
) -> Sentence:
    s = Sentence(
        text=text,
        start_offset=0,
        end_offset=len(text),
        role=role,
        is_code_block=is_code,
    )
    s.sentence_index = idx
    return s


def _run(sentences: list[Sentence]):
    return scan_patterns(sentences, "p1", "s1")


# ── USE_X pattern ────────────────────────────────────────────────────────────


class TestUseXPattern:
    def test_imperative_use_captures_subject(self) -> None:
        events = _run([_sent("Use PostgreSQL.")])
        assert len(events) == 1
        e = events[0]
        assert e.event_type == "DECISION"
        assert e.payload["subject"] == "PostgreSQL"
        assert e.payload["matched_phrase"] == "USE_X"

    def test_wee_will_use_form(self) -> None:
        events = _run([_sent("We'll use argon2id.")])
        assert len(events) == 1
        assert events[0].payload["subject"] == "argon2id"

    def test_lets_use_form(self) -> None:
        events = _run([_sent("Let's use shadcn/ui.")])
        assert len(events) == 1
        assert events[0].payload["subject"] == "shadcn/ui"

    def test_mid_sentence_use_is_rejected_by_shape_guard(self) -> None:
        """'I'll use that later' — not a directive opening."""
        events = _run([_sent("I'll use that later if it makes sense.")])
        assert events == []

    def test_use_inside_explanation_is_rejected(self) -> None:
        events = _run([_sent("The handler chose to use bcrypt earlier.")])
        assert events == []

    def test_user_role_keeps_event_assistant_role_keeps_event(self) -> None:
        """USE_X accepts both roles — classifier later applies the source-role
        penalty for assistant statements."""
        u = _run([_sent("Use PostgreSQL.", role="user")])
        a = _run([_sent("Use PostgreSQL.", role="assistant")])
        assert len(u) == 1
        assert len(a) == 1
        assert u[0].payload["source_role"] == "user"
        assert a[0].payload["source_role"] == "assistant"


# ── NO_MULTI pattern ─────────────────────────────────────────────────────────


class TestNoMultiPattern:
    def test_no_x_no_y_captures_both_subjects(self) -> None:
        """The D7 (shadcn/ui) case: 'no Material UI, no Chakra' → two events."""
        events = _run([_sent("No Material UI, no Chakra in this project.")])
        subjects = [e.payload["subject"] for e in events]
        assert "Material UI" in subjects
        assert "Chakra" in subjects

    def test_no_multi_is_constraint_hard(self) -> None:
        events = _run([_sent("No SQLite, no MySQL.")])
        assert all(e.event_type == "CONSTRAINT_HARD" for e in events)

    def test_no_multi_has_high_confidence(self) -> None:
        events = _run([_sent("No SQLite, no MySQL.")])
        assert all(e.payload["confidence"] >= 0.85 for e in events)

    def test_no_multi_in_question_is_rejected(self) -> None:
        events = _run([_sent("No SQLite, no MySQL?")])
        # The shape guard rejects question-form sentences.
        assert events == []

    def test_no_multi_assistant_role_is_rejected(self) -> None:
        """Pattern NO_MULTI is user-only — assistant rejections are softer
        and go through other paths."""
        events = _run([_sent("No SQLite, no MySQL.", role="assistant")])
        assert events == []


# ── NO_SINGLE pattern ────────────────────────────────────────────────────────


class TestNoSinglePattern:
    def test_sentence_starting_no_x(self) -> None:
        events = _run([_sent("No Celery.")])
        # NO_SINGLE captures 'Celery'.
        assert any(
            e.payload["matched_phrase"] == "NO_SINGLE" and e.payload["subject"] == "Celery"
            for e in events
        )

    def test_mid_sentence_no_not_matched(self) -> None:
        """'There's no time' should not produce a constraint."""
        events = _run([_sent("There's no time for that today.")])
        assert all(e.payload["matched_phrase"] != "NO_SINGLE" for e in events)

    def test_question_form_rejected(self) -> None:
        events = _run([_sent("No tests?")])
        assert all(e.payload["matched_phrase"] != "NO_SINGLE" for e in events)


# ── PAREN_NOT pattern ────────────────────────────────────────────────────────


class TestParenNotPattern:
    def test_parenthetical_negation_captures_subject(self) -> None:
        """D1 stack-proposal case: 'production-ready SQL (not SQLite)'."""
        events = _run([_sent("Production-ready SQL (not SQLite) please.")])
        assert any(
            e.payload["matched_phrase"] == "PAREN_NOT" and e.payload["subject"] == "SQLite"
            for e in events
        )

    def test_paren_not_is_constraint_hard(self) -> None:
        events = _run([_sent("Use Postgres (not MySQL).")])
        paren_events = [e for e in events if e.payload["matched_phrase"] == "PAREN_NOT"]
        assert paren_events
        assert all(e.event_type == "CONSTRAINT_HARD" for e in paren_events)

    def test_no_paren_no_match(self) -> None:
        events = _run([_sent("Use PostgreSQL, not MySQL.")])
        assert all(e.payload["matched_phrase"] != "PAREN_NOT" for e in events)


# ── ONLY_X pattern ───────────────────────────────────────────────────────────


class TestOnlyXPattern:
    def test_only_at_start_captures(self) -> None:
        events = _run([_sent("Only async SQLAlchemy.")])
        only_events = [e for e in events if e.payload["matched_phrase"] == "ONLY_X"]
        assert len(only_events) == 1
        assert only_events[0].payload["subject"] == "async SQLAlchemy"

    def test_only_mid_sentence_rejected(self) -> None:
        events = _run([_sent("I want only the best.")])
        assert all(e.payload["matched_phrase"] != "ONLY_X" for e in events)


# ── code blocks + role filtering ─────────────────────────────────────────────


class TestSentenceFilters:
    def test_code_block_sentences_skipped(self) -> None:
        events = _run([_sent("Use PostgreSQL.", is_code=True)])
        assert events == []

    def test_empty_sentence_skipped(self) -> None:
        events = _run([_sent("   ")])
        assert events == []

    def test_multiple_sentences_each_scanned(self) -> None:
        events = _run([
            _sent("Use PostgreSQL.", idx=0),
            _sent("Only async SQLAlchemy.", idx=1),
        ])
        # Two patterns should fire across the two sentences.
        assert len(events) == 2
        assert {e.payload["matched_phrase"] for e in events} == {"USE_X", "ONLY_X"}


# ── stopword subjects rejected ───────────────────────────────────────────────


class TestStopwordSubjects:
    def test_use_the_rejected(self) -> None:
        """'Use the system' captures 'the system' as subject — but 'the' alone
        as a single-word subject should be filtered."""
        events = _run([_sent("Use the.")])
        # 'the' is in the stopword list.
        assert all(e.payload["subject"].lower() != "the" for e in events)


# ── integration: pattern events round-trip through pipeline ──────────────────


class TestPipelineIntegration:
    def test_patterns_appear_in_extract_session_output(self) -> None:
        """Confirms scan_patterns is wired into extract_session."""
        from memlora.extraction.pipeline import SessionMetadata, extract_session

        transcript = (
            "User:\nNo Material UI, no Chakra. Use shadcn/ui instead.\n\n"
            "Assistant:\nUnderstood.\n"
        )
        meta = SessionMetadata(
            project_id="p1", session_id="s1",
            started_at=0, ended_at=0,
        )
        events = extract_session(transcript, meta)

        # Either trie or pattern path should yield the rejection signals.
        subjects = [e.payload.get("subject", "") for e in events]
        # Multi-negation case yields both:
        assert any("Material UI" in s for s in subjects)
        assert any("Chakra" in s for s in subjects)
