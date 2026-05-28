"""Tests for the A-4 Pending Confirmation section + suppression rule."""
from __future__ import annotations

import pytest

from memlora.extraction.authority import (
    ASSISTANT_ANSWER_TO_QUESTION,
    ASSISTANT_DECIDED,
    USER_STATED,
)
from memlora.injection.ordering import partition_events
from memlora.injection.template import (
    InjectionContext,
    _render_pending_confirmation,
    render_injection,
)
from memlora.storage.events import Event


# ── helpers ──────────────────────────────────────────────────────────────────


def _event(
    event_type: str,
    description: str,
    *,
    authority: str,
    subject: str = "",
    session_id: str = "sess1",
    content_hash: str | None = None,
) -> Event:
    return Event(
        project_id="p1",
        session_id=session_id,
        event_type=event_type,
        payload={
            "description": description,
            "rationale": "",
            "subject": subject,
            "authority": authority,
        },
        content_hash=content_hash or description[:32].ljust(64, "0"),
        weight=0.5,
    )


def _ctx(**overrides) -> InjectionContext:
    defaults = dict(
        project_name="proj", session_number=1, total_sessions=1, state_version=1,
        hard_constraints=[], graveyard=[], components=[], decisions=[],
        active_threads=[], summary_text="", token_budget=2000,
    )
    defaults.update(overrides)
    return InjectionContext(**defaults)


# ── _render_pending_confirmation ─────────────────────────────────────────────


class TestRenderPendingConfirmation:
    def test_empty_returns_empty(self) -> None:
        assert _render_pending_confirmation([]) == ""

    def test_renders_header(self) -> None:
        e = _event("CONSTRAINT_SOFT", "JWT lives in JWT_SECRET_KEY.",
                   authority=ASSISTANT_ANSWER_TO_QUESTION)
        out = _render_pending_confirmation([e])
        assert "### Pending confirmation" in out
        assert "not yet user-confirmed" in out

    def test_renders_description_with_session_marker(self) -> None:
        e = _event("CONSTRAINT_SOFT", "JWT lives in JWT_SECRET_KEY.",
                   authority=ASSISTANT_ANSWER_TO_QUESTION,
                   session_id="abc1234567890def")
        out = _render_pending_confirmation([e])
        assert "JWT lives in JWT_SECRET_KEY" in out
        assert "assistant" in out
        # Session id truncated to 12 chars for consistency with B-2 header.
        assert "abc123456789" in out

    def test_section_appears_in_full_injection(self) -> None:
        e = _event("CONSTRAINT_SOFT", "JWT lives in JWT_SECRET_KEY.",
                   authority=ASSISTANT_ANSWER_TO_QUESTION)
        ctx = _ctx(pending_confirmations=[e])
        block = render_injection(ctx)
        assert "### Pending confirmation" in block


# ── partition_events suppression rule ────────────────────────────────────────


class TestSuppressionByMatchingSubject:
    def test_co_capture_suppressed_when_user_states_same_subject(self) -> None:
        """User explicitly stated the JWT secret rule; co-capture with same
        normalized subject must be suppressed."""
        co_capture = _event(
            "CONSTRAINT_SOFT", "JWT secret env var is JWT_SECRET_KEY.",
            authority=ASSISTANT_ANSWER_TO_QUESTION, subject="JWT_SECRET_KEY",
        )
        user_assert = _event(
            "CONSTRAINT_HARD", "Use JWT_SECRET_KEY env var.",
            authority=USER_STATED, subject="JWT_SECRET_KEY",
        )

        buckets = partition_events([co_capture, user_assert])
        assert buckets["pending_confirmations"] == []
        assert len(buckets["hard_constraints"]) == 1

    def test_co_capture_kept_when_no_matching_confirming_event(self) -> None:
        co_capture = _event(
            "CONSTRAINT_SOFT", "Maybe use Argon2.",
            authority=ASSISTANT_ANSWER_TO_QUESTION, subject="Argon2",
        )

        buckets = partition_events([co_capture])
        assert len(buckets["pending_confirmations"]) == 1
        assert buckets["hard_constraints"] == []

    def test_suppression_is_case_and_article_insensitive(self) -> None:
        co_capture = _event(
            "CONSTRAINT_SOFT", "Use the postgresql database.",
            authority=ASSISTANT_ANSWER_TO_QUESTION, subject="the PostgreSQL",
        )
        user_assert = _event(
            "CONSTRAINT_HARD", "Use PostgreSQL.",
            authority=USER_STATED, subject="postgresql",
        )

        buckets = partition_events([co_capture, user_assert])
        # Normalized subjects match → suppressed.
        assert buckets["pending_confirmations"] == []

    def test_assistant_decided_also_suppresses(self) -> None:
        """An assistant_decided event with the same subject is also a
        'confirming' authority — it suppresses co-capture."""
        co_capture = _event(
            "CONSTRAINT_SOFT", "Use Argon2.",
            authority=ASSISTANT_ANSWER_TO_QUESTION, subject="Argon2",
        )
        assistant_dec = _event(
            "DECISION", "We'll use Argon2.",
            authority=ASSISTANT_DECIDED, subject="Argon2",
        )

        buckets = partition_events([co_capture, assistant_dec])
        assert buckets["pending_confirmations"] == []
        assert len(buckets["decisions"]) == 1

    def test_co_capture_with_empty_subject_kept(self) -> None:
        """Without a subject, suppression can't apply — the event survives."""
        co_capture = _event(
            "CONSTRAINT_SOFT", "Something assistant said.",
            authority=ASSISTANT_ANSWER_TO_QUESTION, subject="",
        )
        buckets = partition_events([co_capture])
        assert len(buckets["pending_confirmations"]) == 1


# ── routing: pending_confirmations supersedes type-based routing ─────────────


class TestRoutingPrecedence:
    def test_constraint_hard_with_co_capture_authority_goes_to_pending(self) -> None:
        """A CONSTRAINT_HARD whose authority is the co-capture marker MUST
        land in Pending Confirmation, not in Hard Constraints — the whole
        point of A-4 is to not auto-promote assistant claims."""
        e = _event(
            "CONSTRAINT_HARD", "JWT lives in JWT_SECRET_KEY.",
            authority=ASSISTANT_ANSWER_TO_QUESTION, subject="JWT_SECRET_KEY",
        )
        buckets = partition_events([e])
        assert buckets["hard_constraints"] == []
        assert len(buckets["pending_confirmations"]) == 1
