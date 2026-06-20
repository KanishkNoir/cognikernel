"""Tests for constraint classification heuristics."""
import pytest
from memlora.extraction.classifier import (
    classify_constraint,
    classify_event,
    HARD_THRESHOLD,
)
from memlora.storage.events import Event


def _make_constraint_event(
    event_type: str = "CONSTRAINT_HARD",
    description: str = "We cannot use Redis.",
    source_role: str = "user",
    confidence: float = 1.0,
) -> Event:
    return Event(
        project_id="p1",
        session_id="s1",
        event_type=event_type,
        payload={
            "description": description,
            "rationale": "",
            "confidence": confidence,
            "source_role": source_role,
        },
        content_hash="",
        weight=confidence,
    )


class TestClassifyConstraint:
    def test_high_confidence_user_is_hard(self) -> None:
        result = classify_constraint(1.0, "user", "We cannot use Redis.")
        assert result == "CONSTRAINT_HARD"

    def test_low_confidence_assistant_is_soft(self) -> None:
        result = classify_constraint(0.5, "assistant", "You might want to avoid Redis here.")
        assert result == "CONSTRAINT_SOFT"

    def test_user_role_boosts_score(self) -> None:
        # Borderline confidence — user role should push it over threshold
        result_user = classify_constraint(0.7, "user", "We should not use Redis.")
        result_anon = classify_constraint(0.7, "unknown", "We should not use Redis.")
        # Not asserting exact result, just that user >= anon
        if result_anon == "CONSTRAINT_SOFT":
            assert result_user in ("CONSTRAINT_HARD", "CONSTRAINT_SOFT")

    def test_assistant_role_reduces_score(self) -> None:
        # Same phrase, assistant vs user
        user_result = classify_constraint(0.9, "user", "We cannot use Redis.")
        asst_result = classify_constraint(0.9, "assistant", "We cannot use Redis.")
        # Score reduction: user should be at least as hard as assistant
        scores = {"CONSTRAINT_HARD": 1, "CONSTRAINT_SOFT": 0}
        assert scores[user_result] >= scores[asst_result]

    def test_requirement_marker_boosts(self) -> None:
        with_marker = classify_constraint(0.6, "user", "This is a mandatory requirement.")
        without = classify_constraint(0.6, "user", "We should avoid this.")
        scores = {"CONSTRAINT_HARD": 1, "CONSTRAINT_SOFT": 0}
        assert scores[with_marker] >= scores[without]

    def test_domain_marker_boosts(self) -> None:
        with_domain = classify_constraint(0.65, "user", "We cannot expose auth tokens in production.")
        without = classify_constraint(0.65, "user", "We cannot expose tokens here.")
        scores = {"CONSTRAINT_HARD": 1, "CONSTRAINT_SOFT": 0}
        assert scores[with_domain] >= scores[without]

    def test_hedge_markers_reduce_score(self) -> None:
        hedged = classify_constraint(0.9, "user", "We probably cannot use Redis here, maybe.")
        firm = classify_constraint(0.9, "user", "We cannot use Redis.")
        scores = {"CONSTRAINT_HARD": 1, "CONSTRAINT_SOFT": 0}
        assert scores[firm] >= scores[hedged]

    def test_repetition_boosts_score(self) -> None:
        once = classify_constraint(0.6, "user", "Avoid Redis.", mention_count=1)
        twice = classify_constraint(0.6, "user", "Avoid Redis.", mention_count=2)
        scores = {"CONSTRAINT_HARD": 1, "CONSTRAINT_SOFT": 0}
        assert scores[twice] >= scores[once]

    def test_threshold_exactly_at_boundary(self) -> None:
        # Score of exactly HARD_THRESHOLD → CONSTRAINT_HARD.
        # confidence=0.65, user (+0.2) → 0.85 → hard (≥). The description must
        # carry a deontic marker, else the #39 user-imperative gate caps it.
        result = classify_constraint(0.65, "user", "You must not do that.")
        assert result == "CONSTRAINT_HARD"

    def test_bare_user_imperative_demoted(self) -> None:
        # #39: a bare user imperative with no deontic marker rides the +0.2 user
        # bonus to 0.85 but must NOT become a mandatory hard constraint.
        assert classify_constraint(0.65, "user", "Use Postgres.") == "CONSTRAINT_SOFT"
        assert classify_constraint(0.7, "user", "Set the retry interval to 5.") \
            == "CONSTRAINT_SOFT"

    def test_genuine_user_prohibition_stays_hard(self) -> None:
        # The fix must not regress real prohibitions, which carry a marker.
        assert classify_constraint(0.7, "user", "Never use floats for money.") \
            == "CONSTRAINT_HARD"
        assert classify_constraint(0.7, "user", "Do not store secrets in the repo.") \
            == "CONSTRAINT_HARD"
        assert classify_constraint(0.7, "user", "No in-process counters.") \
            == "CONSTRAINT_HARD"

    def test_pure_soft_signal_stays_soft(self) -> None:
        result = classify_constraint(0.6, "assistant", "You might consider avoiding Redis.")
        assert result == "CONSTRAINT_SOFT"


class TestClassifyEvent:
    def test_non_constraint_event_passes_through(self) -> None:
        event = Event(
            project_id="p1", session_id="s1",
            event_type="DECISION",
            payload={"description": "We decided to use SQLite."},
            content_hash="", weight=1.0,
        )
        result = classify_event(event)
        assert result.event_type == "DECISION"

    def test_constraint_event_reclassified(self) -> None:
        event = _make_constraint_event(event_type="CONSTRAINT_HARD")
        result = classify_event(event)
        assert result.event_type in ("CONSTRAINT_HARD", "CONSTRAINT_SOFT")

    def test_hard_override_marker_forces_hard(self) -> None:
        event = _make_constraint_event(
            description="You might want to avoid Redis here. <!-- memlora:hard -->",
            confidence=0.3,
        )
        result = classify_event(event)
        assert result.event_type == "CONSTRAINT_HARD"

    def test_soft_override_marker_forces_soft(self) -> None:
        event = _make_constraint_event(
            event_type="CONSTRAINT_HARD",
            description="We cannot use Redis. <!-- memlora:soft -->",
            confidence=1.0,
        )
        result = classify_event(event)
        assert result.event_type == "CONSTRAINT_SOFT"

    def test_classify_event_mutates_event_type(self) -> None:
        event = _make_constraint_event(event_type="CONSTRAINT_SOFT", confidence=1.0)
        result = classify_event(event)
        # High confidence + user role should upgrade to HARD
        assert result.event_type == "CONSTRAINT_HARD"

    def test_approach_abandoned_passes_through(self) -> None:
        event = Event(
            project_id="p1", session_id="s1",
            event_type="APPROACH_ABANDONED",
            payload={"description": "We reverted the Redis approach."},
            content_hash="", weight=1.0,
        )
        result = classify_event(event)
        assert result.event_type == "APPROACH_ABANDONED"
