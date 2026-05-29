"""Tests for greedy knapsack fill and field-level compression."""
import pytest
from memlora.compression.greedy import (
    _MANDATORY_TOKEN_LIMIT,
    _MANDATORY_TYPES,
    compress_field_level,
    greedy_fill,
)
from memlora.compression.token_count import estimate_tokens
from memlora.storage.events import Event


def _make_event(
    description: str = "Use SQLite.",
    event_type: str = "DECISION",
    rationale: str = "",
    weight: float = 1.0,
    archived: bool = False,
    content_hash: str | None = None,
    **payload_extra,
) -> Event:
    payload = {"description": description, "rationale": rationale, **payload_extra}
    e = Event(
        project_id="p1", session_id="s1",
        event_type=event_type, payload=payload,
        content_hash=content_hash or (description[:32].ljust(64, "x")),
        weight=weight,
        archived=archived,
    )
    return e


class TestGreedyFill:
    def test_empty_events_returns_empty(self) -> None:
        assert greedy_fill([], 800) == []

    def test_all_items_fit_within_large_budget(self) -> None:
        events = [_make_event(f"Event {i}") for i in range(5)]
        for e in events:
            e.weight = 1.0
        result = greedy_fill(events, 10_000)
        assert len(result) == 5

    def test_archived_events_excluded(self) -> None:
        events = [
            _make_event("Active event", archived=False, weight=1.0),
            _make_event("Archived event", archived=True, weight=2.0),
        ]
        result = greedy_fill(events, 10_000)
        descriptions = {e.payload["description"] for e in result}
        assert "Archived event" not in descriptions

    def test_mandatory_always_included(self) -> None:
        mandatory = _make_event(
            event_type="CONSTRAINT_HARD",
            description="We cannot use Redis.",
            weight=0.001,  # low weight but mandatory
        )
        cheap_candidate = _make_event(description="Minor thing", weight=100.0)
        result = greedy_fill([mandatory, cheap_candidate], budget_tokens=5)
        mandatory_tokens = estimate_tokens(mandatory)
        included_descriptions = {e.payload["description"] for e in result}
        assert "We cannot use Redis." in included_descriptions

    def test_do_not_retry_mandatory(self) -> None:
        e = _make_event(
            event_type="APPROACH_ABANDONED_DO_NOT_RETRY",
            description="Never use Redis again.",
            weight=0.001,
        )
        result = greedy_fill([e], budget_tokens=1)
        assert any(r.payload["description"] == "Never use Redis again." for r in result)

    def test_highest_weight_selected_first(self) -> None:
        low   = _make_event("Low weight event",  weight=0.1, content_hash="l" * 64)
        high  = _make_event("High weight event", weight=5.0, content_hash="h" * 64)
        budget = estimate_tokens(high) + 1  # only one fits
        result = greedy_fill([low, high], budget)
        descriptions = {e.payload["description"] for e in result}
        assert "High weight event" in descriptions

    def test_small_items_fill_gaps_after_large_item_fails(self) -> None:
        # Large item (weight=5) doesn't fit, but two small ones do
        big   = _make_event("Big item " + "x" * 200, weight=5.0, content_hash="b" * 64)
        small = _make_event("Small A", weight=3.0, content_hash="a" * 64)
        small2 = _make_event("Small B", weight=2.0, content_hash="c" * 64)
        budget = estimate_tokens(small) + estimate_tokens(small2) + 2
        result = greedy_fill([big, small, small2], budget)
        descriptions = {e.payload["description"] for e in result}
        assert "Small A" in descriptions
        assert "Small B" in descriptions

    def test_total_tokens_within_budget(self) -> None:
        events = [_make_event(f"Event number {i} with some text", weight=float(10 - i))
                  for i in range(10)]
        budget = 30
        result = greedy_fill(events, budget)
        total = sum(estimate_tokens(e) for e in result)
        assert total <= budget

    def test_mandatory_types_constant(self) -> None:
        assert "CONSTRAINT_HARD" in _MANDATORY_TYPES
        assert "APPROACH_ABANDONED_DO_NOT_RETRY" in _MANDATORY_TYPES

    def test_mandatory_authorities_constant(self) -> None:
        from memlora.compression.greedy import _MANDATORY_AUTHORITIES
        assert "user_stated" in _MANDATORY_AUTHORITIES

    def test_user_stated_thread_survives_over_assistant_musing(self) -> None:
        """Tier-1.5: a low-weight user-stated thread must not be evicted by a
        high-weight assistant musing under budget pressure."""
        user_thread = _make_event(
            event_type="THREAD_OPEN",
            description="JWT authentication end-to-end.",
            weight=0.01,
            authority="user_stated",
            content_hash="u" * 64,
        )
        assistant_musing = _make_event(
            event_type="THREAD_OPEN",
            description="Maybe revisit membership tiers.",
            weight=5.0,
            authority="assistant_decided",
            content_hash="m" * 64,
        )
        result = greedy_fill([user_thread, assistant_musing], budget_tokens=1)
        descriptions = {e.payload["description"] for e in result}
        assert "JWT authentication end-to-end." in descriptions
        assert "Maybe revisit membership tiers." not in descriptions


class TestCompressFieldLevel:
    def test_already_under_target_unchanged(self) -> None:
        events = [_make_event("Short", rationale="Brief")]
        result = compress_field_level(events, 10_000)
        assert result[0].payload["rationale"] == "Brief"

    def test_long_rationale_truncated(self) -> None:
        long_rationale = "x" * 200
        events = [_make_event(
            "DECISION event",
            rationale=long_rationale,
            event_type="DECISION",
        )]
        original_tokens = sum(estimate_tokens(e) for e in events)
        result = compress_field_level(events, max(1, original_tokens - 10))
        assert result[0].payload["rationale"].endswith("...")
        assert len(result[0].payload["rationale"]) == 80

    def test_many_files_trimmed_to_three(self) -> None:
        events = [_make_event(
            "DECISION event",
            affected_files=["a.py", "b.py", "c.py", "d.py", "e.py"],
        )]
        original_tokens = sum(estimate_tokens(e) for e in events)
        result = compress_field_level(events, max(1, original_tokens - 5))
        files = result[0].payload.get("affected_files", [])
        assert len(files) <= 3

    def test_constraint_hard_rationale_not_compressed(self) -> None:
        long_rationale = "Security requirement — do not compress me ever." * 4
        events = [_make_event(
            "Cannot use Redis.",
            event_type="CONSTRAINT_HARD",
            rationale=long_rationale,
        )]
        original_tokens = sum(estimate_tokens(e) for e in events)
        result = compress_field_level(events, max(1, original_tokens - 5))
        # CONSTRAINT_HARD rationale should not be truncated by stage 2
        assert result[0].payload["rationale"] == long_rationale

    def test_does_not_modify_originals(self) -> None:
        e = _make_event("Event", affected_files=["a.py", "b.py", "c.py", "d.py"])
        original_files = list(e.payload["affected_files"])
        original_tokens = estimate_tokens(e)
        compress_field_level([e], max(1, original_tokens - 5))
        assert e.payload["affected_files"] == original_files

    def test_mandatory_files_not_trimmed(self) -> None:
        files = ["a.py", "b.py", "c.py", "d.py", "e.py"]
        events = [_make_event(
            "Hard constraint.",
            event_type="CONSTRAINT_HARD",
            affected_files=files,
        )]
        original_tokens = sum(estimate_tokens(e) for e in events)
        result = compress_field_level(events, max(1, original_tokens - 5))
        # CONSTRAINT_HARD files must not be trimmed at stage 1
        assert result[0].payload.get("affected_files") == files
