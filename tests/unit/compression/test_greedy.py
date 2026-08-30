"""Tests for greedy knapsack fill and field-level compression."""
import pytest
from cognikernel.compression.greedy import (
    _MANDATORY_TOKEN_LIMIT,
    _MANDATORY_TYPES,
    greedy_fill,
)
from cognikernel.compression.token_count import estimate_tokens
from cognikernel.storage.events import Event


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
        from cognikernel.compression.greedy import _MANDATORY_AUTHORITIES
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


class TestReservedTokens:
    """reserved_tokens lets a caller pay for content it will append after the
    fill (the active thread), without that reservation shrinking the zone caps
    that are derived from the CONFIGURED budget."""

    @staticmethod
    def _descs(events: list[Event]) -> list[str]:
        return [e.payload["description"] for e in events]

    def test_zero_reserve_is_identical_to_omitting_the_argument(self) -> None:
        events = [_make_event(f"Event number {i}", weight=float(i)) for i in range(10)]
        assert self._descs(greedy_fill(events, 200, reserved_tokens=0)) == \
               self._descs(greedy_fill(events, 200))

    def test_reserve_reduces_phase_two_admissions(self) -> None:
        # Self-calibrating: sizes the budget from the events' real cost so the
        # test cannot silently pass if estimate_tokens changes. A hardcoded
        # budget large enough to hold everything makes this assertion vacuous.
        events = [_make_event(f"Event number {i}", weight=float(i)) for i in range(10)]
        total = sum(estimate_tokens(e) for e in events)
        budget = total + 10                      # everything fits with room to spare
        full = greedy_fill(events, budget)
        reserved = greedy_fill(events, budget, reserved_tokens=total // 2)
        assert len(full) == 10
        assert len(reserved) < 10

    def test_reserve_does_not_shrink_the_mandatory_zone(self) -> None:
        # THE GUARD for spec section 2.2. mandatory_limit = int(500 * budget/1500);
        # at budget 3500 that is 1166. These ten constraints cost 570 tokens under
        # tiktoken (installed here) and ~1030 under the len/4 heuristic — the two
        # counters disagree, but both sit between the mutated limit (166, below)
        # and the correct limit (1166), so they all fit and _compress_mandatory
        # never fires under either counter.
        # If reserved_tokens fed `scale` (the rejected design), the limit would be
        # int(500 * 500/1500) = 166 and _compress_mandatory would collapse them.
        hard = [
            _make_event("Constraint %d: %s" % (i, "x" * 380),
                        event_type="CONSTRAINT_HARD",
                        content_hash=("hard%d" % i).ljust(64, "0"))
            for i in range(10)
        ]
        without = greedy_fill(hard, 3500)
        with_reserve = greedy_fill(hard, 3500, reserved_tokens=3000)
        assert len(without) == 10
        assert len(with_reserve) == 10
