"""Tests for the composite weight formula."""
import math

import pytest

from cognikernel.compression.weights import (
    BASE_WEIGHT,
    TYPE_MULTIPLIER,
    activity_factor,
    compute_weight,
    repetition_factor,
)
from cognikernel.storage.events import Event


def _make_event(**overrides) -> Event:
    defaults = dict(
        project_id="p1", session_id="s1",
        event_type="DECISION",
        payload={"description": "Use SQLite", "rationale": ""},
        content_hash="a" * 64, weight=1.0,
        mention_count=1,
    )
    defaults.update(overrides)
    e = Event(**defaults)
    e.last_mentioned_session = overrides.get("last_mentioned_session", 0)
    return e


class TestRepetitionFactor:
    def test_mention_count_one_returns_one(self) -> None:
        assert repetition_factor(1) == pytest.approx(1.0)

    def test_grows_with_mention_count(self) -> None:
        assert repetition_factor(2) > repetition_factor(1)
        assert repetition_factor(10) > repetition_factor(2)

    def test_logarithmic_formula(self) -> None:
        assert repetition_factor(5) == pytest.approx(1.0 + 0.3 * math.log(5))

    def test_saturates_sublinearly(self) -> None:
        # Growth from 1→10 should be less than 10× growth from 1→2
        gain_1_to_10 = repetition_factor(10) - repetition_factor(1)
        gain_1_to_2 = repetition_factor(2) - repetition_factor(1)
        assert gain_1_to_10 < 10 * gain_1_to_2

    def test_mention_count_zero_treated_as_one(self) -> None:
        assert repetition_factor(0) == pytest.approx(1.0)

    def test_high_mention_count_still_finite(self) -> None:
        assert repetition_factor(1000) < 10.0


class TestActivityFactor:
    def test_no_file_paths_returns_one(self) -> None:
        assert activity_factor([], {}) == pytest.approx(1.0)

    def test_in_flux_returns_two(self) -> None:
        cmap = {"src/auth.py": {"status": "in_flux"}}
        assert activity_factor(["src/auth.py"], cmap) == pytest.approx(2.0)

    def test_stable_suppressed(self) -> None:
        cmap = {"src/old.py": {"status": "stable"}}
        assert activity_factor(["src/old.py"], cmap) < 1.0

    def test_abandoned_strongly_suppressed(self) -> None:
        cmap = {"src/dead.py": {"status": "abandoned"}}
        assert activity_factor(["src/dead.py"], cmap) == pytest.approx(0.3)

    def test_unknown_status_returns_one(self) -> None:
        cmap = {"src/x.py": {"status": "unknown"}}
        assert activity_factor(["src/x.py"], cmap) == pytest.approx(1.0)

    def test_missing_file_treated_as_unknown(self) -> None:
        assert activity_factor(["not_in_map.py"], {}) == pytest.approx(1.0)

    def test_returns_max_across_files(self) -> None:
        cmap = {
            "a.py": {"status": "stable"},
            "b.py": {"status": "in_flux"},
        }
        assert activity_factor(["a.py", "b.py"], cmap) == pytest.approx(2.0)

    def test_blocked_higher_than_needs_review(self) -> None:
        cmap_blocked = {"a.py": {"status": "blocked"}}
        cmap_review  = {"a.py": {"status": "needs_review"}}
        assert activity_factor(["a.py"], cmap_blocked) > activity_factor(["a.py"], cmap_review)


class TestBaseWeightAndTypeMultiplier:
    def test_hard_constraint_highest_base(self) -> None:
        assert BASE_WEIGHT["CONSTRAINT_HARD"] >= max(
            v for k, v in BASE_WEIGHT.items() if k != "CONSTRAINT_HARD"
        )

    def test_thread_close_lowest_base(self) -> None:
        assert BASE_WEIGHT["THREAD_CLOSE"] == min(BASE_WEIGHT.values())

    def test_hard_constraint_highest_type_multiplier(self) -> None:
        assert TYPE_MULTIPLIER["CONSTRAINT_HARD"] == max(TYPE_MULTIPLIER.values())

    def test_thread_close_lowest_type_multiplier(self) -> None:
        assert TYPE_MULTIPLIER["THREAD_CLOSE"] == min(TYPE_MULTIPLIER.values())

    def test_all_valid_event_types_covered(self) -> None:
        from cognikernel.storage.events import VALID_EVENT_TYPES
        assert set(BASE_WEIGHT.keys()) == VALID_EVENT_TYPES
        assert set(TYPE_MULTIPLIER.keys()) == VALID_EVENT_TYPES


class TestComputeWeight:
    def test_returns_positive_float(self) -> None:
        e = _make_event()
        w = compute_weight(e, {}, {})
        assert w > 0.0

    def test_hard_constraint_higher_than_thread_close(self) -> None:
        hard = _make_event(event_type="CONSTRAINT_HARD")
        close = _make_event(event_type="THREAD_CLOSE")
        assert compute_weight(hard, {}, {}) > compute_weight(close, {}, {})

    def test_recency_reduces_weight_for_old_events(self) -> None:
        recent = _make_event(last_mentioned_session=9)
        old = _make_event(last_mentioned_session=0)
        w_recent = compute_weight(recent, {}, {}, current_session=10)
        w_old = compute_weight(old, {}, {}, current_session=10)
        assert w_recent > w_old

    def test_higher_mention_count_raises_weight(self) -> None:
        once = _make_event(mention_count=1)
        five = _make_event(mention_count=5)
        assert compute_weight(five, {}, {}) > compute_weight(once, {}, {})

    def test_in_flux_file_boosts_weight(self) -> None:
        cmap = {"auth.py": {"status": "in_flux"}}
        with_flux = _make_event(payload={
            "description": "Use SQLite", "rationale": "", "affected_files": ["auth.py"]
        })
        without = _make_event()
        assert compute_weight(with_flux, cmap, {}) > compute_weight(without, {}, {})

    def test_full_formula_example_from_deep_dive(self) -> None:
        # Hard constraint, 5 mentions, last session 2 ago, in_flux + high centrality
        e = _make_event(
            event_type="CONSTRAINT_HARD", mention_count=5,
            last_mentioned_session=8,
            payload={"description": "No Redis", "rationale": "", "affected_files": ["auth.py"]},
        )
        cmap = {"auth.py": {"status": "in_flux"}}
        centrality_map = {"auth.py": 1.0}
        w = compute_weight(e, cmap, centrality_map, current_session=10)
        # Expected ≈ 1.0 × (1/1.3) × 1.48 × 1.5 × 2.0 × 1.5 ≈ high
        assert w > 4.0

    def test_old_stable_thread_close_near_zero(self) -> None:
        e = _make_event(event_type="THREAD_CLOSE", last_mentioned_session=0)
        cmap = {"stable.py": {"status": "stable"}}
        e.payload["affected_files"] = ["stable.py"]
        w = compute_weight(e, cmap, {}, current_session=50)
        assert w < 0.1
