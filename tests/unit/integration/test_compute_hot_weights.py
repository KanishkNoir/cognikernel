"""Tests for cognikernel.integration.session._compute_hot_weights.

This is the scoring-side sibling of _compute_hot_files (which drives the
"Most active files" display list and keeps its own min_mentions cliff
unchanged). _compute_hot_weights feeds compress_to_skeleton's graded hot
bonus and must express recency continuously, not as a threshold.
"""
from __future__ import annotations

from cognikernel.integration.session import _compute_hot_weights
from cognikernel.storage.events import Event


def _component(path: str, session_id: str, created_at: int, mentions: int = 1) -> Event:
    return Event(
        project_id="p1",
        session_id=session_id,
        event_type="COMPONENT_STATUS",
        payload={"path": path, "intent": path, "description": f"{path} modified",
                 "rationale": ""},
        content_hash=("h" + path + session_id)[:32].ljust(64, "0"),
        weight=0.6,
        mention_count=mentions,
        created_at=created_at,
    )


class TestComputeHotWeights:
    def test_empty_events_returns_empty(self) -> None:
        assert _compute_hot_weights([]) == {}

    def test_non_component_status_events_ignored(self) -> None:
        events = [Event(project_id="p1", session_id="s1", event_type="DECISION",
                        payload={"affected_files": ["a.py"]}, content_hash="h" * 64,
                        weight=0.6, mention_count=1, created_at=1000)]
        assert _compute_hot_weights(events) == {}

    def test_bare_basename_filtered_out(self) -> None:
        events = [_component("env.py", "s1", 1000, mentions=8)]
        assert "env.py" not in _compute_hot_weights(events)

    def test_single_file_gets_full_weight(self) -> None:
        events = [_component("src/a.py", "s1", 1000)]
        weights = _compute_hot_weights(events)
        assert weights == {"src/a.py": 1.0}

    def test_weights_are_normalised_to_at_most_one(self) -> None:
        events = [
            _component("src/a.py", "s1", 1000, mentions=1),
            _component("src/b.py", "s1", 1000, mentions=20),
        ]
        weights = _compute_hot_weights(events)
        assert max(weights.values()) == 1.0
        assert all(0.0 < v <= 1.0 for v in weights.values())

    def test_more_recent_session_can_outrank_more_mentioned_older_one(self) -> None:
        """The real property this function exists for: recency competes with
        (and here, beats) raw mention count — the flat binary scheme this
        replaces could only see mention count and always got this backwards."""
        events = [
            # session 1 of 3 (oldest): mentioned twice
            _component("src/old_favorite.py", "s1", 1_000, mentions=2),
            # session 2 of 3 (middle) — establishes the gap; unrelated file
            _component("src/unrelated.py", "s2", 2_000, mentions=1),
            # session 3 of 3 (newest): mentioned once
            _component("src/just_edited.py", "s3", 3_000, mentions=1),
        ]
        weights = _compute_hot_weights(events)
        assert weights["src/just_edited.py"] > weights["src/old_favorite.py"]

    def test_same_session_repeated_mentions_score_higher_than_single(self) -> None:
        events = [
            _component("src/mentioned_once.py", "s1", 1_000, mentions=1),
            _component("src/mentioned_often.py", "s1", 1_000, mentions=5),
        ]
        weights = _compute_hot_weights(events)
        assert weights["src/mentioned_often.py"] > weights["src/mentioned_once.py"]

    def test_session_order_derived_from_created_at_not_list_order(self) -> None:
        """Events for the later session appear first in the input list — the
        function must sort by created_at, not trust arrival order."""
        events = [
            _component("src/newer.py", "s2", 5_000, mentions=1),
            _component("src/older.py", "s1", 1_000, mentions=1),
        ]
        weights = _compute_hot_weights(events)
        assert weights["src/newer.py"] > weights["src/older.py"]

    def test_multiple_mentions_of_same_file_across_sessions_accumulate(self) -> None:
        events = [
            _component("src/a.py", "s1", 1_000, mentions=1),
            _component("src/a.py", "s2", 2_000, mentions=1),
            _component("src/b.py", "s2", 2_000, mentions=1),
        ]
        weights = _compute_hot_weights(events)
        # a.py was touched in both sessions; b.py only in the latest one.
        assert weights["src/a.py"] >= weights["src/b.py"]
