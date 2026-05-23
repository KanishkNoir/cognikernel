"""Tests for event partitioning and InjectionContext construction."""
import pytest
from memlora.injection.ordering import make_injection_context, partition_events
from memlora.storage.events import Event, VALID_EVENT_TYPES


def _event(event_type: str, description: str = "x", **payload_extra) -> Event:
    return Event(
        project_id="p1", session_id="s1",
        event_type=event_type,
        payload={"description": description, "rationale": "", **payload_extra},
        content_hash=description[:32].ljust(64, "0"),
        weight=1.0,
    )


class TestPartitionEvents:
    def test_constraint_hard_goes_to_hard_constraints(self) -> None:
        e = _event("CONSTRAINT_HARD", "No Redis.")
        result = partition_events([e])
        assert e in result["hard_constraints"]
        assert e not in result["decisions"]

    def test_do_not_retry_goes_to_graveyard(self) -> None:
        e = _event("APPROACH_ABANDONED_DO_NOT_RETRY", "No Redis ever.")
        result = partition_events([e])
        assert e in result["graveyard"]

    def test_component_status_goes_to_components(self) -> None:
        e = _event("COMPONENT_STATUS", "Modified", path="src/auth.py")
        result = partition_events([e])
        assert e in result["components"]

    def test_decision_goes_to_decisions(self) -> None:
        e = _event("DECISION", "Use SQLite.")
        result = partition_events([e])
        assert e in result["decisions"]

    def test_constraint_soft_goes_to_decisions(self) -> None:
        e = _event("CONSTRAINT_SOFT", "Prefer async.")
        result = partition_events([e])
        assert e in result["decisions"]

    def test_approach_abandoned_goes_to_decisions(self) -> None:
        e = _event("APPROACH_ABANDONED", "Tried Redis.")
        result = partition_events([e])
        assert e in result["decisions"]

    def test_thread_open_goes_to_active_threads(self) -> None:
        e = _event("THREAD_OPEN", "Auth work.")
        result = partition_events([e])
        assert e in result["active_threads"]

    def test_thread_close_excluded(self) -> None:
        e = _event("THREAD_CLOSE", "Finished auth.")
        result = partition_events([e])
        for bucket in result.values():
            assert e not in bucket

    def test_empty_input_returns_empty_buckets(self) -> None:
        result = partition_events([])
        for bucket in result.values():
            assert bucket == []

    def test_all_buckets_present_in_result(self) -> None:
        result = partition_events([])
        assert set(result.keys()) == {
            "hard_constraints", "graveyard", "components",
            "decisions", "active_threads",
        }

    def test_mixed_events_partitioned_correctly(self) -> None:
        events = [
            _event("CONSTRAINT_HARD", "A"),
            _event("DECISION", "B"),
            _event("COMPONENT_STATUS", "C", path="x.py"),
            _event("THREAD_CLOSE", "D"),
        ]
        result = partition_events(events)
        assert len(result["hard_constraints"]) == 1
        assert len(result["decisions"]) == 1
        assert len(result["components"]) == 1
        total_in_buckets = sum(len(v) for v in result.values())
        assert total_in_buckets == 3  # THREAD_CLOSE excluded


class TestMakeInjectionContext:
    def test_returns_injection_context(self) -> None:
        from memlora.injection.template import InjectionContext
        ctx = make_injection_context([], "proj", 1, 5, 1)
        assert isinstance(ctx, InjectionContext)

    def test_project_name_set(self) -> None:
        ctx = make_injection_context([], "myrepo", 1, 5, 1)
        assert ctx.project_name == "myrepo"

    def test_session_numbers_set(self) -> None:
        ctx = make_injection_context([], "proj", 3, 10, 2)
        assert ctx.session_number == 3
        assert ctx.total_sessions == 10
        assert ctx.state_version == 2

    def test_events_partitioned_into_sections(self) -> None:
        events = [
            _event("CONSTRAINT_HARD", "No Redis."),
            _event("DECISION", "Use SQLite."),
        ]
        ctx = make_injection_context(events, "proj", 1, 1, 1)
        assert len(ctx.hard_constraints) == 1
        assert len(ctx.decisions) == 1

    def test_summary_text_generated(self) -> None:
        events = [_event("COMPONENT_STATUS", "Modified", path="src/app.py")]
        ctx = make_injection_context(events, "proj", 1, 1, 1)
        assert isinstance(ctx.summary_text, str)
        assert len(ctx.summary_text) > 0

    def test_token_budget_default(self) -> None:
        ctx = make_injection_context([], "proj", 1, 1, 1)
        assert ctx.token_budget == 2000

    def test_token_budget_custom(self) -> None:
        ctx = make_injection_context([], "proj", 1, 1, 1, token_budget=400)
        assert ctx.token_budget == 400
