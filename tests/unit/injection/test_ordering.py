"""Tests for event partitioning and InjectionContext construction."""
import pytest
from cognikernel.extraction.authority import (
    ASSISTANT_ANSWER_TO_QUESTION,
    ASSISTANT_DECIDED,
    INFERRED_FROM_CODE,
    LLM,
    USER_STATED,
)
from cognikernel.injection.ordering import make_injection_context, partition_events
from cognikernel.storage.events import Event, VALID_EVENT_TYPES


def _event(event_type: str, description: str = "x", *, weight: float = 1.0, **payload_extra) -> Event:
    return Event(
        project_id="p1", session_id="s1",
        event_type=event_type,
        payload={"description": description, "rationale": "", **payload_extra},
        content_hash=description[:32].ljust(64, "0"),
        weight=weight,
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
            "decisions", "active_threads", "pending_confirmations",
        }

    def test_mixed_events_partitioned_correctly(self) -> None:
        events = [
            _event("CONSTRAINT_HARD", "A"),
            _event("DECISION", "B"),
            # Qualified path so the bare-basename filter doesn't drop it.
            _event("COMPONENT_STATUS", "C", path="src/x.py"),
            _event("THREAD_CLOSE", "D"),
        ]
        result = partition_events(events)
        assert len(result["hard_constraints"]) == 1
        assert len(result["decisions"]) == 1
        assert len(result["components"]) == 1
        total_in_buckets = sum(len(v) for v in result.values())
        assert total_in_buckets == 3  # THREAD_CLOSE excluded


class TestActiveThreadRanking:
    """The singular `Active thread` slot in the renderer takes threads[0].

    `partition_events` must therefore sort the active_threads bucket so the
    user's explicit directive wins over higher-weight assistant musings.
    This is the T1 mis-ranking fix from the post-Arm-C-v2 analysis.
    """

    def test_user_stated_beats_higher_weight_assistant_decided(self) -> None:
        user = _event("THREAD_OPEN", "Implement JWT auth end-to-end.",
                      weight=0.5, authority=USER_STATED)
        assistant = _event("THREAD_OPEN", "Maybe try a context manager pattern.",
                           weight=2.0, authority=ASSISTANT_DECIDED)
        result = partition_events([assistant, user])
        # Even though assistant has 4x the weight, user_stated wins the slot.
        assert result["active_threads"][0] is user
        assert result["active_threads"][1] is assistant

    def test_higher_weight_wins_within_same_authority(self) -> None:
        low = _event("THREAD_OPEN", "Less important user thread.",
                     weight=0.5, authority=USER_STATED)
        high = _event("THREAD_OPEN", "Critical user thread.",
                      weight=1.5, authority=USER_STATED)
        result = partition_events([low, high])
        assert result["active_threads"][0] is high
        assert result["active_threads"][1] is low

    def test_priority_order_across_authorities_reaching_threads_bucket(self) -> None:
        # ASSISTANT_ANSWER_TO_QUESTION events route to pending_confirmations
        # before they can reach the active_threads bucket, so they're not
        # included here. The remaining four authorities are the realistic set
        # for THREAD_OPEN events.
        user = _event("THREAD_OPEN", "u", weight=1.0, authority=USER_STATED)
        llm = _event("THREAD_OPEN", "l", weight=1.0, authority=LLM)
        assistant = _event("THREAD_OPEN", "d", weight=1.0, authority=ASSISTANT_DECIDED)
        inferred = _event("THREAD_OPEN", "i", weight=1.0, authority=INFERRED_FROM_CODE)
        # Pass in deliberately-wrong order to prove the sort takes over.
        result = partition_events([inferred, assistant, llm, user])
        assert [e.payload["description"] for e in result["active_threads"]] == [
            "u", "l", "d", "i",
        ]

    def test_missing_authority_falls_back_to_lowest_priority(self) -> None:
        user = _event("THREAD_OPEN", "User directive.",
                      weight=0.1, authority=USER_STATED)
        no_auth = _event("THREAD_OPEN", "Legacy event without authority.",
                         weight=10.0)  # no authority field
        result = partition_events([no_auth, user])
        # User wins despite weight 0.1 vs 10.0.
        assert result["active_threads"][0] is user

    def test_T1_regression_scenario(self) -> None:
        """Reproduces the Arm-C-v2 T1 mis-ranking exactly.

        In S3, the user's S1 directive 'implement JWT authentication end-to-end'
        (authority=user_stated, weight 0.89) was beaten for the Active thread
        slot by an assistant musing about transaction management
        (authority=assistant_decided, weight 2.05). The fix must surface T1.
        """
        t1 = _event(
            "THREAD_OPEN",
            "We need to implement JWT authentication end-to-end.",
            weight=0.89, authority=USER_STATED,
        )
        musing = _event(
            "THREAD_OPEN",
            "The alternative (catch Exception, rollback, re-raise) hides…",
            weight=2.05, authority=ASSISTANT_DECIDED,
        )
        result = partition_events([musing, t1])
        # Renderer takes [0] — must be T1.
        assert "JWT authentication" in result["active_threads"][0].payload["description"]


class TestMakeInjectionContext:
    def test_returns_injection_context(self) -> None:
        from cognikernel.injection.template import InjectionContext
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

    def test_bare_basename_component_dropped(self) -> None:
        """COMPONENT_STATUS with a bare-basename path is rejected by partition_events."""
        events = [_event("COMPONENT_STATUS", "Modified", path="config.py")]
        result = partition_events(events)
        assert result["components"] == []

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
