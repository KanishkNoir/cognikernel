"""Tests for event partitioning and InjectionContext construction."""
import pytest
from cognikernel.extraction.authority import (
    ASSISTANT_ANSWER_TO_QUESTION,
    ASSISTANT_DECIDED,
    INFERRED_FROM_CODE,
    LLM,
    USER_STATED,
)
from cognikernel.injection.ordering import make_injection_context, partition_events, select_active_thread
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


class TestHandoffOutranksNarration:
    """#25 item 2: at equal authority, an explicit handoff to a later session
    outranks narration, whatever the weight.

    Without this term the sort is (authority, -weight), so every assistant
    statement typed THREAD_OPEN sits at one priority and the heaviest wins —
    and weight has no view on whether something is queued. Measured on the
    four benchmark stores at the session boundaries the probes actually ask
    at, a genuine handoff existed but narration was rendered at 4 boundaries.

    The fixtures below are the real Relay pair from that measurement.
    """

    _HANDOFF = ("That's the shape of the router; next session picks up here "
                "rather than re-deriving any of it.")
    _NARRATION = "Running the new integration test now."

    def test_handoff_beats_heavier_narration_at_equal_authority(self) -> None:
        narration = _event("THREAD_OPEN", self._NARRATION,
                           weight=0.63, authority=ASSISTANT_DECIDED)
        handoff = _event("THREAD_OPEN", self._HANDOFF,
                         weight=0.41, authority=ASSISTANT_DECIDED)
        result = partition_events([narration, handoff])
        assert result["active_threads"][0] is handoff
        assert select_active_thread([narration, handoff]) is handoff

    def test_authority_still_outranks_handoff(self) -> None:
        """The term sits BELOW authority on purpose. Promoting handoffs above
        it would let an assistant's note outrank what the user said is
        outstanding — D9's defect, inverted."""
        user_thread = _event("THREAD_OPEN", "Finish the token refresh path.",
                             weight=0.1, authority=USER_STATED)
        assistant_handoff = _event("THREAD_OPEN", self._HANDOFF,
                                   weight=5.0, authority=ASSISTANT_DECIDED)
        result = partition_events([assistant_handoff, user_thread])
        assert result["active_threads"][0] is user_thread

    def test_weight_still_decides_between_two_handoffs(self) -> None:
        light = _event("THREAD_OPEN", "Next session: finish the parser.",
                       weight=0.4, authority=ASSISTANT_DECIDED)
        heavy = _event("THREAD_OPEN", "Next session: finish the scheduler.",
                       weight=0.9, authority=ASSISTANT_DECIDED)
        result = partition_events([light, heavy])
        assert result["active_threads"][0] is heavy

    def test_weight_still_decides_between_two_narrations(self) -> None:
        """Unchanged behaviour where no handoff is present — which is every
        Conductor boundary in the corpus (that project states no handoff at
        all, and selecting narration there is not a defect)."""
        light = _event("THREAD_OPEN", "Now writing the tests.",
                       weight=0.4, authority=ASSISTANT_DECIDED)
        heavy = _event("THREAD_OPEN", "Now running mypy and ruff.",
                       weight=0.9, authority=ASSISTANT_DECIDED)
        result = partition_events([light, heavy])
        assert result["active_threads"][0] is heavy

    def test_narration_only_slate_still_fills_the_slot(self) -> None:
        """The term reorders candidates; it must never empty the section."""
        narration = _event("THREAD_OPEN", self._NARRATION,
                           weight=0.63, authority=ASSISTANT_DECIDED)
        assert select_active_thread([narration]) is narration


class TestSelectActiveThread:
    """The selector must agree with partition_events by construction, because
    render_state reserves budget for whatever it returns. If it picks an event
    that partition_events routes elsewhere, the reserve is spent on a section
    that renders nothing."""

    def test_returns_none_for_no_events(self) -> None:
        assert select_active_thread([]) is None

    def test_returns_none_when_no_threads_present(self) -> None:
        assert select_active_thread([_event("DECISION", "Use SQLite.")]) is None

    def test_picks_user_stated_over_higher_weight_assistant(self) -> None:
        user = _event("THREAD_OPEN", "Implement JWT auth.",
                      weight=0.5, authority=USER_STATED)
        assistant = _event("THREAD_OPEN", "Maybe a context manager.",
                           weight=2.0, authority=ASSISTANT_DECIDED)
        assert select_active_thread([assistant, user]) is user

    def test_picks_higher_weight_within_same_authority(self) -> None:
        low = _event("THREAD_OPEN", "Less important.", weight=0.5, authority=USER_STATED)
        high = _event("THREAD_OPEN", "Critical.", weight=1.5, authority=USER_STATED)
        assert select_active_thread([low, high]) is high

    def test_ignores_thread_routed_to_pending_confirmations(self) -> None:
        # partition_events diverts on authority BEFORE event_type (ordering.py:86-92),
        # so this thread never reaches the active_threads bucket. Selecting it
        # would reserve budget for a section that then renders nothing.
        t = _event("THREAD_OPEN", "Is it Postgres?", weight=5.0,
                   authority=ASSISTANT_ANSWER_TO_QUESTION)
        assert select_active_thread([t]) is None

    def test_agrees_with_partition_events_first_thread(self) -> None:
        a = _event("THREAD_OPEN", "a", weight=1.0, authority=ASSISTANT_DECIDED)
        b = _event("THREAD_OPEN", "b", weight=1.0, authority=USER_STATED)
        events = [a, b]
        assert select_active_thread(events) is partition_events(events)["active_threads"][0]
