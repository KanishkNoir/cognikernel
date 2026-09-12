"""Admission gate behaviour (spec §4)."""
from cognikernel.model import Event
from cognikernel.quality.gate import GroundingContext, Verdict, admit


def _event(event_type: str, description: str, **payload) -> Event:
    return Event(
        project_id="p",
        session_id="s",
        event_type=event_type,
        payload={"description": description, **payload},
        content_hash="h",
        weight=1.0,
    )


class TestStatementRules:
    def test_admits_well_formed_decision(self) -> None:
        v = admit(_event("DECISION", "The dispatcher retries twice before dead-lettering."))
        assert v.action == "admit"

    def test_downgrades_subject_less_statement(self) -> None:
        # DOWNGRADE, not reject: the harm is that it renders in the block, and
        # weight collapse already prevents that. Rejecting would also remove it
        # from recall/find_related, which is strictly more destructive. This
        # matches the policy pipeline.py already states for context-dependent
        # fragments ("We DEMOTE (not drop)").
        v = admit(_event("DECISION", "It must not be able to take down the pipeline."))
        assert v.action == "downgrade"
        assert v.rule_id == "D7"

    def test_subject_less_downgrade_is_not_applied_twice(self) -> None:
        # v1/v2 head paths already demote fragments pre-hash (_FRAG_DEMOTE).
        # The gate must not stack a second multiplier on the same event.
        e = _event("DECISION", "It must not take down the pipeline.")
        e.payload["provenance"] = "salience_v2+frag"
        assert admit(e).action == "admit"

    def test_rejects_boilerplate(self) -> None:
        v = admit(_event("CONSTRAINT_HARD", "Pick up the last task as if the break never happened."))
        assert v.action == "reject"
        assert v.rule_id == "D4"

    def test_rejects_interrogative_constraint(self) -> None:
        v = admit(_event("CONSTRAINT_HARD", "Should we just bring in Celery?"))
        assert v.action == "reject"
        assert v.rule_id == "D2"

    def test_admits_imperative_constraint(self) -> None:
        v = admit(_event("CONSTRAINT_HARD", "Do not cache in Redis."))
        assert v.action == "admit"


class TestBoilerplateIsTypeIndependent:
    """Harness chatter is never a project fact, whatever type it was classified
    as. Found end-to-end: the same compaction sentence was extracted twice, and
    the CONSTRAINT_HARD copy was rejected while the THREAD_OPEN copy sailed
    through because D4 was scoped to statement types.
    """

    def test_rejects_boilerplate_as_thread_open(self) -> None:
        v = admit(_event("THREAD_OPEN",
                         "Pick up the last task as if the break never happened."))
        assert v.action == "reject"
        assert v.rule_id == "D4"

    def test_rejects_boilerplate_as_component_status(self) -> None:
        v = admit(_event("COMPONENT_STATUS",
                         "read the full transcript at: C:/x/a.jsonl", path="a/b.py"))
        assert v.action == "reject"
        assert v.rule_id == "D4"

    def test_ordinary_thread_open_still_admitted(self) -> None:
        v = admit(_event("THREAD_OPEN", "Wire the dispatcher into the worker pool."))
        assert v.action == "admit"

    def test_subject_less_still_scoped_to_statements(self) -> None:
        # D7 stays statement-scoped: a THREAD_OPEN naturally references the
        # current work item and is not defective for doing so.
        v = admit(_event("THREAD_OPEN", "It must not take down the pipeline."))
        assert v.action == "admit"


class TestStepNarration:
    """D11 — the assistant announcing its next step, typed as a decision by the head.

    In the 2026-09-12 micro benchmark store, 24 step announcements ("Now let's
    run the full test suite.") were stored at head confidence 0.66–0.99.
    """

    def test_downgrades_assistant_step_narration(self) -> None:
        v = admit(_event("DECISION", "Now update dispatcher.py to use the renamed store API.",
                         source_role="assistant"))
        assert (v.action, v.rule_id) == ("downgrade", "D11")

    def test_marks_it_as_step_narration(self) -> None:
        from cognikernel.quality.gate import apply_verdict

        e = _event("THREAD_OPEN", "Now let's run the full test suite.", source_role="assistant")
        apply_verdict(e, admit(e))
        assert e.payload["quality"] == "step_narration"

    def test_the_user_saying_it_is_admitted(self) -> None:
        v = admit(_event("DECISION", "Now update dispatcher.py to use the renamed store API.",
                         source_role="user"))
        assert v.action == "admit"

    def test_a_decision_with_its_reason_is_admitted(self) -> None:
        v = admit(_event("DECISION", "Let's use the SDK client instead, which matches production.",
                         source_role="assistant"))
        assert v.action == "admit"


class TestGrounding:
    def test_admits_known_path(self) -> None:
        g = GroundingContext(frozenset({"src/storage/connection.py"}))
        v = admit(_event("COMPONENT_STATUS", "x", path="src/storage/connection.py"), g)
        assert v.action == "admit"

    def test_downgrades_unknown_path(self) -> None:
        # A genuinely new file must survive — downgraded, never dropped.
        g = GroundingContext(frozenset({"src/storage/connection.py"}))
        v = admit(_event("COMPONENT_STATUS", "x", path="src/brand/new_file.py"), g)
        assert v.action == "downgrade"

    def test_rejects_near_miss_truncation(self) -> None:
        g = GroundingContext(frozenset({"src/storage/connection.py"}))
        v = admit(_event("COMPONENT_STATUS", "x", path="rc/storage/connection.py"), g)
        assert v.action == "reject"
        assert v.rule_id == "D1"

    def test_no_grounding_context_admits(self) -> None:
        v = admit(_event("COMPONENT_STATUS", "x", path="anything/at/all.py"), None)
        assert v.action == "admit"


class TestFailOpen:
    def test_detector_exception_admits(self, monkeypatch) -> None:
        import cognikernel.quality.gate as gate_mod

        def boom(*_a, **_k):
            raise RuntimeError("detector exploded")

        monkeypatch.setattr(gate_mod, "detect_subject_less", boom)
        v = admit(_event("DECISION", "It must not take down the pipeline."))
        assert v.action == "admit"
        assert v.rule_id == "gate_error"

    def test_verdict_is_a_verdict(self) -> None:
        assert isinstance(admit(_event("DECISION", "The queue is durable.")), Verdict)


class TestApplyVerdict:
    def test_downgrade_halves_weight_and_marks_quality(self) -> None:
        from cognikernel.quality.gate import apply_verdict

        e = _event("DECISION", "It must not take down the pipeline.")
        apply_verdict(e, admit(e))
        assert e.weight == 0.5
        assert e.payload["quality"] == "context_dependent"

    def test_path_downgrade_marks_grounding(self) -> None:
        from cognikernel.quality.gate import apply_verdict

        g = GroundingContext(frozenset({"src/known.py"}))
        e = _event("COMPONENT_STATUS", "x", path="src/new.py")
        apply_verdict(e, admit(e, g))
        assert e.payload["grounding"] == "unverified"

    def test_admit_leaves_event_untouched(self) -> None:
        from cognikernel.quality.gate import apply_verdict

        e = _event("DECISION", "The queue is durable.")
        apply_verdict(e, admit(e))
        assert e.weight == 1.0
        assert "quality" not in e.payload


class TestBareInstructionThread:
    """D9 (T-202a / #20 Defect A): an ordinary instruction must not hold the
    top authority tier a genuinely-queued thread needs to win selection.

    Both decisive fixtures below are verbatim from the real Taskflow store —
    the instruction that actually won the graded probe, and the gold thread
    it beat (research/fixes/thread_precision.md)."""

    def test_bare_instruction_is_downgraded(self) -> None:
        e = _event("THREAD_OPEN", "Add the Pydantic response schema for a task.",
                   authority="user_stated")
        v = admit(e)
        assert v.action == "downgrade"
        assert v.rule_id == "D9"

    def test_genuine_deferral_is_admitted(self) -> None:
        """A SELF-CONTAINED deferral keeps its tier.

        The fixture here was originally "This is the active work item for the
        next session." — which D9 does admit, since it plainly refers to
        deferred work. D10 now demotes it on a different axis (it is a pointer
        at a thread, not a statement of one: #30), so it is no longer a valid
        fixture for "D9 leaves real threads alone". Replaced with a deferral
        that names its own subject; the old sentence is covered by
        TestAnaphoricThreadReference below.
        """
        e = _event("THREAD_OPEN",
                   "Next session's focus: implementing the fallback+retry router.",
                   authority="user_stated")
        assert admit(e).action == "admit"

    def test_obligation_phrasing_is_admitted(self) -> None:
        """The gold Taskflow thread. An earlier draft of the marker list
        missed plain 'need to' and would have demoted exactly this — the
        false negative the real-data validation caught."""
        e = _event(
            "THREAD_OPEN",
            "We need to implement JWT authentication end-to-end — login endpoint, "
            "token issuance, and the FastAPI dependency guard.",
            authority="user_stated",
        )
        assert admit(e).action == "admit"

    def test_demotion_changes_authority_not_provenance(self) -> None:
        from cognikernel.quality.gate import apply_verdict

        e = _event("THREAD_OPEN", "Write the cache lookup.",
                   authority="user_stated", source_role="user")
        apply_verdict(e, admit(e))
        assert e.payload["authority"] == "assistant_decided"
        assert e.payload["quality"] == "instruction_not_thread"
        # Who actually said it is untouched — this changes precedence only.
        assert e.payload["source_role"] == "user"

    def test_only_applies_to_thread_open(self) -> None:
        """A DECISION phrased as a bare instruction is not this defect — the
        rule is scoped to the one type whose selection this distorts."""
        e = _event("DECISION", "Add the Pydantic response schema for a task.",
                   authority="user_stated")
        assert admit(e).action == "admit"

    def test_only_applies_to_user_stated(self) -> None:
        """A thread already below the top tier has nothing to demote, so the
        rule must not fire on it (and must not double-halve its weight)."""
        e = _event("THREAD_OPEN", "Now writing the tests.",
                   authority="assistant_decided")
        assert admit(e).action == "admit"


class TestFutureSessionHandoffPredicate:
    """#25: the narrow predicate that shields a thread from recency
    supersession. It is deliberately NOT the same vocabulary as D9's — the
    two have opposite failure costs, so a merge would break one of them.
    """

    def test_flags_explicit_next_session_handoff(self) -> None:
        from cognikernel.quality.detectors import describes_future_session_handoff

        assert describes_future_session_handoff(
            "Next session's focus: implementing the fallback+retry router.")
        assert describes_future_session_handoff(
            "So the next step is picking back up on toolbelt/retry.py.")
        assert describes_future_session_handoff(
            "This is the active work item for the next session.")

    def test_does_not_flag_narration_the_broad_predicate_matches(self) -> None:
        """The whole reason this predicate exists. All four are real store
        descriptions that describes_deferred_work flags on incidental wording;
        shielding them would restore the narration pile-up."""
        from cognikernel.quality.detectors import (
            describes_deferred_work,
            describes_future_session_handoff,
        )

        narration = [
            "Continuing with tests now.",
            "Continuing with the package scaffolding.",
            "Let me fix the remaining long line manually.",
            "Now let's fix the one remaining long line.",
        ]
        for text in narration:
            assert describes_deferred_work(text), text     # broad matches
            assert not describes_future_session_handoff(text), text   # narrow does not

    def test_narrow_is_a_subset_of_broad_on_real_handoffs(self) -> None:
        """Anything explicit enough to be a handoff is also deferred work, so
        the shield can never protect something D9 would have demoted."""
        from cognikernel.quality.detectors import (
            describes_deferred_work,
            describes_future_session_handoff,
        )

        for text in ["Next session's focus: the router.",
                     "Queued for next time.",
                     "Ready to continue with the migration whenever you are."]:
            assert describes_future_session_handoff(text)
            assert describes_deferred_work(text)

    def test_empty_and_none_are_not_handoffs(self) -> None:
        from cognikernel.quality.detectors import describes_future_session_handoff

        assert not describes_future_session_handoff("")
        assert not describes_future_session_handoff(None)  # type: ignore[arg-type]


class TestAnaphoricThreadReference:
    """D10 (#30): a user-stated thread that only points at another thread.

    The extractor splits a turn into sentences, so a pointer and its own
    antecedent arrive as two competing THREAD_OPEN events. In the real Taskflow
    store they were 44ms apart, and the pointer superseded the statement it
    refers to — leaving the store holding the pronoun and not the referent.
    """

    _POINTER = "This is the active work item for the next session."
    _ANTECEDENT = ("We need to implement JWT authentication end-to-end — login "
                   "endpoint, token issuance, and the FastAPI dependency guard.")

    def test_pointer_is_demoted(self) -> None:
        v = admit(_event("THREAD_OPEN", self._POINTER, authority="user_stated"))
        assert v.action == "downgrade"
        assert v.rule_id == "D10"

    def test_its_antecedent_is_untouched(self) -> None:
        assert admit(
            _event("THREAD_OPEN", self._ANTECEDENT, authority="user_stated")
        ).action == "admit"

    def test_demotion_marks_it_as_a_reference_not_an_instruction(self) -> None:
        """D9 and D10 both demote, but they say different things about why —
        the debugger should be able to tell them apart."""
        from cognikernel.quality.gate import apply_verdict

        e = _event("THREAD_OPEN", self._POINTER,
                   authority="user_stated", source_role="user")
        apply_verdict(e, admit(e))
        assert e.payload["authority"] == "assistant_decided"
        assert e.payload["quality"] == "thread_reference"
        assert e.payload["source_role"] == "user"

    def test_conjunction_openers_are_not_demoted(self) -> None:
        """D10 reuses only the bare-pronoun half of D7's shape. D7 also flags
        discourse-connective openers, and on the real corpus that half catches
        'So the next step is picking back up on toolbelt/retry.py' — a genuine
        handoff and the correct answer to a graded probe. Demoting it would
        break a passing probe to fix a failing one."""
        e = _event("THREAD_OPEN",
                   "So the next step is picking back up on toolbelt/retry.py — "
                   "implementing/finishing RetryPolicy and the retry helper.",
                   authority="user_stated")
        assert admit(e).action == "admit"

    def test_a_demonstrative_with_a_noun_head_is_not_a_reference(self) -> None:
        """'This migration is ...' names its subject and is self-contained —
        the same distinction D7's regex already draws."""
        e = _event("THREAD_OPEN", "This migration is still outstanding for next session.",
                   authority="user_stated")
        assert admit(e).action == "admit"

    def test_only_applies_to_user_stated_threads(self) -> None:
        """A thread already below the top tier can neither outrank nor
        supersede its antecedent, so there is nothing to correct."""
        e = _event("THREAD_OPEN", self._POINTER, authority="assistant_decided")
        assert admit(e).action == "admit"

    def test_only_applies_to_threads(self) -> None:
        e = _event("DECISION", self._POINTER, authority="user_stated")
        # DECISION is a statement type, so D7 handles it on its own terms —
        # what must not happen is D10 claiming it.
        assert admit(e).rule_id != "D10"


class TestUnverifiableLanguagesAreNotPenalised:
    """Grounding must not punish a file merely because we cannot parse its
    language. The inventory is built from the discovery walk, which globs only
    .py/.ts/.tsx/.js/.jsx — so a real internal/db/pool.go was being marked
    'unverified' and having its weight halved, pushing every component event in
    a Go/Rust/Java project off the budget-ranked block.

    'Cannot verify' is not 'does not exist', the same distinction already made
    for an empty inventory.
    """

    def test_unindexable_suffix_is_admitted(self) -> None:
        g = GroundingContext(
            frozenset({"src/known.py"}),
            verifiable_suffixes=frozenset({".py", ".ts"}),
        )
        v = admit(_event("COMPONENT_STATUS", "x", path="internal/db/pool.go"), g)
        assert v.action == "admit"

    def test_indexable_suffix_still_downgrades_when_unknown(self) -> None:
        g = GroundingContext(
            frozenset({"src/known.py"}),
            verifiable_suffixes=frozenset({".py", ".ts"}),
        )
        v = admit(_event("COMPONENT_STATUS", "x", path="src/ghost.py"), g)
        assert v.action == "downgrade"

    def test_indexable_and_known_is_admitted(self) -> None:
        g = GroundingContext(
            frozenset({"src/known.py"}),
            verifiable_suffixes=frozenset({".py"}),
        )
        assert admit(_event("COMPONENT_STATUS", "x", path="src/known.py"), g).action == "admit"

    def test_empty_suffix_set_verifies_everything(self) -> None:
        # Back-compat: callers that supply no suffix set keep the old behaviour.
        g = GroundingContext(frozenset({"src/known.py"}))
        assert admit(_event("COMPONENT_STATUS", "x", path="src/ghost.py"), g).action == "downgrade"
