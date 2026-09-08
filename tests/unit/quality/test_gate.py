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
        e = _event("THREAD_OPEN", "This is the active work item for the next session.",
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
