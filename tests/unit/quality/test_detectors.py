"""Tests for the shared defect detectors (spec §2)."""
from cognikernel.quality.detectors import (
    DetectorHit,
    detect_boilerplate,
    detect_junk_constraint,
    detect_subject_less,
    normalized_key,
)


class TestIsMemoryMeta:
    """R1 — the assistant narrating CogniKernel's own memory, not a project fact.

    Measured 2026-09-12 on every tagged claim in the local stores (162): 30 were
    real project facts. They matched because a project discusses CogniKernel
    as a design subject, or uses "from memory" / "graveyard" as ordinary words.
    Sentences here are synthetic, one per measured class.
    """

    def test_flags_cognikernel_acting_as_the_tool(self) -> None:
        from cognikernel.quality.detectors import is_memory_meta

        assert is_memory_meta("CogniKernel's Stop hook will persist the updated rationale.")
        assert is_memory_meta("The CogniKernel MCP server is holding the database lock.")
        assert is_memory_meta("There is no source code yet, only CogniKernel scaffolding.")

    def test_flags_session_context_and_harness_narration(self) -> None:
        from cognikernel.quality.detectors import is_memory_meta

        assert is_memory_meta("The session context flagged a hard constraint before I touched the cache.")
        assert is_memory_meta("Resume directly — do not acknowledge the summary.")

    def test_flags_memory_framing(self) -> None:
        from cognikernel.quality.detectors import is_memory_meta

        assert is_memory_meta("I'll pull the established invariants from memory before touching code.")
        assert is_memory_meta("Key decisions from memory: the upsert returns is_new.")
        assert is_memory_meta("This will be recorded in the graveyard at session end.")

    def test_a_project_that_discusses_cognikernel_is_not_meta(self) -> None:
        from cognikernel.quality.detectors import is_memory_meta

        assert not is_memory_meta("Keep our own store rather than vendoring CogniKernel's event model.")
        assert not is_memory_meta("The memory systems compared were Mem0, Zep, Letta and CogniKernel.")
        assert not is_memory_meta("CogniKernel has no world-time concept, so the resolver is needed anyway.")
        assert not is_memory_meta("Add the CogniKernel memory store directory to .gitignore.")

    def test_from_memory_as_an_ordinary_phrase_is_not_meta(self) -> None:
        from cognikernel.quality.detectors import is_memory_meta

        assert not is_memory_meta("On a cache hit, re-emit the stored response as synthetic chunks from memory.")
        assert not is_memory_meta("The connect timeout (5 s, from memory) belongs inside the iterator.")

    def test_the_projects_own_graveyard_is_not_meta(self) -> None:
        from cognikernel.quality.detectors import is_memory_meta

        assert not is_memory_meta("Per-tenant credentials stay in the graveyard until multi-tenancy is scoped.")
        assert not is_memory_meta("LangChain is rejected for the request path and goes in the graveyard.")


class TestDetectSubjectLess:
    """D7 — statements whose subject only exists in unstated context.

    Positive cases are verbatim from the 163-store sweep (spec §0.1).
    """

    def test_flags_bare_pronoun_with_adverb(self) -> None:
        hit = detect_subject_less("This only guarantees the event is captured exactly once.")
        assert hit is not None
        assert hit.rule_id == "D7"

    def test_flags_bare_pronoun_with_modal(self) -> None:
        assert detect_subject_less("It must not be able to take down the pipeline.") is not None

    def test_flags_demonstrative_with_verb(self) -> None:
        assert detect_subject_less("This also connects back to the erasure angle.") is not None

    def test_flags_conjunction_opener(self) -> None:
        assert detect_subject_less("And make the raw payload hard to log by accident.") is not None

    def test_allows_statement_with_explicit_subject(self) -> None:
        assert detect_subject_less("The dispatcher must not take down the pipeline.") is None

    def test_allows_demonstrative_with_noun_head(self) -> None:
        # "This migration" names its referent — recoverable, not subject-less.
        assert detect_subject_less("This migration is idempotent.") is None

    def test_allows_imperative_constraint(self) -> None:
        assert detect_subject_less("Do not cache in Redis.") is None

    def test_allows_empty(self) -> None:
        assert detect_subject_less("") is None

    def test_hit_carries_note(self) -> None:
        hit = detect_subject_less("It must not be able to take down the pipeline.")
        assert isinstance(hit, DetectorHit)
        assert hit.note


class TestDetectJunkConstraint:
    """D2 — non-propositional content stored as a constraint."""

    def test_flags_box_drawing_table(self) -> None:
        text = "┌─────────┬────────┐ │ Layer │ Choice │"
        hit = detect_junk_constraint(text, "CONSTRAINT_HARD")
        assert hit is not None
        assert hit.rule_id == "D2"

    def test_flags_interrogative_stored_as_constraint(self) -> None:
        text = "For the queue between worker and dispatcher, should we just bring in Celery?"
        assert detect_junk_constraint(text, "CONSTRAINT_HARD") is not None

    def test_flags_mostly_non_alphabetic(self) -> None:
        assert detect_junk_constraint("=== 12 | 34 || 56 -- 78 ===", "CONSTRAINT_SOFT") is not None

    # ── regression pins: these are WELL-FORMED and were false positives in the
    # exploratory sweep (spec §0.1). They must never be flagged again.

    def test_allows_leading_imperative_do(self) -> None:
        text = "Do the slow work outside any transaction (LLM call, HTTP dispatch)."
        assert detect_junk_constraint(text, "CONSTRAINT_HARD") is None

    def test_allows_leading_imperative_do_not(self) -> None:
        text = "Do not cache in Redis — that adds a network hop and defeats the point."
        assert detect_junk_constraint(text, "CONSTRAINT_HARD") is None

    def test_allows_colon_list_constraint(self) -> None:
        text = "What must be redacted: message content, system prompt text, tool definitions."
        assert detect_junk_constraint(text, "CONSTRAINT_HARD") is None

    def test_ignores_non_constraint_types(self) -> None:
        # D2 is scoped to constraints; a question captured as a DECISION is out of scope here.
        assert detect_junk_constraint("Should we use Celery?", "DECISION") is None


class TestDetectBoilerplate:
    """D4 — harness/compaction chatter captured as project memory."""

    def test_flags_compaction_resume_instruction(self) -> None:
        text = "Pick up the last task as if the break never happened."
        hit = detect_boilerplate(text)
        assert hit is not None
        assert hit.rule_id == "D4"

    def test_flags_transcript_pointer(self) -> None:
        assert detect_boilerplate("read the full transcript at: C:/Users/x/a.jsonl") is not None

    def test_flags_cognikernel_own_injection_header(self) -> None:
        assert detect_boilerplate("## Session context [auto-generated — do not edit]") is not None

    def test_flags_greenfield_self_echo(self) -> None:
        text = "Confirmed greenfield — no prior decisions stored. Here's a concrete proposal."
        hit = detect_boilerplate(text)
        assert hit is not None
        assert hit.rule_id == "D4"

    def test_flags_no_prior_decisions_stored_alone(self) -> None:
        assert detect_boilerplate("No prior decisions stored for this component.") is not None

    def test_allows_ordinary_statement(self) -> None:
        assert detect_boilerplate("The worker retries twice before dead-lettering.") is None


class TestNormalizedKey:
    """D5 — the key that makes 'same fact, different type' detectable."""

    def test_collapses_case_and_punctuation(self) -> None:
        a = normalized_key("Record Celery as an explicitly abandoned approach!")
        b = normalized_key("record celery as an explicitly abandoned approach")
        assert a == b

    def test_collapses_whitespace(self) -> None:
        assert normalized_key("a   b\n c") == normalized_key("a b c")

    def test_distinguishes_different_statements(self) -> None:
        assert normalized_key("use postgres") != normalized_key("use redis")

    def test_empty_is_empty(self) -> None:
        assert normalized_key("   ") == ""
