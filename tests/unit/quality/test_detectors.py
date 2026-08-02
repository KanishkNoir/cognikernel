"""Tests for the shared defect detectors (spec §2)."""
from cognikernel.quality.detectors import (
    DetectorHit,
    detect_boilerplate,
    detect_junk_constraint,
    detect_subject_less,
)


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

    def test_allows_ordinary_statement(self) -> None:
        assert detect_boilerplate("The worker retries twice before dead-lettering.") is None
