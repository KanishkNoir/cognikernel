"""Tests for the shared defect detectors (spec §2)."""
from cognikernel.quality.detectors import DetectorHit, detect_subject_less


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
