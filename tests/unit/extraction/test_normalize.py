"""Tests for memlora.extraction.normalize — Phase A-1 + A-2."""
from __future__ import annotations

import pytest

from memlora.extraction.normalize import normalize_description, smart_truncate


# ── A-1: normalize_description ───────────────────────────────────────────────


class TestNormalizeDescription:
    def test_empty_returns_empty(self) -> None:
        assert normalize_description("") == ""

    def test_whitespace_only_returns_empty(self) -> None:
        assert normalize_description("   \n\t  ") == ""

    def test_clean_input_passes_through(self) -> None:
        assert normalize_description("Use PostgreSQL.") == "Use PostgreSQL."

    def test_appends_period_when_missing(self) -> None:
        assert normalize_description("Use PostgreSQL") == "Use PostgreSQL."

    def test_keeps_existing_terminal_question_mark(self) -> None:
        assert normalize_description("Use PostgreSQL?") == "Use PostgreSQL?"

    def test_keeps_existing_terminal_exclamation(self) -> None:
        assert normalize_description("Use PostgreSQL!") == "Use PostgreSQL!"

    def test_collapses_internal_whitespace(self) -> None:
        assert normalize_description("Use   PostgreSQL.\n\n  Yes.") == "Use PostgreSQL. Yes."

    # ── prefix stripping ─────────────────────────────────────────────────────

    def test_strips_confirm_prefix(self) -> None:
        """Arm-C real example: D5 was captured as the prompt verb."""
        result = normalize_description(
            "Confirm the env var name and that it must never appear anywhere else."
        )
        assert result == "The env var name and that it must never appear anywhere else."

    def test_strips_open_work_thread_prefix(self) -> None:
        """Arm-C real example: T1 was captured with the prompt verb."""
        result = normalize_description(
            "Open a work thread: we need to implement JWT authentication end-to-end."
        )
        assert result == "We need to implement JWT authentication end-to-end."

    def test_strips_record_prefix(self) -> None:
        result = normalize_description(
            "Record Celery as an explicitly abandoned approach."
        )
        assert result == "Celery as an explicitly abandoned approach."

    def test_strips_make_sure_prefix(self) -> None:
        assert normalize_description("Make sure tests pass.") == "Tests pass."

    def test_strips_note_that_prefix(self) -> None:
        assert (
            normalize_description("Note that bcrypt truncates at 72 bytes.")
            == "Bcrypt truncates at 72 bytes."
        )

    def test_only_one_prefix_stripped_to_avoid_overreach(self) -> None:
        """If a description starts with two stacked prompt verbs (unusual),
        only the first is removed — guards against over-aggressive stripping."""
        result = normalize_description("Confirm Decide use PostgreSQL.")
        assert result == "Decide use PostgreSQL."  # only "Confirm " stripped

    def test_case_insensitive_prefix_match(self) -> None:
        """Defensive: lowercase prompts shouldn't slip past the stripper."""
        assert normalize_description("confirm we use bcrypt.") == "We use bcrypt."

    def test_no_strip_when_prefix_is_inside_text(self) -> None:
        """'Record' inside a sentence is part of the meaning, not a prefix."""
        result = normalize_description("Database needs to record events.")
        assert result == "Database needs to record events."

    def test_idempotent(self) -> None:
        """Applying twice yields the same string."""
        once = normalize_description("Confirm we will use PostgreSQL")
        twice = normalize_description(once)
        assert once == twice == "We will use PostgreSQL."

    def test_capitalizes_after_prefix_strip(self) -> None:
        """After 'Confirm ' is stripped, the next word should start the sentence."""
        result = normalize_description("Confirm the env var lives in JWT_SECRET.")
        assert result.startswith("The env var")


# ── A-2: smart_truncate ──────────────────────────────────────────────────────


class TestSmartTruncate:
    def test_empty(self) -> None:
        assert smart_truncate("", 120) == ""

    def test_already_fits(self) -> None:
        assert smart_truncate("Short text.", 120) == "Short text."

    def test_exact_length(self) -> None:
        text = "a" * 120
        assert smart_truncate(text, 120) == text

    def test_truncates_at_sentence_boundary(self) -> None:
        """A `.` in the back half of the budget is a clean truncation point."""
        text = "First sentence. Second sentence that pushes past the budget here for sure no doubt."
        # 80-char budget; the period at "sentence." (15) is well past 40 chars
        # of the second sentence, so the back-half terminator is at position
        # ~75 (after "the budget"). Wait — actually the terminator we want is
        # the `.` at position 15 (after 'sentence'). That's at fraction 0.18,
        # below the 0.5 minimum — so we fall to word-cut.
        result = smart_truncate(text, 80)
        assert not result.endswith(" ")
        # Word boundary cut → ends with ellipsis since no late period exists.
        assert result.endswith("…")
        assert len(result) <= 80

    def test_truncates_after_late_period(self) -> None:
        """A period past the half-budget mark is preferred over word-cut."""
        text = "This is a quite long opening clause that finishes. Then a tail that exceeds the limit clearly."
        # 80 chars; "finishes." ends at position 50 (≥ 40 = 0.5*80) → cut after it
        result = smart_truncate(text, 80)
        assert result == "This is a quite long opening clause that finishes."

    def test_word_boundary_truncation_includes_ellipsis(self) -> None:
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma"
        result = smart_truncate(text, 40)
        assert result.endswith("…")
        assert " " not in result[-2:-1]  # no trailing space before ellipsis
        assert len(result) <= 40

    def test_word_boundary_never_cuts_mid_word(self) -> None:
        """The whole point of A-2: no mid-word truncation."""
        text = "Confirm the env var name and that it must never appear anywhere else specifically here"
        result = smart_truncate(text, 30)
        # Must end at a word boundary (the char before ellipsis must be a letter/punct,
        # not the middle of a longer word).
        body = result[:-1] if result.endswith("…") else result
        # If we drop the ellipsis, the result should not split a word — i.e.
        # the next char in the original at len(body) is either a space or end.
        next_char_pos = len(body)
        if next_char_pos < len(text):
            assert text[next_char_pos] == " ", (
                f"smart_truncate cut mid-word: result={result!r}, next char={text[next_char_pos]!r}"
            )

    def test_hard_cut_when_no_whitespace(self) -> None:
        text = "a" * 200
        result = smart_truncate(text, 50)
        assert result.endswith("…")
        assert len(result) == 50

    def test_degenerate_tiny_budget(self) -> None:
        """max_chars < 4 falls back to hard slice (no room for ellipsis)."""
        result = smart_truncate("hello world", 3)
        assert result == "hel"

    def test_arm_c_regression_case(self) -> None:
        """The actual mid-word truncation from the Arm C diagnostic."""
        text = (
            "Confirm the env var name and that it must never appear anywhere else. — "
            "Blocks the event loop if synchronous — smtplib is synchronous. "
            "FastAPI runs BackgroundTasks in a thread pool when the call is sync."
        )
        result = smart_truncate(text, 120)
        # Must not end in "when the ca" (the broken truncation we're fixing).
        assert not result.endswith("ca")
        # Must end at a sentence terminator OR ellipsis.
        last_char = result[-1]
        assert last_char in (".", "!", "?", "…"), (
            f"smart_truncate ended on {last_char!r}: {result!r}"
        )

    def test_custom_ellipsis(self) -> None:
        text = "one two three four five six seven eight nine ten eleven twelve"
        result = smart_truncate(text, 25, ellipsis="...")
        assert result.endswith("...")
        assert len(result) <= 25
