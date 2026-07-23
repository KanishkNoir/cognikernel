"""Tests for description and rationale sanitization."""
import pytest

from cognikernel.extraction.sanitize import (
    is_question_description,
    sanitize_description,
    sanitize_rationale,
)


class TestBlockLevelStripping:
    def test_strips_markdown_heading_markers(self) -> None:
        assert sanitize_description("## Key decision") == "Key decision"

    def test_strips_bullet_markers(self) -> None:
        assert sanitize_description("- We chose SQLite") == "We chose SQLite"

    def test_drops_table_rows(self) -> None:
        text = "Intro line.\n| col1 | col2 |\n|------|------|\nClosing line."
        assert "col1" not in sanitize_description(text)
        assert "Intro line" in sanitize_description(text)

    def test_replaces_code_fence_with_hint(self) -> None:
        text = "Apply this:\n```python\nx = 1\n```"
        out = sanitize_description(text)
        assert "[code: python]" in out
        assert "x = 1" not in out

    def test_strips_role_prefix(self) -> None:
        assert sanitize_description("Assistant: We chose SQLite") == "We chose SQLite"

    def test_facts_are_not_truncated_mid_value(self) -> None:
        # v1 A-2: the operative tail of a fact (a number / env var / model id) must
        # survive. The old 120-char cap severed exactly this.
        fact = (
            "Every upstream call has a hard timeout: no configurable no-timeout "
            "option, the ceiling is gated at 120 s max and is configurable down "
            "but never up past 300 s."
        )
        out = sanitize_description(fact)
        assert "120 s" in out
        assert "300 s" in out
        assert not out.endswith("…")
        assert not out.endswith("...")


class TestInlineMarkdownStripping:
    """Inline markdown (bold/italic/code/link/strike) must be removed.

    Block-level patterns alone leave **bold**, *italic*, `code`, [text](url) intact
    in descriptions. These artifacts make stored events look like raw transcript
    noise when re-injected into a future LLM context.
    """

    def test_strips_bold_asterisks(self) -> None:
        assert sanitize_description("We **chose** SQLite") == "We chose SQLite"

    def test_strips_bold_underscores(self) -> None:
        assert sanitize_description("We __chose__ SQLite") == "We chose SQLite"

    def test_strips_italic_asterisks(self) -> None:
        assert sanitize_description("We *chose* SQLite") == "We chose SQLite"

    def test_strips_italic_underscores(self) -> None:
        # Underscore italic only between non-word chars so identifiers like
        # snake_case_var stay intact.
        assert sanitize_description("It was _the_ best choice") == "It was the best choice"

    def test_preserves_snake_case_identifiers(self) -> None:
        out = sanitize_description("Use snake_case_var for the field name")
        assert "snake_case_var" in out

    def test_strips_inline_code(self) -> None:
        assert sanitize_description("Set `WAL` mode on SQLite") == "Set WAL mode on SQLite"

    def test_strips_link_keeps_text(self) -> None:
        out = sanitize_description("See [the docs](https://example.com/x) for details")
        assert "the docs" in out
        assert "https" not in out
        assert "(" not in out

    def test_strips_strikethrough(self) -> None:
        assert sanitize_description("We ~~rejected~~ adopted Redis") == "We rejected adopted Redis"

    def test_strips_nested_inline_markdown(self) -> None:
        out = sanitize_description("**The `WAL` mode** is required")
        assert "WAL" in out
        assert "**" not in out
        assert "`" not in out

    def test_bold_with_colon_label(self) -> None:
        # The bold wrapper is stripped; structural-label filtering is the
        # windowing layer's job, not sanitize's.
        assert sanitize_description("**Key decisions:**") == "Key decisions:"


class TestRationale:
    def test_rationale_strips_same_inline_markdown(self) -> None:
        assert sanitize_rationale("This **matters** because X") == "This matters because X"

    def test_rationale_truncates(self) -> None:
        assert len(sanitize_rationale("b" * 500)) <= 120


class TestIsQuestionDescription:
    """`is_question_description` flags trailing-question-mark sentences that
    lack any declarative verb — the heuristic for "this is a stray user
    question, not a captured decision". Sentences with verbs like
    should/use/decided are treated as declarative even if punctuated as a
    question (rhetorical or tag question), so they aren't downgraded."""

    def test_pure_question_no_declarative_verb_is_flagged(self) -> None:
        assert is_question_description("What about Redis?")

    def test_question_with_declarative_verb_is_not_flagged(self) -> None:
        assert not is_question_description("Should we use SQLite?")

    def test_statement_is_not(self) -> None:
        assert not is_question_description("We chose SQLite")


# ── J5 data contracts ─────────────────────────────────────────────────────────

from cognikernel.extraction.sanitize import is_context_dependent_fragment


class TestJ5Sanitation:
    def test_blockquote_markers_stripped(self) -> None:
        # Measured artifact: agent quoting a constraint produced "> ..." lines.
        text = "> Do not alias to Opus.\n> Defaulting to Opus makes cost control impossible."
        out = sanitize_description(text)
        assert ">" not in out
        assert "Do not alias to Opus" in out

    def test_nested_blockquote_stripped(self) -> None:
        assert sanitize_description("> > deep quote fact") == "deep quote fact"

    def test_unpaired_bold_residue_removed(self) -> None:
        # Measured artifact: window split a bold span -> "impossible.**."
        out = sanitize_description("Defaulting to Opus makes cost control impossible.**.")
        assert "**" not in out
        assert out.startswith("Defaulting to Opus")

    def test_paired_bold_still_resolves(self) -> None:
        assert sanitize_description("**Do not alias to Opus.**") == "Do not alias to Opus."

    def test_single_star_identifier_preserved(self) -> None:
        assert "*args" in sanitize_description("pass *args through unchanged")


class TestContextDependentFragment:
    # Positives — the measured false-mint class.
    @pytest.mark.parametrize("desc", [
        "The 2x multiplier only matters if _MAX_ATTEMPTS were raised above 2.",
        "This flag only applies when the semantic cache is enabled.",
        "It only fires if the deadline has already passed.",
        "The cap would only be reached if every retry consumed the full window.",
    ])
    def test_fragment_positive(self, desc: str) -> None:
        assert is_context_dependent_fragment(desc)

    # Negatives — genuine constraints with "only"/"if" semantics must NOT demote.
    @pytest.mark.parametrize("desc", [
        "Only cache when temperature is explicitly 0 in the request.",
        "Backoff applies only to 5xx and 429 responses.",
        "Provider API keys come from environment variables only, never a database.",
        "If the first byte does not arrive within 10 s, cancel the upstream request.",
        "The raw key is shown to the issuer exactly once on creation.",
        "Do not offer per-request TTL override via an API header.",
    ])
    def test_fragment_negative(self, desc: str) -> None:
        assert not is_context_dependent_fragment(desc)


class TestTableScaffoldingStrip:
    """Windowed table debris (header+separator+id cells collapsed onto a content
    cell, ending in prose so _TABLE_ROW misses it) must be reduced to the fact."""

    def test_header_separator_id_prefix_stripped(self) -> None:
        line = ("| # | Invariant | |---|-----------| | O1 | Every status "
                "transition logs event_id and worker_id to a structured log.")
        out = sanitize_description(line)
        assert out.startswith("Every status transition logs")
        assert "|" not in out and "Invariant" not in out and "O1" not in out

    def test_trailing_table_cell_merged_into_prose(self) -> None:
        line = ("Never two separate commits. | | S6 | events.status = NOTIFIED "
                "is set only by the relay, never by the enrichment worker.")
        out = sanitize_description(line)
        assert "Never two separate commits" in out
        assert "events.status = NOTIFIED is set only by the relay" in out
        assert "S6" not in out and "|" not in out

    def test_prose_with_pipe_is_untouched(self) -> None:
        # No separator/empty-cell signal → not a table → must not be altered.
        for prose in (
            "The field type is str | None for optional values.",
            "Use A | B union syntax rather than Optional[A].",
        ):
            assert sanitize_description(prose) == prose.rstrip()

    def test_pure_scaffolding_line_drops_empty(self) -> None:
        assert sanitize_description("| # | Invariant | |---|---| | S1 |") == ""
