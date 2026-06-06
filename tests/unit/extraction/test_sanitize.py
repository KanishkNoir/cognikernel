"""Tests for description and rationale sanitization."""
import pytest

from memlora.extraction.sanitize import (
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
