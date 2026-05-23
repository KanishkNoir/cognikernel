"""Tests for the sentence tokenizer."""
from memlora.extraction.tokenize import tokenize, Sentence


class TestCodeBlocks:
    def test_code_block_is_atomic(self) -> None:
        transcript = "Human: Some prose.\n\n```python\nfoo = 1\nbar = 2\n```\n\nMore prose."
        sentences = tokenize(transcript)
        code = [s for s in sentences if s.is_code_block]
        assert len(code) == 1
        assert "foo = 1" in code[0].text

    def test_code_block_not_split_internally(self) -> None:
        transcript = "Human: ```python\nx = 1. y = 2. z = 3.\n```"
        sentences = tokenize(transcript)
        # Should be one code-block sentence, not three split on periods
        code = [s for s in sentences if s.is_code_block]
        assert len(code) == 1

    def test_prose_before_code_extracted(self) -> None:
        transcript = "Human: We decided to use WAL mode.\n```sql\nPRAGMA journal_mode = WAL;\n```"
        sentences = tokenize(transcript)
        prose = [s for s in sentences if not s.is_code_block]
        assert any("WAL mode" in s.text for s in prose)

    def test_prose_after_code_extracted(self) -> None:
        transcript = "Human: ```python\npass\n```\nWe rolled back this approach."
        sentences = tokenize(transcript)
        prose = [s for s in sentences if not s.is_code_block]
        assert any("rolled back" in s.text for s in prose)


class TestBullets:
    def test_each_bullet_is_own_sentence(self) -> None:
        transcript = "Human: Constraints:\n- Cannot use Redis\n- Must not call network\n- No external APIs"
        sentences = tokenize(transcript)
        bullets = [s for s in sentences if s.text.startswith("-")]
        assert len(bullets) == 3

    def test_bullet_text_preserved(self) -> None:
        transcript = "Human: - We cannot use Redis"
        sentences = tokenize(transcript)
        assert any("cannot use Redis" in s.text for s in sentences)

    def test_asterisk_bullets_recognized(self) -> None:
        transcript = "Human: * First rule\n* Second rule"
        sentences = tokenize(transcript)
        bullets = [s for s in sentences if s.text.startswith("*")]
        assert len(bullets) == 2


class TestRoleDetection:
    def test_human_prefix_is_user(self) -> None:
        transcript = "Human: We decided to use SQLite."
        sentences = tokenize(transcript)
        assert all(s.role == "user" for s in sentences if s.text.strip())

    def test_assistant_prefix_is_assistant(self) -> None:
        transcript = "Assistant: Good choice for local-first tools."
        sentences = tokenize(transcript)
        assert all(s.role == "assistant" for s in sentences if s.text.strip())

    def test_claude_prefix_is_assistant(self) -> None:
        transcript = "Claude: That approach makes sense."
        sentences = tokenize(transcript)
        assert all(s.role == "assistant" for s in sentences if s.text.strip())

    def test_mixed_roles_correct(self) -> None:
        transcript = "Human: We decided to use SQLite.\n\nAssistant: Good choice."
        sentences = tokenize(transcript)
        user_sentences = [s for s in sentences if s.role == "user"]
        assistant_sentences = [s for s in sentences if s.role == "assistant"]
        assert len(user_sentences) >= 1
        assert len(assistant_sentences) >= 1

    def test_no_role_marker_defaults_to_user(self) -> None:
        transcript = "We decided to use SQLite for storage."
        sentences = tokenize(transcript)
        assert all(s.role == "user" for s in sentences if s.text.strip())


class TestSentenceIndices:
    def test_indices_are_sequential_from_zero(self) -> None:
        transcript = "Human: First sentence. Second sentence. Third sentence."
        sentences = tokenize(transcript)
        non_empty = [s for s in sentences if s.text.strip()]
        indices = [s.sentence_index for s in non_empty]
        assert indices == list(range(len(non_empty)))

    def test_offsets_increase_monotonically(self) -> None:
        transcript = "Human: First line.\n\nSecond line.\n\nThird line."
        sentences = tokenize(transcript)
        non_empty = [s for s in sentences if s.text.strip()]
        for i in range(1, len(non_empty)):
            assert non_empty[i].start_offset >= non_empty[i - 1].start_offset


class TestEdgeCases:
    def test_empty_transcript_returns_empty(self) -> None:
        assert tokenize("") == []

    def test_only_whitespace_returns_empty(self) -> None:
        assert tokenize("   \n\n   ") == []

    def test_transcript_with_only_code_block(self) -> None:
        transcript = "```python\npass\n```"
        sentences = tokenize(transcript)
        assert len(sentences) == 1
        assert sentences[0].is_code_block

    def test_multiple_code_blocks(self) -> None:
        transcript = "Human: First block:\n```\ncode1\n```\nSecond block:\n```\ncode2\n```"
        sentences = tokenize(transcript)
        code = [s for s in sentences if s.is_code_block]
        assert len(code) == 2


class TestSentenceSplitting:
    def test_period_followed_by_capital_splits(self) -> None:
        transcript = "Human: We decided to use SQLite. The reason is local-first design."
        sentences = tokenize(transcript)
        texts = [s.text for s in sentences if not s.is_code_block and s.text.strip()]
        assert len(texts) >= 2

    def test_abbreviation_does_not_split(self) -> None:
        # "e.g." should NOT trigger a sentence split
        transcript = "Human: Use a simple format e.g. JSON for storage."
        sentences = tokenize(transcript)
        prose = [s for s in sentences if not s.is_code_block and s.text.strip()]
        # Should be one sentence, not split at "e.g."
        assert len(prose) == 1
