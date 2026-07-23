"""Tests for the sentence tokenizer."""
from cognikernel.extraction.tokenize import tokenize, Sentence


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
        # v1 A-1: each bullet is still its own sentence, now flagged list_item
        # with the leading marker stripped from the text.
        transcript = "Human: Constraints:\n- Cannot use Redis\n- Must not call network\n- No external APIs"
        sentences = tokenize(transcript)
        bullets = [s for s in sentences if s.list_item]
        assert len(bullets) == 3
        # marker stripped — text no longer starts with "- "
        assert all(not s.text.startswith(("-", "*", "•")) for s in bullets)

    def test_bullet_marker_stripped_text_preserved(self) -> None:
        # v1 A-1: the marker is gone but the fact text is intact.
        transcript = "Human: - We cannot use Redis"
        sentences = tokenize(transcript)
        item = next(s for s in sentences if s.list_item)
        assert item.text == "We cannot use Redis"

    def test_numbered_list_marker_stripped(self) -> None:
        # v1 A-1: numbered ordinals are stripped (no "4. " polluting the fact).
        transcript = "Human:\n1. PostgreSQL only\n2. UUID primary keys"
        sentences = tokenize(transcript)
        items = [s for s in sentences if s.list_item]
        assert len(items) == 2
        assert items[0].text == "PostgreSQL only"
        assert all(s.list_group_id == items[0].list_group_id for s in items)
        assert items[0].list_group_id != -1

    def test_asterisk_bullets_recognized(self) -> None:
        transcript = "Human: * First rule\n* Second rule"
        sentences = tokenize(transcript)
        bullets = [s for s in sentences if s.list_item]
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


class TestLabelValueLines:
    """I7e: 'Label: value' lines are self-contained facts — never merged into
    the prose accumulator (GAMMA_CK_TEST: max-attempts/recovery-window/open-
    threshold all merged into one blob and died in the salience head)."""

    def test_consecutive_label_lines_stay_separate(self):
        from cognikernel.extraction.tokenize import tokenize
        text = ("Assistant:\n"
                "Open threshold: 3 consecutive connection errors\n"
                "Recovery window: 30 s (configurable per provider)\n"
                "Max attempts: 2 (original + 1 failover).\n")
        sents = [s.text for s in tokenize(text)]
        assert any(s.startswith("Open threshold: 3") for s in sents)
        assert any(s.startswith("Recovery window: 30") for s in sents)
        assert any(s.startswith("Max attempts: 2") for s in sents)
        # No merged mega-sentence.
        assert not any("Open threshold" in s and "Recovery window" in s for s in sents)

    def test_label_line_with_trailing_sentence_splits(self):
        from cognikernel.extraction.tokenize import tokenize
        text = ("Assistant:\n"
                "Max attempts: 2 (one failover). Do not retry the same deployment.\n")
        sents = [s.text for s in tokenize(text)]
        assert any(s.startswith("Max attempts: 2") for s in sents)
        assert any(s.startswith("Do not retry") for s in sents)

    def test_normal_prose_with_midline_colon_not_split(self):
        from cognikernel.extraction.tokenize import tokenize
        text = ("Assistant:\n"
                "We considered the following options: rotate keys or shard the pool,\n"
                "and decided the latter is simpler to operate.\n")
        sents = [s.text for s in tokenize(text)]
        # "We considered the following options: rotate..." starts capitalized with
        # a colon beyond 48 chars? No — label must be <=48 chars and line is a
        # label-line only if it matches the tight pattern. This wraps two lines
        # of one sentence; it must remain joined.
        assert any("rotate keys" in s and "simpler to operate" in s for s in sents)
