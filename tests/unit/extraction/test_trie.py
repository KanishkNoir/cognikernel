"""Tests for the Aho-Corasick trie scanner."""
from memlora.extraction.tokenize import tokenize
from memlora.extraction.trie import TrieScanner, TrieMatch, get_scanner, _word_boundary


class TestSignalDetection:
    def test_detects_decided(self) -> None:
        transcript = "Human: We decided to use SQLite."
        sentences = tokenize(transcript)
        scanner = TrieScanner()
        matches = scanner.scan(sentences, transcript)
        assert any(m.matched_phrase == "decided" for m in matches)

    def test_detects_cannot(self) -> None:
        transcript = "Human: We cannot use Redis in production."
        sentences = tokenize(transcript)
        scanner = TrieScanner()
        matches = scanner.scan(sentences, transcript)
        assert any(m.matched_phrase == "cannot" for m in matches)

    def test_detects_multi_word_phrase(self) -> None:
        transcript = "Human: We went with SQLite for its simplicity."
        sentences = tokenize(transcript)
        scanner = TrieScanner()
        matches = scanner.scan(sentences, transcript)
        assert any(m.matched_phrase == "went with" for m in matches)

    def test_detects_reverted(self) -> None:
        transcript = "Human: We reverted the Redis integration."
        sentences = tokenize(transcript)
        scanner = TrieScanner()
        matches = scanner.scan(sentences, transcript)
        assert any(m.matched_phrase == "reverted" for m in matches)

    def test_detects_todo(self) -> None:
        transcript = "Human: Todo: add index on weight column."
        sentences = tokenize(transcript)
        scanner = TrieScanner()
        matches = scanner.scan(sentences, transcript)
        assert any(m.matched_phrase == "todo" for m in matches)


class TestCaseInsensitivity:
    def test_uppercase_match(self) -> None:
        transcript = "Human: We DECIDED to use WAL mode."
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        assert any(m.matched_phrase == "decided" for m in matches)

    def test_mixed_case_match(self) -> None:
        transcript = "Human: We Cannot use Redis."
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        assert any(m.matched_phrase == "cannot" for m in matches)


class TestWordBoundaries:
    def test_no_match_inside_word(self) -> None:
        # "decided" inside "undecided" must NOT match
        transcript = "Human: The choice was undecided at that point."
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        assert not any(m.matched_phrase == "decided" for m in matches)

    def test_signal_at_start_of_sentence(self) -> None:
        transcript = "Human: Decided: we'll use SQLite."
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        assert any(m.matched_phrase == "decided" for m in matches)

    def test_word_boundary_helper_start(self) -> None:
        assert _word_boundary("decided to use", 0, 7) is True

    def test_word_boundary_helper_mid(self) -> None:
        # "decided" inside "undecided" — not a boundary at start
        assert _word_boundary("undecided", 2, 9) is False

    def test_word_boundary_helper_end(self) -> None:
        assert _word_boundary("we decided", 3, 10) is True


class TestCodeBlockSkipping:
    def test_signal_inside_code_block_skipped(self) -> None:
        transcript = "Human: ```python\n# we decided to use redis\nredis_client = Redis()\n```"
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        # "decided" inside code should be skipped
        assert not any(m.matched_phrase == "decided" for m in matches)


class TestMatchMetadata:
    def test_match_has_correct_signal_type(self) -> None:
        transcript = "Human: We cannot use Redis."
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        cannot_matches = [m for m in matches if m.matched_phrase == "cannot"]
        assert cannot_matches
        assert cannot_matches[0].signal_type == "CONSTRAINT_HARD"

    def test_match_has_confidence(self) -> None:
        transcript = "Human: We decided to use SQLite."
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        assert all(0.0 < m.confidence <= 1.0 for m in matches)

    def test_match_sentence_index_valid(self) -> None:
        transcript = "Human: We decided to use SQLite. We cannot use Redis."
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        for m in matches:
            assert 0 <= m.sentence_index < len(sentences)


class TestSingleton:
    def test_get_scanner_returns_same_instance(self) -> None:
        s1 = get_scanner()
        s2 = get_scanner()
        assert s1 is s2

    def test_singleton_produces_same_results(self) -> None:
        transcript = "Human: We decided to use SQLite."
        sentences = tokenize(transcript)
        r1 = get_scanner().scan(sentences, transcript)
        r2 = get_scanner().scan(sentences, transcript)
        assert [(m.matched_phrase, m.sentence_index) for m in r1] == [
            (m.matched_phrase, m.sentence_index) for m in r2
        ]
