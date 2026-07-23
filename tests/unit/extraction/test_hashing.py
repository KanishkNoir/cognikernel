"""Tests for content hashing and normalization."""
from cognikernel.extraction.hashing import compute_content_hash, normalize_for_hash


class TestNormalizeForHash:
    def test_lowercased(self) -> None:
        assert normalize_for_hash("REDIS") == normalize_for_hash("redis")

    def test_whitespace_collapsed(self) -> None:
        assert normalize_for_hash("use  SQLite") == normalize_for_hash("use SQLite")

    def test_leading_trailing_stripped(self) -> None:
        assert normalize_for_hash("  use SQLite  ") == normalize_for_hash("use SQLite")

    def test_punctuation_removed(self) -> None:
        assert normalize_for_hash("use SQLite!") == normalize_for_hash("use SQLite")
        assert normalize_for_hash("use SQLite.") == normalize_for_hash("use SQLite")

    def test_cannot_normalized_to_cant(self) -> None:
        assert normalize_for_hash("cannot use Redis") == normalize_for_hash("can't use Redis")

    def test_do_not_normalized_to_dont(self) -> None:
        assert normalize_for_hash("do not use Redis") == normalize_for_hash("don't use Redis")

    def test_we_will_normalized_to_well(self) -> None:
        assert normalize_for_hash("we will use SQLite") == normalize_for_hash("we'll use SQLite")

    def test_different_text_different_result(self) -> None:
        assert normalize_for_hash("use SQLite") != normalize_for_hash("use Redis")


class TestComputeContentHash:
    def test_same_inputs_same_hash(self) -> None:
        h1 = compute_content_hash("DECISION", "We decided to use SQLite.")
        h2 = compute_content_hash("DECISION", "We decided to use SQLite.")
        assert h1 == h2

    def test_different_description_different_hash(self) -> None:
        h1 = compute_content_hash("DECISION", "We decided to use SQLite.")
        h2 = compute_content_hash("DECISION", "We decided to use Redis.")
        assert h1 != h2

    def test_different_event_type_different_hash(self) -> None:
        h1 = compute_content_hash("DECISION", "We cannot use Redis.")
        h2 = compute_content_hash("CONSTRAINT_HARD", "We cannot use Redis.")
        assert h1 != h2

    def test_cannot_and_cant_same_hash(self) -> None:
        h1 = compute_content_hash("CONSTRAINT_HARD", "We cannot use Redis.")
        h2 = compute_content_hash("CONSTRAINT_HARD", "We can't use Redis.")
        assert h1 == h2

    def test_hash_is_hex_string(self) -> None:
        h = compute_content_hash("DECISION", "We decided to use SQLite.")
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_is_64_chars(self) -> None:
        h = compute_content_hash("DECISION", "We decided to use SQLite.")
        assert len(h) == 64

    def test_case_insensitive_hash(self) -> None:
        h1 = compute_content_hash("DECISION", "We DECIDED to use SQLite.")
        h2 = compute_content_hash("DECISION", "We decided to use SQLite.")
        assert h1 == h2

    def test_punctuation_invariant(self) -> None:
        h1 = compute_content_hash("DECISION", "We decided to use SQLite!")
        h2 = compute_content_hash("DECISION", "We decided to use SQLite.")
        assert h1 == h2
