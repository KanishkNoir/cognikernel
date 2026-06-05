"""Tests for sliding-window context extraction."""
import pytest
from memlora.extraction.tokenize import Sentence, tokenize
from memlora.extraction.trie import TrieScanner, TrieMatch
from memlora.extraction.windowing import extract_window, extract_events_from_matches


def _make_sentence(text: str, idx: int, role: str = "user", code: bool = False) -> Sentence:
    return Sentence(
        text=text, start_offset=idx * 50, end_offset=idx * 50 + len(text),
        role=role, is_code_block=code, sentence_index=idx,
    )


class TestExtractWindow:
    def test_description_is_matching_sentence(self) -> None:
        sentences = [
            _make_sentence("Context before.", 0),
            _make_sentence("We decided to use SQLite.", 1),
            _make_sentence("Context after.", 2),
        ]
        desc, _ = extract_window(sentences, 1, "DECISION")
        assert "decided to use SQLite" in desc

    def test_rationale_excludes_description(self) -> None:
        sentences = [
            _make_sentence("Context before.", 0),
            _make_sentence("We decided to use SQLite.", 1),
            _make_sentence("Context after.", 2),
        ]
        _, rationale = extract_window(sentences, 1, "DECISION")
        assert "decided to use SQLite" not in rationale
        assert "Context" in rationale

    def test_default_window_includes_neighbors(self) -> None:
        sentences = [_make_sentence(f"Sentence {i}.", i) for i in range(7)]
        _, rationale = extract_window(sentences, 3, "DECISION")
        # Default: 2 before + 2 after = sentences 1, 2, 4, 5 in rationale
        assert "Sentence 1" in rationale or "Sentence 2" in rationale

    def test_first_sentence_no_before_context(self) -> None:
        sentences = [
            _make_sentence("We decided.", 0),
            _make_sentence("Because of X.", 1),
        ]
        desc, rationale = extract_window(sentences, 0, "DECISION")
        assert "We decided" in desc

    def test_last_sentence_no_after_context(self) -> None:
        sentences = [
            _make_sentence("Some context.", 0),
            _make_sentence("We decided.", 1),
        ]
        desc, _ = extract_window(sentences, 1, "DECISION")
        assert "We decided" in desc

    def test_code_block_after_match_included(self) -> None:
        sentences = [
            _make_sentence("We decided to apply these PRAGMAs.", 0),
            _make_sentence("```python\nPRAGMA WAL;\n```", 1, code=True),
        ]
        _, rationale = extract_window(sentences, 0, "DECISION")
        assert "PRAGMA WAL" in rationale

    def test_hard_constraint_bullet_narrow_window(self) -> None:
        sentences = [
            _make_sentence("Background context here.", 0),
            _make_sentence("- We cannot use Redis.", 1),
            _make_sentence("More context after.", 2),
        ]
        _, rationale = extract_window(sentences, 1, "CONSTRAINT_HARD")
        # Narrow window: rationale should be empty (bullet is self-contained)
        assert rationale.strip() == ""

    def test_backward_expansion_through_connective(self) -> None:
        sentences = [
            _make_sentence("We need local-first because", 0),
            _make_sentence("We decided to use SQLite.", 1),
        ]
        _, rationale = extract_window(sentences, 1, "DECISION")
        assert "local-first" in rationale


class TestExtractEventsFromMatches:
    def test_produces_events_for_each_match(self) -> None:
        # Decision must come from assistant turn; constraint is role-agnostic.
        transcript = "Assistant: We decided to use SQLite. We cannot use Redis."
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        events = extract_events_from_matches(sentences, matches, "p1", "s1")
        assert len(events) >= 2

    def test_event_has_description(self) -> None:
        transcript = "Human: We decided to use SQLite."
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        events = extract_events_from_matches(sentences, matches, "p1", "s1")
        for e in events:
            assert "description" in e.payload
            assert e.payload["description"]

    def test_event_has_source_role(self) -> None:
        transcript = "Human: We decided to use SQLite."
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        events = extract_events_from_matches(sentences, matches, "p1", "s1")
        assert all(e.payload.get("source_role") == "user" for e in events)

    def test_event_type_matches_signal(self) -> None:
        transcript = "Human: We cannot use Redis."
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        events = extract_events_from_matches(sentences, matches, "p1", "s1")
        assert any(e.event_type == "CONSTRAINT_HARD" for e in events)

    def test_content_hash_is_empty_before_hashing_stage(self) -> None:
        transcript = "Human: We decided to use SQLite."
        sentences = tokenize(transcript)
        matches = TrieScanner().scan(sentences, transcript)
        events = extract_events_from_matches(sentences, matches, "p1", "s1")
        # Content hash is computed by the hashing stage, not windowing
        assert all(e.content_hash == "" for e in events)

    def test_out_of_bounds_match_index_skipped(self) -> None:
        sentences = [_make_sentence("We decided.", 0)]
        bad_match = TrieMatch(sentence_index=999, matched_phrase="decided",
                              signal_type="DECISION", confidence=1.0)
        events = extract_events_from_matches(sentences, [bad_match], "p1", "s1")
        assert events == []


def _match(signal_type: str = "DECISION") -> TrieMatch:
    return TrieMatch(sentence_index=0, matched_phrase="decided",
                     signal_type=signal_type, confidence=0.9)


def _run_single(text: str, signal_type: str, role: str) -> list:
    sentence = _make_sentence(text, 0, role=role)
    return extract_events_from_matches([sentence], [_match(signal_type)], "p1", "s1")


_NOISY: list[tuple[str, str, str]] = [
    # Narration prefixes — assistant implementation narration
    ("Let me start by implementing the models.", "DECISION", "assistant"),
    ("I'll use SQLAlchemy for the ORM.", "DECISION", "assistant"),
    ("Now let me add the endpoints.", "DECISION", "assistant"),
    ("First, we set up the schema.", "DECISION", "assistant"),
    ("Let's begin with the database layer.", "DECISION", "assistant"),
    ("Going to implement CRUD now.", "DECISION", "assistant"),
    # Meta-references — CogniKernel / CLAUDE.md talk
    ("As required by CLAUDE.md, all responses must be JSON.", "CONSTRAINT_HARD", "assistant"),
    ("Per CLAUDE.md, endpoints live under /v1/.", "CONSTRAINT_SOFT", "assistant"),
    ("The get_session_state call returned prior decisions.", "DECISION", "assistant"),
    # User-prompt echo — verbatim task statements (imperative, assert no decision)
    ("Decide on the framework for the project.", "DECISION", "user"),
    ("Implement the CRUD endpoints and wire them in.", "DECISION", "user"),
    # Structural labels — markdown headings whose category name re-uses a signal phrase.
    # The label names a category, it isn't an instance of it. Storing the label as
    # the decision/abandonment is meta-discourse, not data.
    ("## Explicitly abandoned approaches", "APPROACH_ABANDONED_DO_NOT_RETRY", "assistant"),
    ("## What's ruled out", "APPROACH_ABANDONED_DO_NOT_RETRY", "assistant"),
    ("### Ruled out:", "APPROACH_ABANDONED_DO_NOT_RETRY", "assistant"),
    ("**Explicitly abandoned approaches:**", "APPROACH_ABANDONED_DO_NOT_RETRY", "assistant"),
    ("**What's ruled out:**", "APPROACH_ABANDONED_DO_NOT_RETRY", "assistant"),
    ("Explicitly abandoned approaches:", "APPROACH_ABANDONED_DO_NOT_RETRY", "assistant"),
    ("Things we won't use:", "APPROACH_ABANDONED_DO_NOT_RETRY", "assistant"),
]


_LABEL_BUT_SUBSTANTIVE: list[tuple[str, str, str]] = [
    # A heading prefix on a substantive long sentence is NOT a label — it's a
    # full sentence styled as a header. These must survive.
    (
        "## We switched to PostgreSQL because the WAL contention on SQLite blocked "
        "concurrent writers in our staging cluster.",
        "DECISION",
        "assistant",
    ),
    (
        "Switched to PostgreSQL after SQLite's WAL contention blocked our concurrent "
        "writers in staging.",
        "DECISION",
        "assistant",
    ),
]

_GOOD: list[tuple[str, str, str]] = [
    ("We chose SQLite over PostgreSQL for local-first deployment.", "DECISION", "assistant"),
    ("FastAPI was selected for its automatic OpenAPI generation.", "DECISION", "assistant"),
    ("SQLAlchemy's ORM was chosen to avoid raw SQL migration drift.", "DECISION", "assistant"),
    # F4: a DECISION the USER states is first-class (highest authority), not noise.
    ("We decided to use FastAPI.", "DECISION", "user"),
    ("We're switching from bcrypt to argon2id for password hashing.", "DECISION", "user"),
    # Constraints survive regardless of source role
    ("The API must not use Redis.", "CONSTRAINT_HARD", "assistant"),
    ("All endpoints require explicit versioning under /v1/.", "CONSTRAINT_HARD", "user"),
]


class TestNarrationFilter:
    """Narration, meta-talk, and user-prompt-echo events must be filtered out."""

    @pytest.mark.parametrize("text,signal_type,role", _NOISY)
    def test_noisy_descriptions_are_dropped(self, text: str, signal_type: str, role: str) -> None:
        events = _run_single(text, signal_type, role)
        descriptions = [e.payload["description"] for e in events]
        assert events == [], f"Expected drop but event survived: {descriptions}"

    @pytest.mark.parametrize("text,signal_type,role", _GOOD)
    def test_good_descriptions_survive(self, text: str, signal_type: str, role: str) -> None:
        events = _run_single(text, signal_type, role)
        assert len(events) == 1, f"Expected event to survive but got {len(events)} events"

    @pytest.mark.parametrize("text,signal_type,role", _LABEL_BUT_SUBSTANTIVE)
    def test_substantive_sentences_with_heading_marker_survive(
        self, text: str, signal_type: str, role: str
    ) -> None:
        events = _run_single(text, signal_type, role)
        assert len(events) == 1, (
            f"Expected substantive sentence to survive label filter, got {len(events)} events"
        )

    def test_f4_user_decision_kept_with_user_authority(self) -> None:
        """F4: a user-turn DECISION is captured with authority=user_stated.

        Regression for Benchmark_CK: the user's 'switching from bcrypt to argon2id'
        decision was previously blanket-dropped, so supersession had nothing to link.
        """
        events = _run_single(
            "We're switching from bcrypt to argon2id for password hashing.",
            "DECISION", "user",
        )
        assert len(events) == 1
        assert events[0].event_type == "DECISION"
        assert events[0].payload["authority"] == "user_stated"

    def test_f4_user_prompt_echo_still_dropped(self) -> None:
        """F4 must not regress the anti-echo guard — imperative prompts are not decisions."""
        assert _run_single("Implement the CRUD endpoints.", "DECISION", "user") == []
        assert _run_single("Decide on the framework.", "DECISION", "user") == []
