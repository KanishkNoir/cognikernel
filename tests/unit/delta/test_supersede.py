"""Tests for memlora.delta.supersede."""
from __future__ import annotations

import json
import sqlite3
from typing import Any
from unittest.mock import patch

import pytest

from memlora.delta.supersede import (
    JACCARD_THRESHOLD,
    LEVENSHTEIN_THRESHOLD,
    apply_supersession,
    derive_subject,
    descriptions_overlap,
    detect_supersession,
    events_overlap,
    find_superseded,
    jaccard_similarity,
    levenshtein_normalized,
    normalize_for_overlap,
    subject_supersedes,
    supersedes,
)
from memlora.embedding.model import EMBEDDING_MODEL_VERSION, embed_text, is_available
from memlora.embedding.store import upsert_embedding
from memlora.storage.events import Event


class TestFindSuperseded:
    """Hybrid finder: semantic + temporal + authority gates over lexical OR."""

    def test_lexical_match_with_gates(self, conn: sqlite3.Connection) -> None:
        old_id = seed_event(
            conn, content_hash="h_old", created_at=1000,
            payload={"description": "Use SQLite for local storage", "authority": "assistant_decided"},
        )
        new = make_event(
            content_hash="h_new", created_at=2000,
            payload={"description": "Use SQLite for local storage", "authority": "assistant_decided"},
        )
        assert old_id in find_superseded(conn, new)

    def test_temporal_gate_newer_not_superseded(self, conn: sqlite3.Connection) -> None:
        seed_event(
            conn, content_hash="h_future", created_at=5000,
            payload={"description": "Use SQLite for local storage", "authority": "assistant_decided"},
        )
        new = make_event(
            content_hash="h_now", created_at=2000,
            payload={"description": "Use SQLite for local storage", "authority": "assistant_decided"},
        )
        assert find_superseded(conn, new) == []

    def test_authority_gate_low_does_not_supersede_high(self, conn: sqlite3.Connection) -> None:
        seed_event(
            conn, content_hash="h_user", created_at=1000,
            payload={"description": "Use SQLite for local storage", "authority": "user_stated"},
        )
        new = make_event(
            content_hash="h_inferred", created_at=2000,
            payload={"description": "Use SQLite for local storage", "authority": "inferred_from_code"},
        )
        assert find_superseded(conn, new) == []

    def test_same_evidence_not_superseded(self, conn: sqlite3.Connection) -> None:
        """E2: a match within the SAME transcript is a restatement, not evolution."""
        seed_evidence(conn, 1)
        seed_event(
            conn, content_hash="h_a", created_at=1000, evidence_id=1,
            payload={"description": "Use SQLite for local storage", "authority": "assistant_decided"},
        )
        new = make_event(
            content_hash="h_b", created_at=2000, evidence_id=1,
            payload={"description": "Use SQLite for local storage", "authority": "assistant_decided"},
        )
        assert find_superseded(conn, new) == []

    def test_cross_evidence_superseded(self, conn: sqlite3.Connection) -> None:
        """E2: the same decision re-asserted in a DIFFERENT transcript supersedes."""
        seed_evidence(conn, 1)
        seed_evidence(conn, 2)
        old_id = seed_event(
            conn, content_hash="h_a", created_at=1000, evidence_id=1,
            payload={"description": "Use SQLite for local storage", "authority": "assistant_decided"},
        )
        new = make_event(
            content_hash="h_b", created_at=2000, evidence_id=2,
            payload={"description": "Use SQLite for local storage", "authority": "assistant_decided"},
        )
        assert old_id in find_superseded(conn, new)

    # ── gates are the baseline even with the semantic axis OFF ────────────────
    # use_embeddings=False is the config.embedding_enabled=False merge path: the
    # temporal/authority/provenance gates must still apply, and no embedding
    # model may be loaded.

    def test_lexical_match_with_gates_embeddings_off(self, conn: sqlite3.Connection) -> None:
        old_id = seed_event(
            conn, content_hash="h_old", created_at=1000,
            payload={"description": "Use SQLite for local storage", "authority": "assistant_decided"},
        )
        new = make_event(
            content_hash="h_new", created_at=2000,
            payload={"description": "Use SQLite for local storage", "authority": "assistant_decided"},
        )
        assert old_id in find_superseded(conn, new, use_embeddings=False)

    def test_embeddings_off_does_not_load_model(self, conn: sqlite3.Connection) -> None:
        """The default path must never touch the embedding model (P5/P6)."""
        old_id = seed_event(
            conn, content_hash="h_old", created_at=1000,
            payload={"description": "Use SQLite for local storage"},
        )
        new = make_event(
            content_hash="h_new", created_at=2000,
            payload={"description": "Use SQLite for local storage"},
        )
        with patch("memlora.embedding.model.embed_text") as mock_embed:
            result = find_superseded(conn, new, use_embeddings=False)
            mock_embed.assert_not_called()
        assert old_id in result

    def test_temporal_gate_holds_with_embeddings_off(self, conn: sqlite3.Connection) -> None:
        seed_event(
            conn, content_hash="h_future", created_at=5000,
            payload={"description": "Use SQLite for local storage"},
        )
        new = make_event(
            content_hash="h_now", created_at=2000,
            payload={"description": "Use SQLite for local storage"},
        )
        assert find_superseded(conn, new, use_embeddings=False) == []

    def test_authority_gate_holds_with_embeddings_off(self, conn: sqlite3.Connection) -> None:
        seed_event(
            conn, content_hash="h_user", created_at=1000,
            payload={"description": "Use SQLite for local storage", "authority": "user_stated"},
        )
        new = make_event(
            content_hash="h_inferred", created_at=2000,
            payload={"description": "Use SQLite for local storage", "authority": "inferred_from_code"},
        )
        assert find_superseded(conn, new, use_embeddings=False) == []

    def test_provenance_gate_holds_with_embeddings_off(self, conn: sqlite3.Connection) -> None:
        seed_evidence(conn, 1)
        seed_event(
            conn, content_hash="h_a", created_at=1000, evidence_id=1,
            payload={"description": "Use SQLite for local storage"},
        )
        new = make_event(
            content_hash="h_b", created_at=2000, evidence_id=1,
            payload={"description": "Use SQLite for local storage"},
        )
        assert find_superseded(conn, new, use_embeddings=False) == []

    @pytest.mark.skipif(not is_available(), reason="embedding model not installed")
    def test_semantic_supersedes_paraphrase(self, conn: sqlite3.Connection) -> None:
        bcrypt_desc = "We will use bcrypt for password hashing."
        argon_desc = "Hash user passwords with argon2id going forward."
        # Lexically disjoint (Jaccard 0) — only the semantic axis catches it.
        assert not descriptions_overlap(bcrypt_desc, argon_desc)
        old_id = seed_event(
            conn, content_hash="h_bcrypt", created_at=1000,
            payload={"description": bcrypt_desc, "authority": "assistant_decided"},
        )
        upsert_embedding(conn, old_id, embed_text(bcrypt_desc), EMBEDDING_MODEL_VERSION)
        conn.commit()
        new = make_event(
            content_hash="h_argon", created_at=2000,
            payload={"description": argon_desc, "authority": "assistant_decided"},
        )
        assert old_id in find_superseded(conn, new)

    # ── F1: cross-type subject supersession ───────────────────────────────────

    def test_cross_type_decision_supersedes_constraint_same_subject(
        self, conn: sqlite3.Connection
    ) -> None:
        """F1 + Benchmark_CK regression: a DECISION supersedes a same-subject
        CONSTRAINT_SOFT. bcrypt was stored as CONSTRAINT_SOFT and the argon2id
        switch as a DECISION, so the same-type-only finder never compared them.
        Cross-type matching (shared derived subject 'password hashing' + the
        Jaccard floor) now links them — with the semantic axis OFF (default path).

        Uses the ACTUAL Benchmark_CK user phrasings (topic-before-verb in S1; a
        'from X to Y for TOPIC' switch in S2) so this proves the real end-to-end fix,
        not just a clean canonical phrasing."""
        seed_evidence(conn, 1)
        seed_evidence(conn, 2)
        old_id = seed_event(
            conn, content_hash="h_bcrypt", created_at=1000, evidence_id=1,
            event_type="CONSTRAINT_SOFT",
            payload={"description": "For password hashing, we will use bcrypt.",
                     "authority": "user_stated"},
        )
        new = make_event(
            content_hash="h_argon", created_at=2000, evidence_id=2,
            event_type="DECISION",
            payload={"description": "After more research, we're switching from bcrypt "
                                    "to argon2id for password hashing.",
                     "authority": "user_stated"},
        )
        assert old_id in find_superseded(conn, new, use_embeddings=False)

    def test_cross_type_requires_subject_match_unrelated_survives(
        self, conn: sqlite3.Connection
    ) -> None:
        """F1 precision: cross-type needs a shared subject. An unrelated constraint
        (no derivable subject) is never superseded by a hashing decision, even
        though both are in the choice family."""
        seed_evidence(conn, 1)
        seed_evidence(conn, 2)
        survivor = seed_event(
            conn, content_hash="h_uuid", created_at=1000, evidence_id=1,
            event_type="CONSTRAINT_HARD",
            payload={"description": "Use UUID primary keys.", "authority": "user_stated"},
        )
        new = make_event(
            content_hash="h_argon", created_at=2000, evidence_id=2,
            event_type="DECISION",
            payload={"description": "we will use argon2id for password hashing instead of bcrypt.",
                     "authority": "user_stated"},
        )
        assert survivor not in find_superseded(conn, new, use_embeddings=False)

    def test_cross_type_authority_gate_still_holds(
        self, conn: sqlite3.Connection
    ) -> None:
        """F1 must not bypass the authority gate: a low-authority cross-type event
        cannot supersede a higher-authority one."""
        seed_evidence(conn, 1)
        seed_evidence(conn, 2)
        survivor = seed_event(
            conn, content_hash="h_user", created_at=1000, evidence_id=1,
            event_type="CONSTRAINT_SOFT",
            payload={"description": "We will use bcrypt for password hashing.",
                     "authority": "user_stated"},
        )
        new = make_event(
            content_hash="h_inferred", created_at=2000, evidence_id=2,
            event_type="DECISION",
            payload={"description": "we will use argon2id for password hashing instead of bcrypt.",
                     "authority": "inferred_from_code"},
        )
        assert survivor not in find_superseded(conn, new, use_embeddings=False)


class TestDeriveSubject:
    """derive_subject extracts the *topic* a decision is about (not the choice)."""

    def test_choice_for_topic(self) -> None:
        assert derive_subject("We will use bcrypt for password hashing.") == "password hashing"

    def test_switch_keeps_topic_not_choice(self) -> None:
        assert derive_subject(
            "we will use argon2id for password hashing instead of bcrypt."
        ) == "password hashing"

    def test_switch_from_to_for_topic_real_phrasing(self) -> None:
        # F4b: the actual Benchmark_CK S2 phrasing. "to argon2id" must NOT win over
        # "for password hashing" — the purpose topic is the stable subject.
        assert derive_subject(
            "After more research, we're switching from bcrypt to argon2id for password hashing."
        ) == "password hashing"

    def test_topic_before_verb(self) -> None:
        # F4b: the actual Benchmark_CK S1 phrasing — topic precedes the verb.
        assert derive_subject("For password hashing, we will use bcrypt.") == "password hashing"

    def test_prohibition_subject_is_the_rejected_thing(self) -> None:
        assert derive_subject("Do not use Celery, we will never revisit it.") == "celery"

    def test_no_subject_returns_empty(self) -> None:
        assert derive_subject("This sentence has no decision verb pattern.") == ""

    def test_unrelated_decisions_no_subject(self) -> None:
        assert derive_subject("Composite PK on (project_id, user_id) instead of a surrogate UUID.") == ""
        assert derive_subject("Use UUID primary keys.") == ""
        assert derive_subject("Scrypt / PBKDF2 fine, but bcrypt is what FastAPI docs default to.") == ""


class TestSubjectSupersedes:
    def test_same_topic_different_choice_supersedes(self) -> None:
        # "Redis for caching" → "Memcached for caching": same topic, different tool,
        # Jaccard = 1/3 = 0.33 — below the 0.6 descriptions_overlap threshold but
        # above the 0.3 subject_supersedes Jaccard floor. Subject-keying catches it.
        a = "We will use Redis for caching."
        b = "We will use Memcached for caching."
        assert not descriptions_overlap(a, b)
        assert subject_supersedes(a, b)
        assert supersedes(a, b)

    def test_no_subject_on_either_does_not_supersede(self) -> None:
        a = "Composite PK on (project_id, user_id) instead of a surrogate UUID."
        b = "we will use argon2id for password hashing instead of bcrypt."
        assert not subject_supersedes(a, b)

    def test_same_subject_but_jaccard_too_low_does_not_supersede(self) -> None:
        # Both about "password hashing" but genuinely different decisions —
        # the Jaccard floor keeps them distinct.
        a = "We will use argon2id for password hashing."
        b = "Set the memory cost to 64 megabytes for password hashing."
        assert not supersedes(a, b)

    def test_mobc_false_positive_protection(self) -> None:
        """None of the MOBC false-positive candidates (Composite PK, UUID PK,
        the Scrypt ecosystem note) should be superseded by a hashing correction."""
        correction = "we will use argon2id for password hashing instead of bcrypt."
        assert not supersedes(correction, "Composite PK on (project_id, user_id) instead of a surrogate UUID.")
        assert not supersedes(correction, "Use UUID primary keys.")
        assert not supersedes(correction, "Scrypt / PBKDF2 fine but bcrypt is FastAPI default.")


def _naive_overlap(a: str, b: str) -> bool:
    """The pre-optimization overlap rule, for parity checks."""
    return (
        jaccard_similarity(a, b) >= JACCARD_THRESHOLD
        or levenshtein_normalized(a, b) <= LEVENSHTEIN_THRESHOLD
    )


class TestDescriptionsOverlapPruneIsExact:
    """The length-bound prune must never change the overlap result."""

    _PAIRS = [
        ("Use SQLite for local storage", "Use SQLite for local storage"),      # identical
        ("Use SQLite for local storage", "Use SQLite for the local store"),    # near
        ("Use SQLite for local storage", "Adopt Postgres in production"),      # different
        ("Never log secrets", "Never ever log any secrets to disk anywhere"),  # length-disparate
        ("auth", "authentication subsystem rewrite end to end"),               # short vs long
        ("", "Use SQLite"),                                                    # empty
        ("Redis Streams for the queue", "Use Redis Streams for the queue"),    # near, prefixed
    ]

    @pytest.mark.parametrize("a,b", _PAIRS)
    def test_matches_naive(self, a: str, b: str) -> None:
        assert descriptions_overlap(a, b) == _naive_overlap(a, b)
        assert descriptions_overlap(b, a) == _naive_overlap(b, a)  # symmetric


# ── helpers ───────────────────────────────────────────────────────────────────

def make_event(**overrides: Any) -> Event:
    defaults: dict[str, Any] = {
        "project_id": "proj1",
        "session_id": "sess1",
        "event_type": "DECISION",
        "payload": {"description": "Use SQLite for local storage"},
        "content_hash": "hash_a",
        "weight": 1.0,
    }
    defaults.update(overrides)
    return Event(**defaults)


def seed_event(conn: sqlite3.Connection, **overrides: Any) -> int:
    """Insert a raw event row and return its id."""
    e = make_event(**overrides)
    payload_json = json.dumps(e.payload, sort_keys=True, separators=(",", ":"))
    cursor = conn.execute(
        """
        INSERT INTO events
            (project_id, session_id, created_at, event_type,
             payload, content_hash, weight, mention_count, evidence_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (e.project_id, e.session_id, e.created_at, e.event_type,
         payload_json, e.content_hash, e.weight, e.mention_count, e.evidence_id),
    )
    conn.commit()
    return cursor.lastrowid  # type: ignore[return-value]


def seed_evidence(conn: sqlite3.Connection, evidence_id: int, project_id: str = "proj1") -> None:
    """Insert a minimal raw_evidence row so events.evidence_id FK is satisfiable."""
    conn.execute(
        """
        INSERT INTO raw_evidence
            (id, project_id, session_id, source_type, captured_at,
             content_sha256, content_encoding, content_blob,
             original_size_bytes, stored_size_bytes)
        VALUES (?, ?, 'sess1', 'transcript', ?, ?, 'zlib', ?, 1, 1)
        """,
        (evidence_id, project_id, evidence_id, f"sha{evidence_id}", b"x"),
    )
    conn.commit()


# ── normalize_for_overlap ─────────────────────────────────────────────────────

class TestNormalizeForOverlap:
    def test_lowercases(self) -> None:
        tokens = normalize_for_overlap("SQLite Database")
        assert "sqlite" in tokens
        assert "database" in tokens

    def test_strips_punctuation(self) -> None:
        tokens = normalize_for_overlap("use SQLite, always!")
        assert "sqlite" in tokens
        assert "always" in tokens
        assert all("," not in t for t in tokens)

    def test_removes_stopwords(self) -> None:
        tokens = normalize_for_overlap("we use the database")
        assert "we" not in tokens
        assert "use" not in tokens
        assert "the" not in tokens

    def test_removes_short_tokens(self) -> None:
        tokens = normalize_for_overlap("a db at home")
        assert "a" not in tokens
        assert "db" not in tokens
        assert "at" not in tokens

    def test_returns_set(self) -> None:
        result = normalize_for_overlap("same same word word")
        assert isinstance(result, set)
        assert "same" in result
        assert "word" in result

    def test_empty_string_returns_empty_set(self) -> None:
        assert normalize_for_overlap("") == set()


# ── jaccard_similarity ────────────────────────────────────────────────────────

class TestJaccardSimilarity:
    def test_identical_texts_return_one(self) -> None:
        s = "Use SQLite for persistent storage"
        assert jaccard_similarity(s, s) == pytest.approx(1.0)

    def test_totally_different_returns_low(self) -> None:
        score = jaccard_similarity("apple orange banana", "database schema migration")
        assert score < 0.2

    def test_partial_overlap(self) -> None:
        score = jaccard_similarity("SQLite database storage", "SQLite database cache")
        assert 0.4 < score < 0.9

    def test_empty_a_returns_zero(self) -> None:
        assert jaccard_similarity("", "something here") == pytest.approx(0.0)

    def test_empty_b_returns_zero(self) -> None:
        assert jaccard_similarity("something here", "") == pytest.approx(0.0)

    def test_both_empty_returns_zero(self) -> None:
        assert jaccard_similarity("", "") == pytest.approx(0.0)

    def test_threshold_constant_is_0_6(self) -> None:
        assert JACCARD_THRESHOLD == pytest.approx(0.6)


# ── levenshtein_normalized ────────────────────────────────────────────────────

class TestLevenshteinNormalized:
    def test_identical_returns_zero(self) -> None:
        assert levenshtein_normalized("hello world", "hello world") == pytest.approx(0.0)

    def test_empty_a_returns_one(self) -> None:
        assert levenshtein_normalized("", "something") == pytest.approx(1.0)

    def test_empty_b_returns_one(self) -> None:
        assert levenshtein_normalized("something", "") == pytest.approx(1.0)

    def test_both_empty_returns_zero(self) -> None:
        assert levenshtein_normalized("", "") == pytest.approx(0.0)

    def test_single_char_difference(self) -> None:
        score = levenshtein_normalized("cat", "bat")
        assert score == pytest.approx(1 / 3)

    def test_case_insensitive(self) -> None:
        assert levenshtein_normalized("Hello", "hello") == pytest.approx(0.0)

    def test_strips_whitespace(self) -> None:
        assert levenshtein_normalized("  hello  ", "hello") == pytest.approx(0.0)

    def test_similar_sentences_low_score(self) -> None:
        score = levenshtein_normalized(
            "always use SQLite for local storage",
            "always use SQLite for persistent storage",
        )
        assert score < 0.5

    def test_very_different_sentences_high_score(self) -> None:
        score = levenshtein_normalized("abcdefghij", "zyxwvutsrq")
        assert score > 0.5

    def test_threshold_constant_is_0_15(self) -> None:
        assert LEVENSHTEIN_THRESHOLD == pytest.approx(0.15)


# ── events_overlap ────────────────────────────────────────────────────────────

class TestEventsOverlap:
    def test_different_types_returns_false(self) -> None:
        a = make_event(event_type="DECISION", content_hash="h1",
                       payload={"description": "Use SQLite for all storage"})
        b = make_event(event_type="CONSTRAINT_HARD", content_hash="h2",
                       payload={"description": "Use SQLite for all storage"})
        assert events_overlap(a, b) is False

    def test_identical_descriptions_returns_true(self) -> None:
        desc = "Never store secrets in plain text configuration files"
        a = make_event(content_hash="h1", payload={"description": desc})
        b = make_event(content_hash="h2", payload={"description": desc})
        assert events_overlap(a, b) is True

    def test_high_jaccard_returns_true(self) -> None:
        a = make_event(content_hash="h1",
                       payload={"description": "SQLite persistent local storage database"})
        b = make_event(content_hash="h2",
                       payload={"description": "SQLite persistent local storage backend"})
        assert events_overlap(a, b) is True

    def test_low_levenshtein_returns_true(self) -> None:
        a = make_event(content_hash="h1",
                       payload={"description": "Always use SQLite for data persistence"})
        b = make_event(content_hash="h2",
                       payload={"description": "Always use SQLite for data persistency"})
        assert events_overlap(a, b) is True

    def test_clearly_different_returns_false(self) -> None:
        a = make_event(content_hash="h1",
                       payload={"description": "Use SQLite for storage"})
        b = make_event(content_hash="h2",
                       payload={"description": "Enforce rate limiting on all endpoints"})
        assert events_overlap(a, b) is False

    def test_one_empty_description_returns_false(self) -> None:
        a = make_event(content_hash="h1", payload={"description": "Use SQLite for storage"})
        b = make_event(content_hash="h2", payload={})
        assert events_overlap(a, b) is False


# ── detect_supersession ───────────────────────────────────────────────────────

class TestDetectSupersession:
    def test_non_supersession_type_returns_empty(self, conn: sqlite3.Connection) -> None:
        seed_event(conn, event_type="COMPONENT_STATUS", content_hash="old1",
                   payload={"description": "Use SQLite for storage", "path": "app.py", "status": "stable"})
        new = make_event(event_type="COMPONENT_STATUS", content_hash="new1",
                         payload={"description": "Use SQLite for storage", "path": "app.py", "status": "stable"})
        assert detect_supersession(conn, new) == []

    def test_finds_overlapping_decision(self, conn: sqlite3.Connection) -> None:
        old_id = seed_event(conn, event_type="DECISION", content_hash="old1",
                            payload={"description": "Use SQLite for persistent local storage"})
        new = make_event(event_type="DECISION", content_hash="new1",
                         payload={"description": "Use SQLite for persistent local data storage"})
        result = detect_supersession(conn, new)
        assert old_id in result

    def test_finds_overlapping_constraint_hard(self, conn: sqlite3.Connection) -> None:
        old_id = seed_event(conn, event_type="CONSTRAINT_HARD", content_hash="old1",
                            payload={"description": "Never expose API keys in source code"})
        new = make_event(event_type="CONSTRAINT_HARD", content_hash="new1",
                         payload={"description": "Never expose API secrets keys source code"})
        result = detect_supersession(conn, new)
        assert old_id in result

    def test_ignores_archived_events(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO events
                (project_id, session_id, created_at, event_type,
                 payload, content_hash, weight, mention_count, archived)
            VALUES ('proj1', 'sess1', 0, 'DECISION',
                    '{"description":"Use SQLite for persistent local storage"}',
                    'old1', 1.0, 1, 1)
            """
        )
        conn.commit()
        new = make_event(event_type="DECISION", content_hash="new1",
                         payload={"description": "Use SQLite for persistent local storage"})
        assert detect_supersession(conn, new) == []

    def test_ignores_already_superseded_events(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO events
                (project_id, session_id, created_at, event_type,
                 payload, content_hash, weight, mention_count, superseded_by)
            VALUES ('proj1', 'sess1', 0, 'DECISION',
                    '{"description":"Use SQLite for persistent local storage"}',
                    'old1', 1.0, 1, 99)
            """
        )
        conn.commit()
        new = make_event(event_type="DECISION", content_hash="new1",
                         payload={"description": "Use SQLite for persistent local storage"})
        assert detect_supersession(conn, new) == []

    def test_ignores_same_content_hash(self, conn: sqlite3.Connection) -> None:
        desc = "Use SQLite for persistent local storage"
        seed_event(conn, event_type="DECISION", content_hash="same_hash",
                   payload={"description": desc})
        new = make_event(event_type="DECISION", content_hash="same_hash",
                         payload={"description": desc})
        assert detect_supersession(conn, new) == []

    def test_no_overlap_returns_empty(self, conn: sqlite3.Connection) -> None:
        seed_event(conn, event_type="DECISION", content_hash="old1",
                   payload={"description": "Use Postgres for cloud deployments"})
        new = make_event(event_type="DECISION", content_hash="new1",
                         payload={"description": "Enforce rate limiting on external API calls"})
        assert detect_supersession(conn, new) == []

    def test_ignores_different_project(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO events
                (project_id, session_id, created_at, event_type,
                 payload, content_hash, weight, mention_count)
            VALUES ('other_proj', 'sess1', 0, 'DECISION',
                    '{"description":"Use SQLite for persistent local storage"}',
                    'old1', 1.0, 1)
            """
        )
        conn.commit()
        new = make_event(project_id="proj1", event_type="DECISION", content_hash="new1",
                         payload={"description": "Use SQLite for persistent local storage"})
        assert detect_supersession(conn, new) == []


# ── apply_supersession ────────────────────────────────────────────────────────

class TestApplySupersession:
    def test_marks_events_superseded(self, conn: sqlite3.Connection) -> None:
        old_id = seed_event(conn, content_hash="old1")
        new_id = seed_event(conn, content_hash="new1")
        apply_supersession(conn, new_id, [old_id])
        conn.commit()
        row = conn.execute("SELECT superseded_by FROM events WHERE id = ?", (old_id,)).fetchone()
        assert row["superseded_by"] == new_id

    def test_returns_count(self, conn: sqlite3.Connection) -> None:
        ids = [seed_event(conn, content_hash=f"old{i}") for i in range(3)]
        new_id = seed_event(conn, content_hash="new1")
        count = apply_supersession(conn, new_id, ids)
        assert count == 3

    def test_empty_list_returns_zero(self, conn: sqlite3.Connection) -> None:
        assert apply_supersession(conn, 99, []) == 0

    def test_multiple_events_all_marked(self, conn: sqlite3.Connection) -> None:
        ids = [seed_event(conn, content_hash=f"h{i}") for i in range(4)]
        new_id = seed_event(conn, content_hash="new1")
        apply_supersession(conn, new_id, ids)
        conn.commit()
        for old_id in ids:
            row = conn.execute("SELECT superseded_by FROM events WHERE id = ?", (old_id,)).fetchone()
            assert row["superseded_by"] == new_id
