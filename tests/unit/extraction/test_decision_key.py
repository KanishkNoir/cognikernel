"""J2.2 — decision-key derivation: ladder, normalization, honest coverage table."""
from __future__ import annotations

import sqlite3

import pytest

from cognikernel.extraction.decision_key import (
    backfill_keys,
    derive_decision_key,
    normalize_key,
)


class TestNormalization:
    def test_token_sort_makes_order_irrelevant(self) -> None:
        assert normalize_key("default alias") == normalize_key("alias default")

    def test_articles_and_stopwords_dropped(self) -> None:
        assert normalize_key("the JWT secret") == normalize_key("JWT Secret.")

    def test_conservative_singularize(self) -> None:
        assert normalize_key("attempts") == "attempt"
        assert normalize_key("counters") == "counter"
        # The naive trailing-s rule would corrupt exactly the identifiers keys
        # exist for — these must survive.
        assert normalize_key("alias") == "alias"
        assert normalize_key("redis") == "redis"
        assert normalize_key("address") == "address"

    def test_token_cap(self) -> None:
        k = normalize_key("alpha beta gamma delta epsilon zeta")
        assert len(k.split()) == 4

    def test_empty(self) -> None:
        assert normalize_key("") == ""
        assert normalize_key("the of an") == ""


class TestCandidateLadder:
    def test_subject_wins_first(self) -> None:
        key = derive_decision_key(
            {"subject": "rate limit counters", "description": "irrelevant"},
            "DECISION",
        )
        assert key == normalize_key("rate limit counters")

    def test_negation_triple_uses_object(self) -> None:
        key = derive_decision_key(
            {"triple": {"operator": "¬", "subject": "", "object": "LangChain"},
             "description": "Do not use LangChain in the hot path."},
            "CONSTRAINT_HARD",
        )
        assert key == "langchain"

    def test_decision_verb_topic(self) -> None:
        key = derive_decision_key(
            {"description": "switch from bcrypt to argon2id for password hashing"},
            "DECISION",
        )
        assert key == normalize_key("password hashing")

    def test_label_prefix(self) -> None:
        key = derive_decision_key(
            {"description": "Retry: 2 attempts per deployment, base=100 ms"},
            "DECISION",
        )
        assert key == "retry"

    def test_junk_labels_rejected(self) -> None:
        # Measured junk: sentence lead-ins must not become topic axes.
        for desc in (
            "New constraint: relay-default uses Opus as the normal model.",
            "Three changes: bump _MAX_ATTEMPTS and add jitter.",
            "Note: this only applies to streaming.",
        ):
            assert derive_decision_key({"description": desc}, "DECISION") == "", desc

    def test_update_directive_keys_via_decision_verb(self) -> None:
        # "Update:" is a junk LABEL, but the sentence itself carries a decision
        # verb + topic — the ladder's derive_subject path keys it (correctly).
        key = derive_decision_key(
            {"description": "Update: switch the default alias to claude-opus."},
            "DECISION",
        )
        assert key != ""

    def test_non_choice_family_unkeyed(self) -> None:
        assert derive_decision_key(
            {"description": "Retry: 2 attempts"}, "THREAD_OPEN"
        ) == ""

    def test_no_candidate_returns_empty(self) -> None:
        # Honesty rule: no head-noun guessing — undeducible topic = no key.
        assert derive_decision_key(
            {"description": "Callers get Sonnet under normal conditions."},
            "CONSTRAINT_HARD",
        ) == ""


class TestHonestCoverageTable:
    """Documents what v1 keys CAN and CANNOT do (measured on the gamma DB).

    Evolution chains whose links share no lexical surface do NOT co-key —
    that is the known F-C residual, tracked on the structural-supersession
    side, NOT a regression to fix by loosening these assertions.
    """

    def test_label_register_restatements_cokey(self) -> None:
        a = derive_decision_key(
            {"description": "Retry: 2 attempts per deployment, base=100 ms"}, "DECISION")
        b = derive_decision_key(
            {"description": "Retry: 3 attempts with full jitter"}, "DECISION")
        assert a == b == "retry"

    def test_cross_register_chain_does_not_cokey(self) -> None:
        a = derive_decision_key(
            {"description": "The default alias relay-default resolves to claude-opus-4-8."},
            "DECISION")
        b = derive_decision_key(
            {"description": 'Callers that send model: "relay-default" get Sonnet.'},
            "CONSTRAINT_HARD")
        assert a != b or a == ""  # documented miss, not a target


class TestBackfill:
    def _db(self) -> sqlite3.Connection:
        from cognikernel.storage.migrations import run_migrations
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        run_migrations(conn)
        return conn

    def test_backfill_idempotent(self) -> None:
        import hashlib
        from cognikernel.storage.events import Event, insert_event

        conn = self._db()
        e = Event(
            project_id="a" * 16, session_id="s1", event_type="DECISION",
            payload={"description": "Retry: 2 attempts per deployment"},
            content_hash=hashlib.sha256(b"x").hexdigest(),
        )
        insert_event(conn, e)
        conn.execute("UPDATE events SET decision_key = NULL")  # simulate pre-016
        conn.commit()
        assert backfill_keys(conn, "a" * 16) == 1
        assert backfill_keys(conn, "a" * 16) == 0  # '' written, never rescans
        row = conn.execute("SELECT decision_key FROM events").fetchone()
        assert row[0] == "retry"
