"""J1.1 — FTS5 lexical index: availability, sync, liveness filtering, sanitization."""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from memlora.storage.events import Event, insert_event
from memlora.storage.fts import (
    bm25_search,
    build_match_query,
    ensure_fts,
    fts_available,
    fts_enabled,
    prohibition_search,
)
from memlora.storage.migrations import run_migrations

PID = "a" * 16


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    run_migrations(c)
    yield c
    c.close()


def _mk(desc: str, etype: str = "DECISION", subject: str = "", pid: str = PID) -> Event:
    return Event(
        project_id=pid,
        session_id="s1",
        event_type=etype,
        payload={"description": desc, "subject": subject},
        content_hash=hashlib.sha256(desc.encode()).hexdigest(),
    )


class TestAvailabilityAndBootstrap:
    def test_fts_available_on_test_build(self, conn) -> None:
        # The dev/CI build must have FTS5; production degrades gracefully.
        assert fts_available(conn)

    def test_run_migrations_enables_fts(self, conn) -> None:
        assert fts_enabled(conn)

    def test_ensure_fts_idempotent(self, conn) -> None:
        assert ensure_fts(conn)
        assert ensure_fts(conn)  # second call is the fast path

    def test_backfill_indexes_preexisting_events(self) -> None:
        c = sqlite3.connect(":memory:")
        c.row_factory = sqlite3.Row
        run_migrations(c)
        # Simulate an event that predates the index: drop index + flag, insert, recreate.
        c.execute("DROP TRIGGER trg_events_fts_ai")
        c.execute("DROP TRIGGER trg_events_fts_au")
        c.execute("DROP TABLE events_fts")
        c.execute("DELETE FROM meta WHERE key='fts_enabled'")
        c.commit()
        insert_event(c, _mk("retry uses exponential backoff base 100ms"))
        assert ensure_fts(c)
        hits = bm25_search(c, PID, "exponential backoff")
        assert len(hits) == 1
        c.close()


class TestTriggerSync:
    def test_insert_is_indexed(self, conn) -> None:
        insert_event(conn, _mk("the default alias relay-default resolves to claude-opus-4-8"))
        hits = bm25_search(conn, PID, "relay-default alias")
        assert len(hits) == 1
        assert "relay-default" in hits[0]["description"]

    def test_identifier_tokens_survive(self, conn) -> None:
        insert_event(conn, _mk("raise _MAX_ATTEMPTS only via config"))
        assert bm25_search(conn, PID, "_MAX_ATTEMPTS")  # underscore token intact

    def test_payload_update_reindexes(self, conn) -> None:
        eid = insert_event(conn, _mk("obsolete phrasing entirely"))
        conn.execute(
            "UPDATE events SET payload = json_set(payload, '$.description', 'fresh replacement text') "
            "WHERE id = ?",
            (eid,),
        )
        conn.commit()
        assert not bm25_search(conn, PID, "obsolete phrasing")
        assert bm25_search(conn, PID, "fresh replacement text")


class TestLivenessFiltering:
    def test_archived_excluded(self, conn) -> None:
        eid = insert_event(conn, _mk("archived fact about websockets"))
        conn.execute("UPDATE events SET archived = 1 WHERE id = ?", (eid,))
        conn.commit()
        assert not bm25_search(conn, PID, "websockets")

    def test_superseded_excluded(self, conn) -> None:
        old = insert_event(conn, _mk("cache TTL is 600 seconds"))
        new = insert_event(conn, _mk("cache TTL is 3600 seconds"))
        conn.execute("UPDATE events SET superseded_by = ? WHERE id = ?", (new, old))
        conn.commit()
        hits = bm25_search(conn, PID, "cache TTL seconds")
        assert [h["id"] for h in hits] == [new]

    def test_other_project_excluded(self, conn) -> None:
        insert_event(conn, _mk("fact in this project"))
        insert_event(conn, _mk("fact in other project", pid="b" * 16))
        hits = bm25_search(conn, PID, "fact project")
        assert len(hits) == 1


class TestMatchSanitization:
    def test_apostrophes_and_parens_safe(self, conn) -> None:
        insert_event(conn, _mk("we can't use websockets (latency)"))
        # Raw text full of FTS5 syntax hazards must not raise.
        assert bm25_search(conn, PID, 'can\'t we use (websockets) OR latency"')

    def test_hyphenated_identifier_quoted(self) -> None:
        q = build_match_query("what does relay-default resolve to?")
        assert '"relay-default"' in q

    def test_stopword_only_query_empty(self) -> None:
        assert build_match_query("the and of it") == ""

    def test_empty_query_no_hits(self, conn) -> None:
        assert bm25_search(conn, PID, "") == []

    def test_token_cap(self) -> None:
        q = build_match_query(" ".join(f"token{i}" for i in range(30)))
        assert q.count(" OR ") == 11  # capped at 12 tokens


class TestProhibitionSearch:
    """K1 — type-restricted retrieval: only graveyard + hard constraints surface."""

    def _seed(self, conn) -> None:
        insert_event(conn, _mk(
            "do not use in-process rate-limit counters; use Redis",
            etype="APPROACH_ABANDONED_DO_NOT_RETRY", subject="rate limiting"))
        insert_event(conn, _mk(
            "money columns must be integer cents, never float",
            etype="CONSTRAINT_HARD", subject="money type"))
        # Same topic, ordinary decision — must NOT be in the prohibition pool.
        insert_event(conn, _mk(
            "we considered a Redis-backed rate-limit token bucket",
            etype="DECISION", subject="rate limiting"))

    def test_only_binding_types_returned(self, conn) -> None:
        self._seed(conn)
        hits = prohibition_search(conn, PID, "adding an in-process rate-limit counter")
        assert hits, "the graveyard prohibition should surface"
        assert all(h["event_type"] in {"APPROACH_ABANDONED_DO_NOT_RETRY", "CONSTRAINT_HARD"}
                   for h in hits)
        assert all(h["event_type"] != "DECISION" for h in hits)

    def test_hard_constraint_surfaces_for_contradicting_diff(self, conn) -> None:
        self._seed(conn)
        hits = prohibition_search(conn, PID, "balance = float(amount) / 100  # money column")
        assert any("integer cents" in h["description"] for h in hits)

    def test_carries_authority_and_rationale_fields(self, conn) -> None:
        insert_event(conn, _mk(
            "never store secrets in the repo",
            etype="CONSTRAINT_HARD", subject="secrets"))
        hits = prohibition_search(conn, PID, "secrets stored in repo")
        assert hits and "authority" in hits[0] and "rationale" in hits[0]

    def test_empty_and_stopword_query(self, conn) -> None:
        self._seed(conn)
        assert prohibition_search(conn, PID, "") == []
        assert prohibition_search(conn, PID, "the and of it") == []

    def test_no_prohibition_when_only_plain_decisions(self, conn) -> None:
        insert_event(conn, _mk("we use postgres for the primary store",
                               etype="DECISION", subject="database"))
        assert prohibition_search(conn, PID, "postgres primary store") == []
