"""S5 T-502: read-only provenance queries behind `cognikernel why`.

A claim has to be findable after it was replaced, placed in session order rather
than shown as an opaque id, traced to the evidence it came from, and shown with
the chain of claims it replaced or was replaced by.
"""
from __future__ import annotations

from cognikernel.storage.events import Event, insert_event, set_superseded_by
from cognikernel.storage.evidence import store_evidence
from cognikernel.storage.provenance import (
    claim_provenance,
    find_claims,
    session_order,
    supersession_chain,
)

P = "proj1"


def _event(conn, session: str, desc: str, *, evidence_id: int | None = None,
           created_at: int = 1000, event_type: str = "DECISION", project: str = P,
           **payload) -> int:
    return insert_event(conn, Event(
        project_id=project,
        session_id=session,
        event_type=event_type,
        payload={"description": desc, **payload},
        content_hash=f"{project}:{desc}",
        evidence_id=evidence_id,
        created_at=created_at,
    ))


class TestSessionOrder:
    def test_sessions_are_numbered_by_first_capture_not_by_id(self, conn) -> None:
        store_evidence(conn, P, "late", "transcript", b"b", captured_at=2000)
        store_evidence(conn, P, "early", "transcript", b"a", captured_at=1000)

        order = session_order(conn, P)

        assert (order["early"].position, order["late"].position) == (1, 2)
        assert order["early"].total == order["late"].total == 2
        assert order["early"].first_seen == 1000

    def test_a_session_with_events_but_no_evidence_is_still_ordered(self, conn) -> None:
        store_evidence(conn, P, "s1", "transcript", b"a", captured_at=1000)
        _event(conn, "manual", "a manually recorded claim", created_at=500)

        order = session_order(conn, P)

        assert order["manual"].position == 1
        assert order["s1"].position == 2

    def test_other_projects_do_not_count(self, conn) -> None:
        store_evidence(conn, P, "s1", "transcript", b"a", captured_at=1000)
        store_evidence(conn, "other", "x", "transcript", b"zz", captured_at=10)

        assert set(session_order(conn, P)) == {"s1"}


class TestFindClaims:
    def test_by_id_with_or_without_hash(self, conn) -> None:
        eid = _event(conn, "s1", "Use SQLite for the store")

        assert [e.id for e in find_claims(conn, P, f"#{eid}")] == [eid]
        assert [e.id for e in find_claims(conn, P, str(eid))] == [eid]

    def test_an_id_from_another_project_is_not_found(self, conn) -> None:
        eid = _event(conn, "s1", "a foreign claim", project="other")

        assert find_claims(conn, P, f"#{eid}") == []

    def test_text_match_includes_superseded_history_live_claim_first(self, conn) -> None:
        old = _event(conn, "s1", "Retry policy: 4 attempts, no jitter", created_at=1000)
        new = _event(conn, "s2", "Retry policy: 6 attempts with full jitter", created_at=2000)
        set_superseded_by(conn, old, new, reason="cross_encoder")
        conn.commit()

        ids = [e.id for e in find_claims(conn, P, "retry policy")]

        assert set(ids) == {old, new}
        assert ids[0] == new

    def test_every_term_must_match(self, conn) -> None:
        _event(conn, "s1", "Retry policy: 4 attempts")

        assert find_claims(conn, P, "retry jitter") == []

    def test_limit_is_respected(self, conn) -> None:
        for n in range(5):
            _event(conn, "s1", f"cache decision number {n}")

        assert len(find_claims(conn, P, "cache decision", limit=2)) == 2


class TestSupersessionChain:
    def test_chain_runs_oldest_to_newest_from_any_member(self, conn) -> None:
        a = _event(conn, "s1", "limit is 2", created_at=1)
        b = _event(conn, "s2", "limit is 4", created_at=2)
        c = _event(conn, "s3", "limit is 8", created_at=3)
        set_superseded_by(conn, a, b, reason="decision_key")
        set_superseded_by(conn, b, c, reason="cross_encoder")
        conn.commit()

        for member in (a, b, c):
            assert [e.id for e in supersession_chain(conn, member)] == [a, b, c]

    def test_a_claim_with_no_history_is_its_own_chain(self, conn) -> None:
        a = _event(conn, "s1", "stands alone")

        assert [e.id for e in supersession_chain(conn, a)] == [a]

    def test_a_corrupt_cycle_does_not_loop(self, conn) -> None:
        """set_superseded_by refuses cycles, but a pre-guard store can hold one."""
        a = _event(conn, "s1", "claim x", created_at=1)
        b = _event(conn, "s1", "claim y", created_at=2)
        conn.execute("UPDATE events SET superseded_by = ? WHERE id = ?", (b, a))
        conn.execute("UPDATE events SET superseded_by = ? WHERE id = ?", (a, b))
        conn.commit()

        assert sorted(e.id for e in supersession_chain(conn, a)) == [a, b]


class TestClaimProvenance:
    def test_joins_the_evidence_the_claim_came_from(self, conn) -> None:
        ev = store_evidence(conn, P, "s1", "jsonl_transcript", b"{}",
                            source_path="/t/s1.jsonl", captured_at=1234)
        eid = _event(conn, "s1", "a sourced claim", evidence_id=ev)

        rows = claim_provenance(conn, eid)

        assert len(rows) == 1
        row = rows[0]
        assert (row["evidence_id"], row["session_id"], row["source_type"],
                row["captured_at"], row["extractor_version"]) == (
            ev, "s1", "jsonl_transcript", 1234, "cognikernel.v2")
        assert row["sentence_index"] is None  # the v2 extractor records no offsets

    def test_no_evidence_means_no_rows(self, conn) -> None:
        assert claim_provenance(conn, _event(conn, "s1", "unsourced claim")) == []
