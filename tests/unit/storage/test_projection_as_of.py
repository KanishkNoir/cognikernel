"""S5 T-503 (G3): the belief set as of a past time, from the 021 transition columns.

A claim created by time T is:
  live      never superseded/archived, or that transition happened after T
  ended     superseded/archived at a RECORDED time at or before T
  unknown   superseded/archived with NO recorded time (every pre-021 row)

Measured on 179 real stores: all 483 superseded and 58 archived rows lack a
transition time. Treating those as live (or as ended) would make replay
confidently wrong, so they are reported separately, never guessed.
"""
from __future__ import annotations

from cognikernel.storage.events import Event, insert_event
from cognikernel.storage.projections import build_projection, rebuild_projection
from cognikernel.storage.provenance import claims_as_of

P = "proj1"


def _claim(conn, name: str, created_at: int, event_type: str = "DECISION", session: str = "s1") -> int:
    return insert_event(conn, Event(
        project_id=P, session_id=session, event_type=event_type,
        payload={"description": f"claim {name}"}, content_hash=f"hash-{name}", created_at=created_at,
    ))


def _timeline(conn) -> dict[str, int]:
    """a live throughout · b superseded by c at 3000 · d archived at 4000 ·
    e superseded by f with no recorded time (a pre-021 row)."""
    ids = {
        "a": _claim(conn, "a", 1000),
        "e": _claim(conn, "e", 1500),
        "f": _claim(conn, "f", 1600),
        "b": _claim(conn, "b", 2000),
        "d": _claim(conn, "d", 2500),
        "c": _claim(conn, "c", 3000, session="s2"),
    }
    conn.execute("UPDATE events SET superseded_by = ?, superseded_at = 3000, supersede_reason = 'cross_encoder' "
                 "WHERE id = ?", (ids["c"], ids["b"]))
    conn.execute("UPDATE events SET archived = 1, archived_at = 4000 WHERE id = ?", (ids["d"],))
    conn.execute("UPDATE events SET superseded_by = ?, superseded_at = NULL WHERE id = ?", (ids["f"], ids["e"]))
    conn.commit()
    return ids


def _names(ids: dict[str, int], events) -> set[str]:
    by_id = {v: k for k, v in ids.items()}
    return {by_id[e.id] for e in events}


class TestClaimsAsOf:
    def test_before_the_first_claim_nothing_exists(self, conn) -> None:
        ids = _timeline(conn)

        result = claims_as_of(conn, P, 999)

        assert (result.live, result.ended, result.unknown) == ([], [], [])

    def test_mid_history(self, conn) -> None:
        ids = _timeline(conn)

        result = claims_as_of(conn, P, 2000)

        assert _names(ids, result.live) == {"a", "b", "f"}
        assert _names(ids, result.ended) == set()
        assert _names(ids, result.unknown) == {"e"}

    def test_a_transition_at_exactly_t_has_happened(self, conn) -> None:
        ids = _timeline(conn)

        result = claims_as_of(conn, P, 3000)

        assert _names(ids, result.live) == {"a", "c", "d", "f"}
        assert _names(ids, result.ended) == {"b"}
        assert _names(ids, result.unknown) == {"e"}

    def test_after_the_archive(self, conn) -> None:
        ids = _timeline(conn)

        result = claims_as_of(conn, P, 5000)

        assert _names(ids, result.live) == {"a", "c", "f"}
        assert _names(ids, result.ended) == {"b", "d"}
        assert _names(ids, result.unknown) == {"e"}

    def test_horizon_names_the_first_recorded_end_time(self, conn) -> None:
        _timeline(conn)

        result = claims_as_of(conn, P, 5000)

        assert result.first_recorded_end == 3000
        assert result.earliest_claim == 1000

    def test_a_store_with_no_recorded_transitions_has_no_horizon(self, conn) -> None:
        _claim(conn, "only", 1000)

        result = claims_as_of(conn, P, 5000)

        assert result.first_recorded_end is None

    def test_other_projects_are_excluded(self, conn) -> None:
        _timeline(conn)
        insert_event(conn, Event(project_id="other", session_id="x", event_type="DECISION",
                                 payload={"description": "foreign"}, content_hash="foreign", created_at=10))

        result = claims_as_of(conn, P, 5000)

        assert all(e.project_id == P for e in result.live + result.ended + result.unknown)


class TestBuildProjection:
    def test_matches_rebuild_for_the_same_claims(self, conn) -> None:
        _claim(conn, "x", 1000, event_type="CONSTRAINT_HARD")
        _claim(conn, "y", 1100)
        _claim(conn, "z", 1200, event_type="THREAD_OPEN")
        rebuilt = rebuild_projection(conn, P)
        from cognikernel.storage.events import get_events_for_projection

        built = build_projection(conn, P, get_events_for_projection(conn, P))

        for bucket in ("hard_constraints", "ranked_decisions", "graveyard", "active_threads", "component_map"):
            assert getattr(built, bucket) == getattr(rebuilt, bucket), bucket

    def test_building_an_as_of_projection_writes_nothing(self, conn) -> None:
        ids = _timeline(conn)
        before = conn.execute("SELECT COUNT(*) FROM state_projections").fetchone()[0]
        keys_before = conn.execute("SELECT id, decision_key FROM events ORDER BY id").fetchall()

        projection = build_projection(conn, P, claims_as_of(conn, P, 3000).live)

        assert conn.execute("SELECT COUNT(*) FROM state_projections").fetchone()[0] == before
        assert conn.execute("SELECT id, decision_key FROM events ORDER BY id").fetchall() == keys_before
        live_ids = {r["id"] for r in projection.ranked_decisions}
        assert live_ids == {ids["a"], ids["c"], ids["d"], ids["f"]}
