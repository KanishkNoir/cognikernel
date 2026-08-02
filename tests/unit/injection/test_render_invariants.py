"""Structural invariants over the event set before rendering (spec §4).

These run over EVENTS, not over the assembled string. render_injection
populates survivors_out — the render ledger's source of truth — from the
post-budget event lists, so dropping lines from the finished block would make
the ledger claim an event rendered when its line had been removed.
"""
from cognikernel.injection.template import filter_structural_defects
from cognikernel.model import Event


def _event(event_type: str, description: str, chash: str) -> Event:
    return Event(
        project_id="p",
        session_id="s",
        event_type=event_type,
        payload={"description": description},
        content_hash=chash,
        weight=1.0,
    )


class TestStructuralInvariants:
    def test_drops_box_drawing_event(self) -> None:
        events = [
            _event("DECISION", "Use SQLite for local state.", "a"),
            _event("DECISION", "┌────┬────┐ │ Layer │ Choice │", "b"),
        ]
        out = filter_structural_defects(events)
        assert len(out) == 1
        assert out[0].content_hash == "a"

    def test_drops_cross_type_duplicate_keeping_first(self) -> None:
        events = [
            _event("APPROACH_ABANDONED", "Record Celery as abandoned.", "a"),
            _event("CONSTRAINT_HARD", "record celery as abandoned!", "b"),
        ]
        out = filter_structural_defects(events)
        assert len(out) == 1
        assert out[0].content_hash == "a"

    def test_keeps_distinct_statements(self) -> None:
        events = [
            _event("DECISION", "Use Postgres.", "a"),
            _event("DECISION", "Use Redis for the cache.", "b"),
        ]
        assert len(filter_structural_defects(events)) == 2

    def test_does_not_ground_paths(self) -> None:
        # Prevent-only: a legacy truncated path still renders. Grounding is the
        # admission gate's job and must never run here.
        e = _event("COMPONENT_STATUS", "rc/storage/connection.py modified", "a")
        e.payload["path"] = "rc/storage/connection.py"
        assert filter_structural_defects([e]) == [e]

    def test_empty_input_is_empty_output(self) -> None:
        assert filter_structural_defects([]) == []

    def test_never_raises_on_malformed_event(self) -> None:
        e = _event("DECISION", "fine", "a")
        e.payload = {}          # no description at all
        assert filter_structural_defects([e]) == [e]


class TestSharedSeenAcrossBuckets:
    """The observed D5 defect was one fact rendered in BOTH 'Key decisions' and
    'Do not retry' — two separate buckets. A per-call dedup set cannot catch
    that, so callers pass one shared `seen` across every bucket."""

    def test_duplicate_across_two_buckets_is_dropped(self) -> None:
        seen: set[str] = set()
        graveyard = [_event("APPROACH_ABANDONED", "Record Celery as abandoned.", "a")]
        decisions = [_event("DECISION", "record celery as abandoned", "b")]

        kept_grave = filter_structural_defects(graveyard, seen)
        kept_decs = filter_structural_defects(decisions, seen)

        assert len(kept_grave) == 1
        assert kept_decs == []

    def test_distinct_across_buckets_both_survive(self) -> None:
        seen: set[str] = set()
        a = filter_structural_defects([_event("APPROACH_ABANDONED", "Drop Celery.", "a")], seen)
        b = filter_structural_defects([_event("DECISION", "Use Postgres.", "b")], seen)
        assert len(a) == 1 and len(b) == 1
