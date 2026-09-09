"""Held-out thread-selection slate — the real store behaviour, frozen.

T-204 / #25 item 3. Every thread fix so far (#21, #22, #26, #28) was verified
against constructed fixtures and then found, on replay against the real stores,
to do something the fixtures did not describe:

  #21 collapsed narration as designed, and also deleted genuine handoffs.
  #22 removed the wrong winner from the top tier without installing the right one.

Both gaps were invisible to unit tests because a unit test asserts what its
author already suspected. This file asserts against data nobody wrote for the
purpose: all 133 THREAD_OPEN/THREAD_CLOSE events from the four benchmark
stores, replayed through the production merge path.

WHAT A BOUNDARY IS, AND WHY IT IS NOT THE END STATE. The graded probes ask
"what's the active thread?" at the START of session N, so the state that
matters is the store after sessions 1..N-1. Measuring the end state after
every session — the obvious thing, and what #25 was originally written from —
reports a state no probe ever sees, because sessions N+1.. do not exist yet
when the question is asked. That mistake made Toolbelt look broken when it was
already correct. Hence one case per boundary, 14 in total.

THIS SET IS HELD OUT. Do not tune a predicate, threshold or vocabulary until
it passes here. If a change moves the slate, the change is either right and
the slate needs re-blessing with a stated reason, or it is wrong — deciding
which is the point of the file, and it cannot be decided by editing the gold
until it is green. Same standing rule as the salience head's eval set.

FIDELITY OF THE TRIMMED FIXTURE. The fixture carries only thread events, not
all 968 events in those stores. That was verified, not assumed: replaying the
full stores and the thread-only subset produced identical selections at all 14
boundaries. If a future change makes thread selection depend on surrounding
events (a centrality or activity term, say), that equivalence breaks and this
fixture must be regenerated from the full stores.

REGENERATING. The fixture is built from ~/.cognikernel benchmark stores, which
are not in the repo. Regeneration is therefore a deliberate act on a machine
that has them; see the issue for the extraction script. Never edit the JSON by
hand to make a test pass.
"""
from __future__ import annotations

import copy
import json
import tempfile
from functools import lru_cache
from pathlib import Path

import pytest

from cognikernel.delta.merge import execute_merge
from cognikernel.injection.ordering import select_active_thread
from cognikernel.model import Event
from cognikernel.quality.detectors import describes_future_session_handoff
from cognikernel.storage.connection import get_connection
from cognikernel.storage.migrations import run_migrations
from cognikernel.storage.projections import load_or_rebuild, projection_to_events

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_EVENTS = _FIXTURES / "thread_boundary_events.json"
_GOLD = _FIXTURES / "thread_boundary_gold.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _replay(rows: list[dict], sessions: list[str], k: int) -> dict:
    """Replay sessions[:k] into a fresh store and report the resulting slate.

    Takes `rows` rather than reading the fixture itself so a test can hand in
    its own object and check afterwards that replaying did not rewrite it.
    """
    with tempfile.TemporaryDirectory() as td:
        with get_connection(Path(td) / "replay.db") as conn:
            run_migrations(conn)
            for sid in sessions[:k]:
                events = [
                    Event(
                        project_id="p",
                        session_id=sid,
                        event_type=r["event_type"],
                        # Deep copy per replay, NOT the caller's dict.
                        # execute_merge runs the quality gate and apply_verdict
                        # mutates event.payload in place (D9 rewrites
                        # `authority` and adds `quality`), so sharing the dict
                        # lets one boundary's demotion leak into every later
                        # boundary. That is not cosmetic: on a second pass D9
                        # sees an already-demoted event, does not fire, and so
                        # does not halve the weight either — identical input
                        # would rank differently depending on replay order.
                        # Measured: 9 of the 133 payloads were affected.
                        payload=copy.deepcopy(r["payload"]),
                        content_hash=r["content_hash"],
                        weight=r["weight"],
                        mention_count=r["mention_count"],
                        created_at=r["created_at"],
                    )
                    for r in rows
                    if r["session"] == sid
                ]
                if events:
                    execute_merge(conn, sid, events)
            live = projection_to_events(load_or_rebuild(conn, "p"))
    winner = select_active_thread(live)
    threads = [e for e in live if e.event_type == "THREAD_OPEN"]
    return {
        "selected": "" if winner is None else winner.payload.get("description", ""),
        "threads": [e.payload.get("description", "") for e in threads],
    }


@lru_cache(maxsize=1)
def _slates() -> dict[tuple[str, int], dict]:
    """Replay every boundary once; keyed by (project, probe_at_session).

    Cached because the same 14 replays back every test below and each one runs
    migrations plus a merge per session.
    """
    fixture = _load(_EVENTS)
    out: dict[tuple[str, int], dict] = {}
    for project, data in fixture.items():
        sessions, rows = data["sessions"], data["events"]
        for k in range(1, len(sessions)):
            out[(project, k + 1)] = _replay(rows, sessions, k)
    return out


def _boundaries() -> list[tuple[str, int]]:
    """Every (project, probe_at_session) pair the gold slate records."""
    gold = _load(_GOLD)
    return sorted(
        (project, entry["probe_at_session"])
        for project, entries in gold.items()
        for entry in entries
    )


_ALL = _boundaries()


class TestThreadBoundaryInvariant:
    """The property the fixes established, stated independently of the slate.

    A snapshot alone would pass just as happily on the wrong answer, as long as
    the wrong answer stayed stable. This is what actually has to hold.
    """

    @pytest.mark.parametrize("project,session", _ALL)
    def test_a_handoff_wins_whenever_one_exists(self, project: str, session: int) -> None:
        slate = _slates()[(project, session)]
        handoffs = [d for d in slate["threads"] if describes_future_session_handoff(d)]
        if not handoffs:
            # Conductor states no handoff at any boundary. Rendering narration
            # there is not a defect — there is nothing better to render — and
            # asserting otherwise would demand the impossible.
            pytest.skip("no handoff candidate exists at this boundary")
        assert describes_future_session_handoff(slate["selected"]), (
            f"{project} S{session}: narration won the slot while "
            f"{len(handoffs)} handoff candidate(s) were available: {handoffs}"
        )

    def test_at_least_one_boundary_has_a_handoff(self) -> None:
        """Guards the skip above from hollowing the whole class out. If a
        change makes the predicate match nothing, every case would skip and
        the suite would go green on a completely broken selector.
        """
        with_handoff = [
            b for b in _ALL
            if any(describes_future_session_handoff(d) for d in _slates()[b]["threads"])
        ]
        assert len(with_handoff) >= 10, (
            f"only {len(with_handoff)} of {len(_ALL)} boundaries have a handoff "
            "candidate — the predicate has probably stopped matching"
        )


class TestThreadBoundarySlate:
    """The exact frozen answer, so any behaviour change shows up as a diff."""

    @pytest.mark.parametrize("project,session", _ALL)
    def test_selection_matches_the_frozen_slate(self, project: str, session: int) -> None:
        gold = next(
            e for e in _load(_GOLD)[project] if e["probe_at_session"] == session
        )
        assert _slates()[(project, session)]["selected"] == gold["selected"], (
            f"{project} S{session} selection changed. If the new answer is "
            "better, re-bless the slate and say why in the commit — do not "
            "edit the gold to make this pass."
        )

    def test_replay_does_not_mutate_its_input(self) -> None:
        """Replaying must not rewrite the events handed to it.

        `execute_merge` runs the quality gate, and `apply_verdict` mutates
        `event.payload` in place. Building Events straight from the loaded
        fixture therefore let one boundary's D9 demotion leak into every later
        boundary — and since a demoted event no longer trips D9, it also
        stopped having its weight halved, so identical input could rank
        differently depending on replay order. Caught in review on #29; 9 of
        the 133 payloads were affected.

        The slate did not move, but that was luck rather than design, so the
        property is asserted directly: replay the busiest project twice from
        one object and require it to come back untouched, and identical.
        """
        fixture = _load(_EVENTS)
        data = fixture["Relay"]
        pristine = copy.deepcopy(data["events"])

        first = _replay(data["events"], data["sessions"], 1)
        assert data["events"] == pristine, "replay rewrote its own input"

        second = _replay(data["events"], data["sessions"], 1)
        assert first == second, "replaying the same boundary twice diverged"

    def test_fixture_is_the_full_133_event_set(self) -> None:
        """The set is held out; silently shrinking it would quietly weaken
        every assertion above."""
        fixture = _load(_EVENTS)
        assert sorted(fixture) == ["Conductor", "Relay", "Taskflow", "Toolbelt"]
        assert sum(len(d["events"]) for d in fixture.values()) == 133
        assert len(_ALL) == 14


class TestGradedProbeAnswers:
    """The three thread probes with a recorded gold answer in the benchmark
    scoring files. Two still fail, for reasons outside thread ranking — they
    are recorded here as strict xfails so they cannot be fixed by accident
    without someone noticing, and cannot be forgotten either.
    """

    def test_toolbelt_s2_selects_the_research_agent_thread(self) -> None:
        """tb_scores_CK.json S2-P1. Gold: 'build a research agent on top of
        toolbelt-core'. This one passes."""
        assert "research agent" in _slates()[("Toolbelt", 2)]["selected"]

    @pytest.mark.xfail(
        strict=True,
        reason="tb_scores_CK.json S3-P1 gold is the T2 thread opened at S2-P8 "
               "(a durable toolbelt-worker package scheduling registered tools). "
               "No session-2 event mentions worker, scheduling or jobs at all: "
               "the thread was never captured. This is an extraction gap — "
               "selection cannot return what was never stored.",
    )
    def test_toolbelt_s3_selects_the_worker_thread(self) -> None:
        selected = _slates()[("Toolbelt", 3)]["selected"].lower()
        assert "worker" in selected or "schedul" in selected

    @pytest.mark.xfail(
        strict=True,
        reason="tf_scores_v2_CK.json S3-P1 gold is the JWT-authentication "
               "thread. It is superseded by 'This is the active work item for "
               "the next session.' — a contentless pointer that carries no "
               "information without the statement it refers to. Handoff-vs-"
               "handoff supersession is deliberately untouched by #26.",
    )
    def test_taskflow_s3_selects_the_jwt_thread(self) -> None:
        # Case-insensitive: this xfail is meant to flip when the selector
        # starts returning the JWT thread, and it should flip whatever casing
        # the winning description happens to use.
        assert "jwt" in _slates()[("Taskflow", 3)]["selected"].lower()
