"""Demotes reach the ranking.

The quality gate's downgrade, the memory-narration demote and the fragment demote
used to lower only the stored weight. The composite ranking recomputes every
weight and never read that value, so a demoted claim ranked exactly like an
admitted one — while the unit tests, which checked the stored value, all passed.
These tests assert the ranking and the block, not the stored value.
"""
from __future__ import annotations

import sqlite3

import pytest

from cognikernel.compression.greedy import greedy_fill
from cognikernel.compression.token_count import estimate_tokens
from cognikernel.quality.gate import Verdict, apply_verdict
from cognikernel.storage.events import Event, get_events_for_projection, insert_event
from cognikernel.storage.projections import build_projection, projection_to_events

P = "p"
CLEAN = "The dispatcher is a polling loop over the deliveries table."


def _claim(text: str, event_type: str = "DECISION", **payload) -> Event:
    return Event(
        project_id=P, session_id="s1", event_type=event_type,
        payload={"description": text, "authority": "assistant_decided", **payload},
        content_hash=f"{event_type}:{text}",
    )


def _project(conn: sqlite3.Connection, *claims: Event):
    for claim in claims:
        insert_event(conn, claim)
    conn.commit()
    return build_projection(conn, P, get_events_for_projection(conn, P))


def _weight(recs: list[dict], text: str) -> float:
    return next(r["weight"] for r in recs if r["payload"]["description"] == text)


class TestDemotesReachTheRanking:
    def test_a_quality_gate_downgrade_halves_the_ranking_weight(self, conn) -> None:
        downgraded = apply_verdict(_claim("It must not take down the pipeline."),
                                   Verdict("downgrade", "D7", "subject-less"))

        projection = _project(conn, _claim(CLEAN), downgraded)

        clean = _weight(projection.ranked_decisions, CLEAN)
        assert _weight(projection.ranked_decisions, "It must not take down the pipeline.") == pytest.approx(clean * 0.5)

    def test_memory_narration_ranks_far_below_a_project_fact(self, conn) -> None:
        narration = "CogniKernel's Stop hook will persist the updated rationale."

        projection = _project(conn, _claim(CLEAN), _claim(narration))

        clean = _weight(projection.ranked_decisions, CLEAN)
        assert _weight(projection.ranked_decisions, narration) == pytest.approx(clean * 0.15)
        assert projection.ranked_decisions[0]["payload"]["description"] == CLEAN

    def test_a_fragment_ranks_below_a_project_fact(self, conn) -> None:
        fragment = "Use the same key for both."

        projection = _project(conn, _claim(CLEAN), _claim(fragment, provenance="salience_v2_broad+frag"))

        assert _weight(projection.ranked_decisions, fragment) == pytest.approx(
            _weight(projection.ranked_decisions, CLEAN) * 0.4)

    def test_assistant_step_narration_ranks_below_a_project_fact(self, conn) -> None:
        from cognikernel.quality.detectors import STEP_NARRATION_DEMOTE

        narration = "Now update dispatcher.py to use the renamed store API."

        projection = _project(conn, _claim(CLEAN), _claim(narration, source_role="assistant"))

        assert _weight(projection.ranked_decisions, narration) == pytest.approx(
            _weight(projection.ranked_decisions, CLEAN) * STEP_NARRATION_DEMOTE)

    def test_thread_authority_demotes_leave_thread_weight_alone(self, conn) -> None:
        demoted = "Add the response schema for a task."
        clean = "Pick up the JWT auth work next session."

        projection = _project(
            conn,
            _claim(clean, "THREAD_OPEN"),
            _claim(demoted, "THREAD_OPEN", quality="instruction_not_thread"),
        )

        assert _weight(projection.active_threads, demoted) == _weight(projection.active_threads, clean)

    def test_when_only_one_fits_the_block_keeps_the_project_fact(self, conn) -> None:
        """Inserted first, so with equal weights the narration would have won the slot."""
        narration = "CogniKernel's Stop hook will persist the new rationale."
        projection = _project(conn, _claim(narration), _claim(CLEAN))
        events = projection_to_events(projection)
        costs = sorted(estimate_tokens(e) for e in events)

        selected = greedy_fill(events, budget_tokens=costs[0] + costs[1] - 1)

        assert [e.payload["description"] for e in selected] == [CLEAN]
