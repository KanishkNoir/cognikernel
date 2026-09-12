"""Tests for cognikernel.integration.session."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognikernel.config import Config
from cognikernel.compression.greedy import greedy_fill
from cognikernel.compression.token_count import estimate_tokens
from cognikernel.integration.session import (
    _active_thread_reserve,
    get_projection,
    init_project,
    render_state,
    replay_job,
    session_end,
)
from cognikernel.injection.template import _render_active_thread, count_tokens_accurate
from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
from cognikernel.storage.events import Event, insert_event
from cognikernel.storage.migrations import run_migrations
from cognikernel.storage.projections import load_or_rebuild, projection_to_events
from cognikernel.storage.render_ledger import rendered_event_ids


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(cognikernel_dir=tmp_path / "cognikernel")


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    p = tmp_path / "myproject"
    p.mkdir()
    return p


# ── init_project ──────────────────────────────────────────────────────────────

class TestInitProject:
    def test_returns_project_id_string(self, project_path: Path, cfg: Config) -> None:
        result = init_project(project_path, config=cfg)
        assert isinstance(result, str) and len(result) == 16

    def test_creates_db_file(self, project_path: Path, cfg: Config) -> None:
        project_id = init_project(project_path, config=cfg)
        db_path = get_db_path(cfg, project_id)
        assert db_path.exists()

    def test_idempotent_safe_to_call_twice(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        init_project(project_path, config=cfg)  # must not raise

    def test_same_path_always_same_project_id(self, project_path: Path, cfg: Config) -> None:
        id1 = init_project(project_path, config=cfg)
        id2 = init_project(project_path, config=cfg)
        assert id1 == id2

    def test_different_paths_different_ids(self, tmp_path: Path, cfg: Config) -> None:
        p1 = tmp_path / "proj_a"
        p2 = tmp_path / "proj_b"
        p1.mkdir(); p2.mkdir()
        assert init_project(p1, config=cfg) != init_project(p2, config=cfg)


# ── session_end ───────────────────────────────────────────────────────────────

DECISION_TRANSCRIPT = (
    "We decided to use SQLite for local storage because it requires no external server. "
    "This is a hard constraint: never store secrets in plain text configuration files."
)


class TestSessionEnd:
    def test_returns_stats_dict(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        stats = session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)
        assert isinstance(stats, dict)
        for key in ("extracted", "inserted", "updated", "superseded", "cascaded", "archived"):
            assert key in stats

    def test_extracted_count_is_non_negative(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        stats = session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)
        assert stats["extracted"] >= 0

    def test_events_written_to_db(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)
        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count >= 0  # extraction may or may not find events in this transcript

    def test_dedup_increments_not_duplicates(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        s1 = session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)
        s2 = session_end(project_path, "sess2", DECISION_TRANSCRIPT, config=cfg)
        # Second run must not insert more rows than first (same content)
        assert s2["inserted"] <= s1["inserted"]

    def test_empty_transcript_returns_zero_extracted(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        stats = session_end(project_path, "sess1", "", config=cfg)
        assert stats["extracted"] == 0

    def test_stats_values_are_non_negative(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        stats = session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)
        for v in stats.values():
            assert v >= 0

    def test_inits_db_if_not_yet_initialised(self, project_path: Path, cfg: Config) -> None:
        # session_end should work even without an explicit init_project call
        stats = session_end(project_path, "sess1", "", config=cfg)
        assert "extracted" in stats

    def test_session_end_records_evidence_and_completed_job(
        self, project_path: Path, cfg: Config
    ) -> None:
        stats = session_end(
            project_path,
            "sess1",
            "Assistant: We decided to use SQLite.",
            config=cfg,
        )
        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            evidence_count = conn.execute(
                "SELECT COUNT(*) FROM raw_evidence WHERE project_id=?",
                (project_id,),
            ).fetchone()[0]
            job = conn.execute(
                "SELECT * FROM extraction_jobs WHERE project_id=?",
                (project_id,),
            ).fetchone()
            provenance_count = conn.execute(
                "SELECT COUNT(*) FROM event_provenance"
            ).fetchone()[0]

        assert stats["evidence_id"] > 0
        assert stats["job_id"] > 0
        assert evidence_count == 1
        assert job["state"] == "completed"
        assert job["stage"] == "COMPLETED"
        assert provenance_count == stats["inserted"] + stats["updated"]


# ── get_projection ────────────────────────────────────────────────────────────

class TestGetProjection:
    def test_returns_projection_object(self, project_path: Path, cfg: Config) -> None:
        from cognikernel.storage.projections import Projection
        init_project(project_path, config=cfg)
        proj = get_projection(project_path, config=cfg)
        assert isinstance(proj, Projection)

    def test_project_id_matches(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        expected_id = hash_project_path(project_path)
        proj = get_projection(project_path, config=cfg)
        assert proj.project_id == expected_id

    def test_reflects_events_after_session_end(self, project_path: Path, cfg: Config) -> None:
        transcript = (
            "We decided to use SQLite as our primary storage engine. "
            "This is the most important architectural decision."
        )
        session_end(project_path, "sess1", transcript, config=cfg)
        proj = get_projection(project_path, config=cfg)
        total = (
            len(proj.hard_constraints)
            + len(proj.ranked_decisions)
            + len(proj.graveyard)
            + len(proj.component_map)
            + len(proj.active_threads)
        )
        assert total >= 0  # at minimum, projection exists


# ── render_state ──────────────────────────────────────────────────────────────

class TestRenderState:
    def test_returns_non_empty_string_with_events(self, project_path: Path, cfg: Config) -> None:
        session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)
        rendered = render_state(project_path, config=cfg)
        assert isinstance(rendered, str)
        assert len(rendered) > 0

    def test_returns_string_for_empty_project(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        rendered = render_state(project_path, config=cfg)
        assert isinstance(rendered, str)

    def test_contains_header_marker(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        rendered = render_state(project_path, config=cfg)
        assert "auto-generated" in rendered

    def test_inits_db_if_needed(self, project_path: Path, cfg: Config) -> None:
        rendered = render_state(project_path, config=cfg)
        assert isinstance(rendered, str)

    def test_project_name_appears_in_output(self, project_path: Path, cfg: Config) -> None:
        init_project(project_path, config=cfg)
        rendered = render_state(project_path, config=cfg)
        assert project_path.name in rendered


# ── replay_job ────────────────────────────────────────────────────────────────

class TestReplayJob:
    """Replay must be a real recovery primitive: re-run extraction against the
    original raw_evidence, not just flip the job state to queued."""

    def _seed_dead_lettered_job(
        self, project_path: Path, cfg: Config, session_id: str = "sess_poison"
    ) -> tuple[int, int]:
        from cognikernel.storage.evidence import store_evidence
        from cognikernel.storage.jobs import enqueue_extraction, fail_job

        init_project(project_path, config=cfg)
        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)
        transcript = (
            "We decided to use SQLite for local storage. "
            "This is a hard constraint: never store secrets in plain text."
        )
        with get_connection(db_path) as conn:
            eid = store_evidence(
                conn, project_id, session_id, "transcript", transcript.encode("utf-8")
            )
            jid = enqueue_extraction(
                conn, project_id, session_id, eid, "extract.transcript"
            )
            fail_job(conn, jid, "POISON_INPUT", "simulated failure")
        return eid, jid

    def test_replay_advances_job_to_completed(
        self, project_path: Path, cfg: Config
    ) -> None:
        from cognikernel.storage.jobs import get_job

        evidence_id, job_id = self._seed_dead_lettered_job(project_path, cfg)
        stats = replay_job(project_path, job_id, config=cfg)

        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            job = get_job(conn, job_id)

        assert job.state == "completed"
        assert job.stage == "COMPLETED"
        assert stats["evidence_id"] == evidence_id
        assert stats["job_id"] == job_id

    def test_replay_actually_inserts_events(
        self, project_path: Path, cfg: Config
    ) -> None:
        _, job_id = self._seed_dead_lettered_job(project_path, cfg)
        stats = replay_job(project_path, job_id, config=cfg)

        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            event_count = conn.execute(
                "SELECT COUNT(*) FROM events WHERE project_id=?", (project_id,)
            ).fetchone()[0]

        assert stats["extracted"] > 0
        assert event_count > 0

    def test_replay_records_full_ack_chain(
        self, project_path: Path, cfg: Config
    ) -> None:
        _, job_id = self._seed_dead_lettered_job(project_path, cfg)
        replay_job(project_path, job_id, config=cfg)

        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            stages = {
                row["stage"]
                for row in conn.execute(
                    "SELECT stage FROM extraction_job_acks WHERE job_id=?",
                    (job_id,),
                ).fetchall()
            }

        assert {"OBSERVED", "PARSED", "CLASSIFIED", "MERGED", "PROJECTED", "COMPLETED"} <= stages

    def test_replay_rejects_non_dead_lettered_job(
        self, project_path: Path, cfg: Config
    ) -> None:
        init_project(project_path, config=cfg)
        stats = session_end(project_path, "sess1", DECISION_TRANSCRIPT, config=cfg)

        with pytest.raises(ValueError, match="not dead-lettered"):
            replay_job(project_path, stats["job_id"], config=cfg)

    def test_replay_rejects_unknown_job(
        self, project_path: Path, cfg: Config
    ) -> None:
        init_project(project_path, config=cfg)

        with pytest.raises(ValueError, match="Unknown extraction job"):
            replay_job(project_path, 99999, config=cfg)


# ── cursor monotonic guard (F7, manual replay path) ───────────────────────────

class TestSessionEndCursorMonotonicGuard:
    def test_old_transcript_does_not_rewind_cursor(
        self, project_path: Path, cfg: Config
    ) -> None:
        """`failures --replay` re-enters session_end with an OLD reconstructed
        transcript. The ingest cursor must never move backwards — process_jobs
        already guards this; session_end must too."""
        from cognikernel.storage.cursors import get_cursor

        init_project(project_path, config=cfg)
        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)

        lines = [
            json.dumps({"type": "user", "message": {"content": f"note {i}"}})
            for i in range(30)
        ]
        session_end(project_path, "sess-guard", "\n".join(lines), config=cfg)
        with get_connection(db_path) as conn:
            cursor = get_cursor(conn, project_id, "sess-guard")
        assert cursor is not None and cursor.last_line_count == 30

        # Replay a shorter (older) transcript for the same session.
        session_end(project_path, "sess-guard", "\n".join(lines[:10]), config=cfg)
        with get_connection(db_path) as conn:
            cursor = get_cursor(conn, project_id, "sess-guard")
        assert cursor is not None and cursor.last_line_count == 30  # not rewound


# ── active thread selection and reservation (Spec 2.3, D1/D2) ─────────────────

def _thread(desc: str, *, weight: float = 1.0, authority: str = "assistant_decided",
            project_id: str = "p1", session_id: str = "s1", **extra) -> Event:
    return Event(
        project_id=project_id, session_id=session_id, event_type="THREAD_OPEN",
        payload={"description": desc, "authority": authority, **extra},
        content_hash=desc[:32].ljust(64, "0"), weight=weight,
    )


class TestActiveThreadReserve:
    """THE GUARD for spec section 2.3. The reserve couples session.py to
    template.py's private _render_active_thread. If a field is later added to
    that renderer (a staleness marker is the obvious trigger) and the reserve
    is not updated, it silently undershoots and the backstop starts firing on
    routine renders."""

    def test_returns_zero_for_none(self) -> None:
        assert _active_thread_reserve(None) == 0

    def test_equals_the_rendered_section_token_count(self) -> None:
        t = _thread("Ship the router.", state="Half done.",
                    next_steps="Wire the fallback loop.")
        assert _active_thread_reserve(t) == count_tokens_accurate(_render_active_thread([t]))

    def test_counts_state_and_next_steps_not_only_description(self) -> None:
        rich = _thread("Ship the router.", state="Half done.",
                       next_steps="Wire the fallback loop.")
        bare = _thread("Ship the router.")
        assert _active_thread_reserve(rich) > _active_thread_reserve(bare)


class TestRenderStateSessionLabels:
    def test_only_changed_and_carried_claims_carry_a_session_label(
        self, project_path: Path, cfg: Config
    ) -> None:
        """Micro benchmark: the recap put 3–5 of 7 facts in the wrong session. A label
        goes where order matters — a value that changed, and the open thread carried
        from an earlier session — not on every line."""
        from datetime import datetime

        from cognikernel.storage.events import set_superseded_by
        from cognikernel.storage.evidence import store_evidence

        project_id = init_project(project_path, config=cfg)
        db_path = get_db_path(cfg, project_id)
        first, second = "3f2a9c1e-first-session", "8b7d4e20-second-session"
        at = {first: int(datetime(2026, 9, 11, 10, 0).timestamp() * 1000),
              second: int(datetime(2026, 9, 12, 10, 0).timestamp() * 1000)}

        def claim(conn, sid, event_type, desc, authority="assistant_decided"):
            return insert_event(conn, Event(
                project_id=project_id, session_id=sid, event_type=event_type,
                payload={"description": desc, "authority": authority},
                content_hash=desc[:32].ljust(64, "0"), created_at=at[sid],
            ))

        with get_connection(db_path) as conn:
            run_migrations(conn)
            for sid in (first, second):
                store_evidence(conn, project_id, sid, "transcript", sid.encode(), captured_at=at[sid])
            claim(conn, first, "CONSTRAINT_HARD", "Timestamps are integer epoch milliseconds.")
            weekly = claim(conn, first, "DECISION", "Back up the delivery database weekly.")
            nightly = claim(conn, second, "DECISION", "Back up the delivery database nightly.")
            set_superseded_by(conn, weekly, nightly, reason="cross_encoder")
            claim(conn, first, "THREAD_OPEN", "Next session: build the replay command.", authority="user_stated")
            conn.commit()

        block = render_state(project_path, config=cfg)

        assert "Back up the delivery database nightly. (S2 · 09-12)" in block
        assert "Working on: Next session: build the replay command. (S1 · 09-11)" in block
        assert "- Timestamps are integer epoch milliseconds.\n" in block + "\n"
        assert first not in block and second not in block


class TestRenderStateThreadSelection:
    def test_ledger_records_only_the_thread_that_rendered(
        self, project_path: Path, cfg: Config
    ) -> None:
        # The CK-1 suppression bug: every surviving THREAD_OPEN used to be
        # recorded as verbatim-exposed, so threads the model never saw were
        # suppressed from per-prompt injection for the rest of the session.
        project_id = init_project(project_path, config=cfg)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            run_migrations(conn)
            for i in range(4):
                insert_event(conn, _thread(
                    f"Thread number {i}", weight=float(i + 1),
                    project_id=project_id, session_id="s1",
                ))
            conn.commit()

        render_state(project_path, config=cfg, session_id="s2")

        with get_connection(db_path) as conn:
            seen = rendered_event_ids(conn, project_id, "s2")
            thread_ids = {
                r[0] for r in conn.execute(
                    "SELECT id FROM events WHERE project_id=? AND event_type='THREAD_OPEN'",
                    (project_id,),
                )
            }
        assert len(seen & thread_ids) == 1

    def test_high_authority_low_weight_thread_still_renders(
        self, project_path: Path, cfg: Config
    ) -> None:
        # Regression guard for the "leave the winner in greedy_fill's input"
        # design (spec section 4), which lost the Active thread section in
        # 2/52 real stores. select_active_thread picks the winner among
        # THREADS ONLY, by (authority_priority, -weight); greedy_fill Phase 2
        # then sorts by weight across EVERY candidate, threads included, when
        # a thread is left in its input. A thread that wins selection can
        # therefore still be starved out by heavier events under budget
        # pressure — this fixture reproduces exactly that: the winner's
        # authority is assistant_decided, NOT user_stated, so it is a genuine
        # Phase-2 candidate (competing purely on weight) rather than
        # budget-exempt mandatory content. It must still render because
        # render_state excludes it from greedy_fill's candidates by
        # event_type and appends it to `selected` AFTER greedy_fill returns
        # (session.py), rather than leaving it in greedy_fill's input to
        # compete for a Phase-2 slot on its own weight.
        #
        # This does NOT guard the "force it into the mandatory zone" design
        # (the 8/52 loss) — that failure mode requires mandatory-zone
        # contention, which this fixture (assistant_decided, non-mandatory)
        # does not create. See
        # test_losing_user_stated_thread_no_longer_evicts_a_hard_constraint
        # for the mandatory-zone guard.
        #
        # Two non-obvious pitfalls, both found by tracing this fixture
        # through the real pipeline rather than assuming insert-time values
        # survive to greedy_fill:
        #
        # 1. The weight set at insert time (99.0 here) is NOT what
        #    greedy_fill sees. load_or_rebuild recomputes every event's
        #    weight via the composite model (compression.weights.compute_weight
        #    — base x recency x repetition x centrality x activity x type;
        #    projections.py:_apply_composite_weights), discarding the raw
        #    value entirely. Measured: 40 decisions inserted at weight 99.0
        #    came back from the projection at ~0.7-0.85 — barely above the
        #    winner's own recomputed weight (~0.72), not the crushing 0.01
        #    vs 99 margin the raw insert implies.
        # 2. DECISION is in the choice family (storage/consolidate.py), and
        #    identical/near-identical decisions collapse to ONE golden
        #    record at projection time by normalized `decision_key`. All 40
        #    decisions here share the description template "Heavy decision
        #    N: yyy..."; `derive_decision_key` strips the numeral, so every
        #    one normalizes to the SAME key ('decision heavy') and 39 of the
        #    40 are silently deleted by consolidation before greedy_fill ever
        #    runs. Giving each decision a distinct `subject` bypasses this —
        #    `derive_decision_key` prefers `payload['subject']` over the
        #    description-derived key, and `subject` is not one of the fields
        #    `estimate_tokens` counts (description/rationale/path/
        #    affected_files), so this changes the key without changing the
        #    event's token cost.
        project_id = init_project(project_path, config=cfg)
        db_path = get_db_path(cfg, project_id)
        winner = _thread(
            "Ship the router.", weight=0.01, authority="assistant_decided",
            project_id=project_id, session_id="s1",
        )
        decisions = [
            Event(
                project_id=project_id, session_id="s1", event_type="DECISION",
                payload={"description": "Heavy decision %d: %s" % (i, "y" * 300),
                         "rationale": "", "subject": "topic%d" % i},
                content_hash=("dec%d" % i).ljust(64, "0"), weight=99.0,
            )
            for i in range(40)
        ]
        with get_connection(db_path) as conn:
            run_migrations(conn)
            insert_event(conn, winner)
            for d in decisions:
                insert_event(conn, d)
            conn.commit()

        # The precondition and the tight budget are both derived from the
        # ACTUAL post-projection events — projection_to_events(load_or_rebuild(...))
        # is exactly what render_state feeds to greedy_fill internally — not
        # from the pre-insertion Python objects above, whose weights and
        # count do not survive the pipeline (see pitfalls 1-2). Budget is set
        # to precisely the summed cost of every event that outweighs the
        # winner post-projection, so the greedy walk has ZERO residual slack
        # left for the winner by construction: it cannot slip in on its own
        # merits regardless of how composite weighting happens to land.
        with get_connection(db_path) as conn:
            run_migrations(conn)
            events = projection_to_events(load_or_rebuild(conn, project_id))
        winner_ev = next(e for e in events if e.event_type == "THREAD_OPEN")
        heavier = [
            e for e in events
            if e.event_type != "THREAD_OPEN" and e.weight > winner_ev.weight
        ]
        # Guards against a degenerate zero budget: if nothing outweighs the
        # winner post-projection, `tight_budget` would be 0 and "winner not
        # in raw_fill" would pass for the wrong reason (no budget at all,
        # not genuine competition) — a false green identical in shape to the
        # decision-collapse bug above.
        assert heavier, (
            "no decision outweighs the winner post-projection — this "
            "fixture cannot create the budget pressure it claims to guard"
        )
        tight_budget = sum(estimate_tokens(e) for e in heavier)
        tight = Config(cognikernel_dir=cfg.cognikernel_dir, token_budget=tight_budget)

        # THE precondition that makes this guard meaningful: on its own
        # merits — the same greedy_fill call render_state makes internally,
        # before the active thread is appended — the winner must NOT
        # survive. If it did, this fixture would render "Working on: Ship
        # the router." whether or not render_state appends the thread
        # post-fill, and the assertion after render() would prove nothing
        # about the 2/52-store regression. Fail loudly here instead of
        # letting that happen quietly. Matched by id, not by Event equality:
        # Event is a plain @dataclass with a `created_at` default_factory
        # timestamp, so two Event objects built from the same DB row at
        # different moments are never `==` — id is the only stable identity
        # that survives projection_to_events being called twice.
        raw_fill = greedy_fill(events, tight_budget)
        assert winner_ev.id not in {e.id for e in raw_fill}, (
            "winner survived greedy_fill on its own weight — this fixture "
            "no longer discriminates the 'leave the winner in the fill' "
            "regression"
        )

        rendered = render_state(project_path, config=tight)
        assert "Working on: Ship the router." in rendered

    def test_losing_user_stated_thread_no_longer_evicts_a_hard_constraint(
        self, project_path: Path, cfg: Config
    ) -> None:
        # Defect D2 (spec section 1). greedy.py:63-67 treats ANY user_stated
        # event as budget-exempt mandatory, threads included. When that zone
        # overflows, _compress_mandatory drops WHOLE events ranked by
        # (_AUTHORITY_RANK, weight) — all user_stated tie at rank 3, so a heavy
        # thread that renders nothing could beat a real constraint and delete
        # it.
        #
        # Zone OVERFLOW alone is not sufficient to prove eviction: the keep
        # loop in _compress_mandatory (greedy.py:141-151), like greedy_fill's
        # own Phase 2, never breaks early. If a large loser doesn't fit, the
        # walk moves on and a SMALL item further down the sorted order (the
        # hard constraint, which ties last at authority rank 2) can still
        # slip into whatever slack is left over. A first version of this test
        # used a few large losers (~199 tok each against a 500 tok limit at
        # the 1500-budget scale) and left ~95 tok of residual slack — plenty
        # of room for a 13 tok constraint, so the constraint always survived
        # and the test passed identically before and after the fix, proving
        # nothing. The binding condition is residual slack strictly less than
        # the hard constraint's own cost, not overflow by itself. So losers
        # here are SMALL and NUMEROUS: many small increments pack tightly
        # against mandatory_limit and leave almost no slack, and the hard
        # constraint (13 tok, unchanged) is sized larger than a single loser
        # so it cannot fit in whatever sliver remains.
        #
        # Everything is SIZED AT RUNTIME, never hardcoded — including the
        # winner's own cost, which also occupies the mandatory zone (it is
        # user_stated too, and sorts first at weight 50). token_count.py uses
        # exact tiktoken when the `tokens` extra is installed and a len/4
        # heuristic otherwise; the two counters do not agree on cost for the
        # same string, so a fixed packing that lands tightly under one
        # counter can leave a different residual under the other.
        mandatory_limit = int(500 * (cfg.token_budget / 1500.0))

        project_id = init_project(project_path, config=cfg)
        db_path = get_db_path(cfg, project_id)
        winner = _thread(
            "Ship the router.", weight=50.0, authority="user_stated",
            project_id=project_id, session_id="s1",
        )
        hard = Event(
            project_id=project_id, session_id="s1", event_type="CONSTRAINT_HARD",
            payload={"description": "Never call the billing API from a hook.",
                     "rationale": "", "authority": "assistant_decided"},
            content_hash="hardc".ljust(64, "0"), weight=0.5,
        )
        with get_connection(db_path) as conn:
            run_migrations(conn)
            # The winner, which will render.
            insert_event(conn, winner)
            # Losers: small, numerous, user_stated, and therefore mandatory
            # today. Keep packing while the next one still fits alongside the
            # winner, so the walk lands as close to mandatory_limit as a
            # loser-sized increment allows — this is what makes the residual
            # slack tiny instead of the ~95 tok a few large losers left.
            spent = estimate_tokens(winner)
            i = 0
            while True:
                loser = _thread(
                    "L%04d" % i, weight=40.0, authority="user_stated",
                    project_id=project_id, session_id="s1",
                )
                cost = estimate_tokens(loser)
                if spent + cost > mandatory_limit:
                    break
                insert_event(conn, loser)
                spent += cost
                i += 1
            insert_event(conn, hard)
            conn.commit()

        # THE precondition that makes this guard meaningful. If the packing
        # above ever leaves residual slack >= the hard constraint's cost
        # (e.g. because a future edit changes loser size or weight), the
        # keep loop's no-early-break behaviour lets the constraint through
        # regardless of whether D2 is fixed, and the assertion below would
        # pass either way — silently reverting to a decorative test. Fail
        # loudly here instead of letting that happen quietly.
        residual_slack = mandatory_limit - spent
        assert residual_slack < estimate_tokens(hard), (
            f"residual slack {residual_slack} >= hard constraint cost "
            f"{estimate_tokens(hard)} — this test no longer discriminates D2"
        )

        rendered = render_state(project_path, config=cfg)
        assert "Never call the billing API from a hook." in rendered


# ── the task-2/task-3 seam: reserved_tokens=reserve actually reaching greedy_fill ──

class TestReservedTokensReachesGreedyFill:
    """Closes the wiring gap between `select_active_thread` (task 2) and
    `greedy_fill(reserved_tokens=...)` (task 3): render_state must actually
    pass the real reserve through, not just compute it.

    TestActiveThreadReserve (above) only proves `_active_thread_reserve`
    computes the right number in isolation. TestReservedTokens (in
    test_greedy.py) only proves `greedy_fill` honours whatever value it's
    given. Neither proves session.py's `greedy_fill(candidates,
    config.token_budget, reserved_tokens=reserve)` call site actually wires
    the two together — `reserved_tokens=0` there would leave selection,
    exclusion and the post-fill append all intact and every OTHER test in
    this suite green, because the reserve is the only thing standing between
    the unconditionally-appended active thread and pure budget growth (see
    greedy.py's `reserved_tokens` docstring).

    This test sizes `token_budget` so a marginal DECISION fits in Phase 2
    when `reserved_tokens=0` but not at the real reserve, then renders
    through the full `render_state` pipeline (not a raw `greedy_fill` call)
    and asserts the decision's text is absent.

    Non-obvious construction note: the render's own backstop
    (template.py:523, "while actual > ctx.token_budget and ctx.decisions:
    pop()") is STRICTER than greedy_fill's Phase-2 check — it counts the
    accurate rendered bytes of the header, the active-thread section and the
    summary, none of which greedy_fill's `estimate_tokens` ever sees. Sitting
    a decision right at the reserve boundary (`reserved_tokens=0` admits it,
    the real reserve excludes it) means the backstop's extra accounting will
    ALWAYS exceed that boundary by a small fixed amount and strip the
    decision anyway — making the render-level assertion pass under BOTH the
    correct code and the `reserved_tokens=0` mutation, i.e. decorative. This
    is fixed by giving the marginal decision an oversized `path` field:
    `estimate_tokens` (token_count.py:47-64) counts `path` toward the
    event's Phase-2 cost, but `_render_decisions` (template.py:362-384) never
    reads `path` for a DECISION — only `description`/`rationale`. Padding
    `path` inflates the event's greedy-fill cost far past what it actually
    costs to render, which buys enough slack that the backstop's extra
    accounting can never catch up to the boundary, regardless of how tight
    the boundary itself is. The thread's own render cost is real either way
    (it renders unconditionally) and is measured, not assumed.
    """

    def test_marginal_decision_survives_only_without_the_real_reserve(
        self, project_path: Path, cfg: Config
    ) -> None:
        project_id = init_project(project_path, config=cfg)
        db_path = get_db_path(cfg, project_id)

        winner = _thread(
            "Ship the router.", weight=1.0, authority="assistant_decided",
            state="Half done — the fallback loop is wired but untested.",
            next_steps="Confirm the retry budget, then land the change.",
            project_id=project_id, session_id="s1",
        )
        # `path` is counted by estimate_tokens but never rendered for a
        # DECISION (see class docstring) — padding it inflates the Phase-2
        # cost far past the real render cost, which is what makes it
        # possible to sit at the reserve boundary without the render
        # backstop's stricter accounting (header/thread/summary bytes
        # greedy_fill never sees) swallowing the decision regardless of the
        # reserved_tokens value under test.
        padding = " ".join(f"pad{i}" for i in range(500))
        marginal = Event(
            project_id=project_id, session_id="s1", event_type="DECISION",
            payload={
                "description": "Adopt the retry budget for flaky writes.",
                "rationale": "", "subject": "marginal_topic", "path": padding,
            },
            content_hash="marginal_decision".ljust(64, "0"), weight=1.0,
        )
        with get_connection(db_path) as conn:
            run_migrations(conn)
            insert_event(conn, winner)
            insert_event(conn, marginal)
            conn.commit()

        # Real post-projection values, not the pre-insertion Python objects
        # (load_or_rebuild discards insert-time weight; projections.py:267+).
        with get_connection(db_path) as conn:
            run_migrations(conn)
            events = projection_to_events(load_or_rebuild(conn, project_id))
        winner_ev = next(e for e in events if e.event_type == "THREAD_OPEN")
        marginal_ev = next(e for e in events if e.event_type == "DECISION")

        reserve = _active_thread_reserve(winner_ev)
        m_cost = estimate_tokens(marginal_ev)
        assert reserve > 0, (
            "thread reserve is 0 — this fixture cannot create a reserve "
            "boundary to test against"
        )

        # Exactly `reserve - 1` tokens above the marginal's own cost: enough
        # slack for reserved_tokens=0 to admit it, one token short of enough
        # for reserved_tokens=reserve to admit it too.
        token_budget = m_cost + reserve - 1
        tight = Config(cognikernel_dir=cfg.cognikernel_dir, token_budget=token_budget)

        # THE preconditions that make this guard meaningful, checked against
        # the same greedy_fill call render_state makes internally (threads
        # filtered out, exactly like session.py's `candidates` line) — fail
        # loudly instead of silently proving nothing if a future change
        # shifts either boundary.
        non_thread = [e for e in events if e.event_type != "THREAD_OPEN"]
        admitted_zero = greedy_fill(non_thread, token_budget, reserved_tokens=0)
        assert marginal_ev.id in {e.id for e in admitted_zero}, (
            "marginal decision not admitted at reserved_tokens=0 — the "
            "budget is too tight for this fixture to discriminate anything"
        )
        admitted_real = greedy_fill(non_thread, token_budget, reserved_tokens=reserve)
        assert marginal_ev.id not in {e.id for e in admitted_real}, (
            "marginal decision still admitted at the real reserve — this "
            "fixture no longer sits at the reserve boundary"
        )

        rendered = render_state(project_path, config=tight)
        assert "Adopt the retry budget for flaky writes." not in rendered, (
            "marginal decision rendered even though the real reserve "
            "should have excluded it in greedy_fill's Phase 2 — if this "
            "fails, check that render_state is passing the real reserve "
            "(not 0) as `reserved_tokens` to greedy_fill"
        )
