"""Two concurrent merges must not close a supersession cycle (T-101 / #11).

`set_superseded_by` (storage/events.py) is meant to close the race where two
independent merges each try to supersede the other's event — its docstring
says the fresh re-read "immediately before writing" is what closes it. That
claim is worth its own test, separate from the single-threaded cycle-shape
tests in `tests/unit/storage/test_events.py`: this drives it with two REAL
OS threads, each on its OWN sqlite3 connection to the SAME database file, so
the read-then-write sequence genuinely interleaves at the OS scheduler's
mercy rather than at whatever order a single test thread happens to call
things in.

In production this exact race is narrowed by the per-project worker
single-flight lock (integration/session.py:_acquire_worker_lock) — but that
lock protects the two known worker entry points (the Stop hook's sync drain
and the MCP server's background drainer), not every caller of the storage
layer (a manual repair pass, a future CLI command, a test). The guard is the
correctness boundary of last resort; this test holds it to that job
directly, without going through the worker machinery the lock also covers.

Repeats the race many times with fresh event pairs per iteration rather than
hand-injecting a pause in the middle of the guard — real thread scheduling
plus real disk/WAL I/O gives enough non-determinism that a genuine gap shows
up under repetition without invasive monkeypatching.
"""
from __future__ import annotations

import threading
from pathlib import Path

from cognikernel.storage.connection import get_connection
from cognikernel.storage.events import Event, insert_event, set_superseded_by

_ITERATIONS = 200


def _seed_pair(conn, tag: str) -> tuple[int, int]:
    a = insert_event(
        conn,
        Event(
            project_id="proj1", session_id="s1", event_type="DECISION",
            payload={"description": f"a-{tag}"}, content_hash=f"a-{tag}",
        ),
    )
    b = insert_event(
        conn,
        Event(
            project_id="proj1", session_id="s1", event_type="DECISION",
            payload={"description": f"b-{tag}"}, content_hash=f"b-{tag}",
        ),
    )
    conn.commit()
    return a, b


class TestConcurrentSupersessionRace:
    def test_two_threads_racing_to_supersede_each_other_never_form_a_cycle(
        self, tmp_db: Path
    ) -> None:
        # Seed all pairs up front on one connection, then race the two
        # directions per pair across two independently-connected threads.
        with get_connection(tmp_db) as seed_conn:
            pairs = [_seed_pair(seed_conn, str(i)) for i in range(_ITERATIONS)]

        errors: list[BaseException] = []
        barrier = threading.Barrier(2)

        def _race(direction: int) -> None:
            # direction 0: try A -> B (a.superseded_by = b)
            # direction 1: try B -> A (b.superseded_by = a)
            try:
                with get_connection(tmp_db) as conn:
                    for a, b in pairs:
                        barrier.wait(timeout=10)
                        if direction == 0:
                            set_superseded_by(conn, a, b)
                        else:
                            set_superseded_by(conn, b, a)
                        conn.commit()
            except BaseException as exc:  # a racing thread must never crash silently
                errors.append(exc)

        t0 = threading.Thread(target=_race, args=(0,))
        t1 = threading.Thread(target=_race, args=(1,))
        t0.start()
        t1.start()
        t0.join(timeout=120)
        t1.join(timeout=120)

        assert not errors, f"racing thread(s) raised: {errors}"
        assert not t0.is_alive() and not t1.is_alive(), "a racing thread hung"

        cycles_formed = []
        both_null = []
        with get_connection(tmp_db) as conn:
            for a, b in pairs:
                row_a = conn.execute(
                    "SELECT superseded_by FROM events WHERE id = ?", (a,)
                ).fetchone()
                row_b = conn.execute(
                    "SELECT superseded_by FROM events WHERE id = ?", (b,)
                ).fetchone()
                a_to_b = row_a["superseded_by"] == b
                b_to_a = row_b["superseded_by"] == a
                if a_to_b and b_to_a:
                    cycles_formed.append((a, b))
                if not a_to_b and not b_to_a:
                    both_null.append((a, b))

        # The invariant under test: NEVER both directions set. Every live
        # query filters `superseded_by IS NULL` — a two-cycle silently drops
        # both events out of memory even though neither actually replaced
        # something outside the pair.
        assert cycles_formed == [], (
            f"{len(cycles_formed)}/{_ITERATIONS} pair(s) formed a two-cycle "
            f"under concurrent writes — the guard's re-read-before-write did "
            f"not close the race: {cycles_formed[:5]}"
        )

        # Exactly one direction should normally win each race. Both-refused
        # is not a correctness bug (no cycle either way) but would mean the
        # guard is over-refusing under legitimate concurrent use, which is
        # worth surfacing rather than silently tolerating.
        assert len(both_null) < _ITERATIONS, (
            "every single pair ended with neither direction written — "
            "the guard is refusing legitimate, non-conflicting supersessions"
        )
