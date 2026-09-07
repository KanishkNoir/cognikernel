"""Synthetic scenarios for scripts/repair_supersession_cycles.py.

These exercise detection and repair against engineered graphs BEFORE the
script is ever pointed at a real store — a corrupt two-cycle, a corrupt
3-node cycle (the shape a real store turned up that a naive pairwise checker
misses), a legitimate chain that must survive untouched, and a mix of the two
in one store.

`scripts/repair_supersession_cycles.py` is local-only research/ops tooling
(gitignored — see .gitignore's "local-only research + training tooling"
section; the published wheel contains only src/cognikernel). It is not
present on a fresh clone or in CI, so — mirroring the precedent in
tests/unit/quality/test_baseline_regression.py::TestWilsonInterval — the test
classes below are skipped when the script is absent, and the import is
deferred into each test body so module collection never touches it.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cognikernel.storage.connection import get_connection
from cognikernel.storage.events import Event, insert_event

_REPAIR = Path(__file__).parents[3] / "scripts" / "repair_supersession_cycles.py"


def _seed(conn: sqlite3.Connection, content_hash: str) -> int:
    return insert_event(
        conn,
        Event(
            project_id="proj1",
            session_id="sess1",
            event_type="DECISION",
            payload={"description": content_hash},
            content_hash=content_hash,
            weight=1.0,
        ),
    )


def _link(conn: sqlite3.Connection, event_id: int, by_id: int) -> None:
    """Write a raw superseded_by link, bypassing the guard — simulates
    corruption from before the guard existed."""
    conn.execute("UPDATE events SET superseded_by = ? WHERE id = ?", (by_id, event_id))
    conn.commit()


@pytest.mark.skipif(
    not _REPAIR.exists(),
    reason="scripts/repair_supersession_cycles.py is local-only research tooling",
)
class TestFindCycles:
    def test_no_cycle_on_clean_store(self, conn: sqlite3.Connection) -> None:
        from scripts.repair_supersession_cycles import find_cycles
        a, b = _seed(conn, "a"), _seed(conn, "b")
        _link(conn, a, b)  # a legitimate one-way supersession
        assert find_cycles(conn) == []

    def test_finds_two_cycle(self, conn: sqlite3.Connection) -> None:
        from scripts.repair_supersession_cycles import find_cycles
        a, b = _seed(conn, "a"), _seed(conn, "b")
        _link(conn, a, b)
        _link(conn, b, a)
        cycles = find_cycles(conn)
        assert len(cycles) == 1
        assert set(cycles[0]) == {a, b}

    def test_finds_three_node_cycle(self, conn: sqlite3.Connection) -> None:
        from scripts.repair_supersession_cycles import find_cycles
        a, b, c = _seed(conn, "a"), _seed(conn, "b"), _seed(conn, "c")
        _link(conn, a, b)
        _link(conn, b, c)
        _link(conn, c, a)
        cycles = find_cycles(conn)
        assert len(cycles) == 1
        assert set(cycles[0]) == {a, b, c}

    def test_chain_into_a_winner_is_not_a_cycle(self, conn: sqlite3.Connection) -> None:
        from scripts.repair_supersession_cycles import find_cycles
        a, b, c = _seed(conn, "a"), _seed(conn, "b"), _seed(conn, "c")
        _link(conn, a, b)
        _link(conn, b, c)  # c has no outgoing link — a genuine chain, not a cycle
        assert find_cycles(conn) == []

    def test_cycle_plus_unrelated_chain_in_same_store(self, conn: sqlite3.Connection) -> None:
        from scripts.repair_supersession_cycles import find_cycles
        a, b = _seed(conn, "a"), _seed(conn, "b")
        x, y, z = _seed(conn, "x"), _seed(conn, "y"), _seed(conn, "z")
        _link(conn, a, b)
        _link(conn, b, a)          # cycle
        _link(conn, x, y)
        _link(conn, y, z)          # chain, untouched
        cycles = find_cycles(conn)
        assert len(cycles) == 1
        assert set(cycles[0]) == {a, b}

    def test_two_independent_cycles(self, conn: sqlite3.Connection) -> None:
        from scripts.repair_supersession_cycles import find_cycles
        a, b = _seed(conn, "a"), _seed(conn, "b")
        x, y, z = _seed(conn, "x"), _seed(conn, "y"), _seed(conn, "z")
        _link(conn, a, b)
        _link(conn, b, a)
        _link(conn, x, y)
        _link(conn, y, z)
        _link(conn, z, x)
        cycles = find_cycles(conn)
        assert len(cycles) == 2
        found = {frozenset(c) for c in cycles}
        assert found == {frozenset({a, b}), frozenset({x, y, z})}


@pytest.mark.skipif(
    not _REPAIR.exists(),
    reason="scripts/repair_supersession_cycles.py is local-only research tooling",
)
class TestRepairStore:
    def test_dry_run_does_not_write(self, tmp_db: Path) -> None:
        from scripts.repair_supersession_cycles import repair_store
        with get_connection(tmp_db) as conn:
            a, b = _seed(conn, "a"), _seed(conn, "b")
            _link(conn, a, b)
            _link(conn, b, a)

        n_cycles, n_events = repair_store(tmp_db, apply=False)
        assert (n_cycles, n_events) == (1, 2)

        with get_connection(tmp_db) as conn:
            row = conn.execute("SELECT superseded_by FROM events WHERE id = ?", (a,)).fetchone()
            assert row["superseded_by"] == b  # untouched

    def test_apply_nulls_every_cycle_member(self, tmp_db: Path) -> None:
        from scripts.repair_supersession_cycles import repair_store
        with get_connection(tmp_db) as conn:
            a, b, c = _seed(conn, "a"), _seed(conn, "b"), _seed(conn, "c")
            _link(conn, a, b)
            _link(conn, b, c)
            _link(conn, c, a)

        n_cycles, n_events = repair_store(tmp_db, apply=True)
        assert (n_cycles, n_events) == (1, 3)

        with get_connection(tmp_db) as conn:
            for node in (a, b, c):
                row = conn.execute(
                    "SELECT superseded_by FROM events WHERE id = ?", (node,)
                ).fetchone()
                assert row["superseded_by"] is None

    def test_apply_leaves_legitimate_links_alone(self, tmp_db: Path) -> None:
        from scripts.repair_supersession_cycles import repair_store
        with get_connection(tmp_db) as conn:
            a, b = _seed(conn, "a"), _seed(conn, "b")
            x, y = _seed(conn, "x"), _seed(conn, "y")
            _link(conn, a, b)  # cycle
            _link(conn, b, a)
            _link(conn, x, y)  # genuine one-way supersession — must survive

        repair_store(tmp_db, apply=True)

        with get_connection(tmp_db) as conn:
            row_a = conn.execute("SELECT superseded_by FROM events WHERE id = ?", (a,)).fetchone()
            row_x = conn.execute("SELECT superseded_by FROM events WHERE id = ?", (x,)).fetchone()
            assert row_a["superseded_by"] is None
            assert row_x["superseded_by"] == y

    def test_no_cycles_returns_zero_and_writes_nothing(self, tmp_db: Path) -> None:
        from scripts.repair_supersession_cycles import repair_store
        with get_connection(tmp_db) as conn:
            a, b = _seed(conn, "a"), _seed(conn, "b")
            _link(conn, a, b)

        assert repair_store(tmp_db, apply=True) == (0, 0)

        with get_connection(tmp_db) as conn:
            row = conn.execute("SELECT superseded_by FROM events WHERE id = ?", (a,)).fetchone()
            assert row["superseded_by"] == b
