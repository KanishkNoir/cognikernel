"""J4.1 — render ledger: dedup, channels, fail-open."""
from __future__ import annotations

import sqlite3

import pytest

from cognikernel.storage.migrations import run_migrations
from cognikernel.storage.render_ledger import record_rendered, rendered_event_ids

PID = "a" * 16


@pytest.fixture()
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    run_migrations(c)
    yield c
    c.close()


def test_record_and_read_back(conn) -> None:
    assert record_rendered(conn, PID, "s1", [1, 2, 3], "block") == 3
    assert rendered_event_ids(conn, PID, "s1") == {1, 2, 3}


def test_dedup_on_repeat(conn) -> None:
    record_rendered(conn, PID, "s1", [1, 2], "block")
    assert record_rendered(conn, PID, "s1", [2, 3], "block") == 1
    assert rendered_event_ids(conn, PID, "s1") == {1, 2, 3}


def test_channels_union_per_session(conn) -> None:
    record_rendered(conn, PID, "s1", [1], "block")
    record_rendered(conn, PID, "s1", [2], "ck1")
    assert rendered_event_ids(conn, PID, "s1") == {1, 2}


def test_sessions_isolated(conn) -> None:
    record_rendered(conn, PID, "s1", [1], "block")
    assert rendered_event_ids(conn, PID, "s2") == set()


def test_invalid_channel_fails_open(conn) -> None:
    assert record_rendered(conn, PID, "s1", [1], "nope") == 0  # CHECK violated → 0


def test_empty_session_id_noop(conn) -> None:
    assert record_rendered(conn, PID, "", [1], "block") == 0
    assert rendered_event_ids(conn, PID, "") == set()


def test_missing_table_fails_open() -> None:
    bare = sqlite3.connect(":memory:")
    assert rendered_event_ids(bare, PID, "s1") == set()
    assert record_rendered(bare, PID, "s1", [1], "block") == 0


def test_render_ex_reports_survivors() -> None:
    """The _ex variant returns the post-enforcement event set (ledger source)."""
    from cognikernel.injection.ordering import make_injection_context
    from cognikernel.injection.template import render_with_budget_enforcement_ex
    from cognikernel.storage.events import Event

    events = [
        Event(project_id=PID, session_id="s", event_type="CONSTRAINT_HARD",
              payload={"description": f"constraint number {i}"},
              content_hash=f"c{i}", id=i)
        for i in range(3)
    ]
    ctx = make_injection_context(
        events=events, project_name="p", session_number=1,
        total_sessions=1, state_version=1, token_budget=2000,
    )
    block, survivors = render_with_budget_enforcement_ex(ctx)
    assert "constraint number" in block
    assert {e.id for e in survivors} == {0, 1, 2}
