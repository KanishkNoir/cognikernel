"""CK-1 — recall_for_prompt: dual-evidence gate + ledger redundancy (J4.2)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from cognikernel.integration.query import recall_for_prompt


def _project(tmp_path: Path, monkeypatch) -> tuple[str, str, object]:
    monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path))
    monkeypatch.setattr("cognikernel.embedding.model.is_ready", lambda: False)
    monkeypatch.setattr("cognikernel.embedding.model.warm", lambda: None)
    from cognikernel.config import Config
    from cognikernel.integration.session import init_project
    from cognikernel.storage.connection import get_db_path, hash_project_path

    proj = str(tmp_path / "proj")
    Path(proj).mkdir()
    init_project(proj)
    pid = hash_project_path(proj)
    db = get_db_path(Config.load(project_path=proj), pid)
    return proj, pid, db


def _insert(db, pid: str, desc: str, etype: str = "DECISION", h: str = "h1") -> int:
    from cognikernel.storage.connection import get_connection

    with get_connection(db) as conn:
        cur = conn.execute(
            "INSERT INTO events (project_id, session_id, created_at, event_type, "
            "payload, content_hash, weight, mention_count) VALUES (?,?,1,?,?,?,1.0,1)",
            (pid, "s", etype, json.dumps({"description": desc}), h),
        )
        conn.commit()
        return cur.lastrowid


def test_missing_project_silent(tmp_path: Path) -> None:
    assert recall_for_prompt(str(tmp_path / "no_project"), "any query") == ""


def test_empty_store_silent(tmp_path: Path, monkeypatch) -> None:
    proj, _, _ = _project(tmp_path, monkeypatch)
    assert recall_for_prompt(proj, "what database should we use") == ""


def test_exception_silent() -> None:
    """Never raises — silence on any error path."""
    with patch("cognikernel.integration.query._resolve", side_effect=RuntimeError("boom")):
        assert recall_for_prompt("/any/path", "query") == ""


def test_hard_constraint_can_inject(tmp_path: Path, monkeypatch) -> None:
    """J4 contract FLIP: the old type filter excluded CONSTRAINT_HARD on the
    false theory it's 'always in the block' — the block carries only a handful
    of the active constraints, and constraints are exactly what's worth
    pushing (the measured Redis/rate-limiter failure)."""
    proj, pid, db = _project(tmp_path, monkeypatch)
    _insert(db, pid, "rate limit counters live in Redis because multiple gateway "
                     "instances share one budget", etype="CONSTRAINT_HARD")
    result = recall_for_prompt(
        proj, "implement the rate limit counters for gateway instances")
    assert "Redis" in result


def test_bm25_only_needs_absolute_term_overlap(tmp_path: Path, monkeypatch) -> None:
    """Cold mode: rank alone is not enough — an off-topic prompt that happens
    to rank first in a tiny store must stay silent (< 3 shared content terms)."""
    proj, pid, db = _project(tmp_path, monkeypatch)
    _insert(db, pid, "use PostgreSQL for the persistence layer")
    assert recall_for_prompt(proj, "which database") == ""


def test_ledger_redundancy_filter(tmp_path: Path, monkeypatch) -> None:
    """An event already exposed to this session (block or earlier push) is
    never re-injected; a different session still gets it."""
    proj, pid, db = _project(tmp_path, monkeypatch)
    eid = _insert(db, pid, "rate limit counters live in Redis because multiple "
                           "gateway instances share one budget",
                  etype="CONSTRAINT_HARD")
    from cognikernel.storage.connection import get_connection
    from cognikernel.storage.render_ledger import record_rendered

    with get_connection(db) as conn:
        record_rendered(conn, pid, "sess-A", [eid], "block")

    prompt = "implement the rate limit counters for gateway instances"
    assert recall_for_prompt(proj, prompt, session_id="sess-A") == ""
    assert "Redis" in recall_for_prompt(proj, prompt, session_id="sess-B")


def test_ck1_injection_recorded_in_ledger(tmp_path: Path, monkeypatch) -> None:
    """A push is itself exposure: the same prompt twice injects only once."""
    proj, pid, db = _project(tmp_path, monkeypatch)
    _insert(db, pid, "rate limit counters live in Redis because multiple "
                     "gateway instances share one budget",
            etype="CONSTRAINT_HARD")
    prompt = "implement the rate limit counters for gateway instances"
    first = recall_for_prompt(proj, prompt, session_id="sess-A")
    second = recall_for_prompt(proj, prompt, session_id="sess-A")
    assert "Redis" in first
    assert second == ""


def test_max_events_cap(tmp_path: Path, monkeypatch) -> None:
    proj, pid, db = _project(tmp_path, monkeypatch)
    for i in range(5):
        _insert(db, pid, f"gateway rate limit counters budget variant {i} in Redis",
                h=f"h{i}")
    result = recall_for_prompt(
        proj, "implement the gateway rate limit counters budget")
    if result:
        # header + at most ck1_max_events lines
        assert len(result.splitlines()) <= 1 + 2
