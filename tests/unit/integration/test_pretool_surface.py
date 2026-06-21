"""K2 — surface_prohibitions_for_edit: JIT bind at the action point.

The gate mirrors CK-1's BM25-only arm but the candidate pool is type-restricted
to prohibitions (graveyard + hard constraints), and the trigger is an edit diff
rather than a prompt. Surfacing is advisory; the hook attaches it to an `allow`.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from memlora.integration.query import surface_prohibitions_for_edit


def _project(tmp_path: Path, monkeypatch) -> tuple[str, str, object]:
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path))
    monkeypatch.setattr("memlora.embedding.model.is_ready", lambda: False)
    monkeypatch.setattr("memlora.embedding.model.warm", lambda: None)
    from memlora.config import Config
    from memlora.integration.session import init_project
    from memlora.storage.connection import get_db_path, hash_project_path

    proj = str(tmp_path / "proj")
    Path(proj).mkdir()
    init_project(proj)
    pid = hash_project_path(proj)
    db = get_db_path(Config.load(project_path=proj), pid)
    return proj, pid, db


def _insert(db, pid: str, desc: str, etype: str, h: str = "h1",
            subject: str = "", rationale: str = "") -> int:
    from memlora.storage.connection import get_connection

    payload = {"description": desc, "subject": subject}
    if rationale:
        payload["rationale"] = rationale
    with get_connection(db) as conn:
        cur = conn.execute(
            "INSERT INTO events (project_id, session_id, created_at, event_type, "
            "payload, content_hash, weight, mention_count) VALUES (?,?,1,?,?,?,1.0,1)",
            (pid, "s", etype, json.dumps(payload), h),
        )
        conn.commit()
        return cur.lastrowid


def test_missing_project_silent(tmp_path: Path) -> None:
    assert surface_prohibitions_for_edit(str(tmp_path / "no"), "x = 1") == ""


def test_empty_diff_silent(tmp_path: Path, monkeypatch) -> None:
    proj, _, _ = _project(tmp_path, monkeypatch)
    assert surface_prohibitions_for_edit(proj, "   ") == ""


def test_exception_silent() -> None:
    with patch("memlora.integration.query._resolve", side_effect=RuntimeError("boom")):
        assert surface_prohibitions_for_edit("/any", "in-process counter += 1") == ""


def test_graveyard_surfaces_for_contradicting_edit(tmp_path: Path, monkeypatch) -> None:
    """The named Relay D5/D16 regression: an edit adding an in-process counter
    must receive the Redis prohibition."""
    proj, pid, db = _project(tmp_path, monkeypatch)
    _insert(db, pid,
            "do not use in-process rate limit counters; use Redis for the shared budget",
            etype="APPROACH_ABANDONED_DO_NOT_RETRY", subject="rate limiting",
            rationale="multiple gateway instances share one budget")
    diff = "self._counter = 0  # in-process rate limit counter for the gateway budget"
    out = surface_prohibitions_for_edit(proj, diff)
    assert "ruled this out" in out
    assert "Redis" in out
    assert "Re-affirm" in out


def test_plain_decision_not_surfaced(tmp_path: Path, monkeypatch) -> None:
    """Type restriction: an ordinary DECISION on the same topic is not a
    prohibition and must not fire."""
    proj, pid, db = _project(tmp_path, monkeypatch)
    _insert(db, pid, "we use a Redis token bucket for rate limit counters",
            etype="DECISION", subject="rate limiting")
    diff = "self._counter = 0  # in-process rate limit counter for the gateway budget"
    assert surface_prohibitions_for_edit(proj, diff) == ""


def test_hard_prohibition_with_marker_surfaces(tmp_path: Path, monkeypatch) -> None:
    """A CONSTRAINT_HARD carrying a negative marker IS a bind target (Conductor
    money-float class): a float-introducing edit must surface it."""
    proj, pid, db = _project(tmp_path, monkeypatch)
    _insert(db, pid, "never store money values as float; money columns are integer cents",
            etype="CONSTRAINT_HARD", subject="money type")
    diff = "amount = float(cents) / 100  # money value stored as float column"
    out = surface_prohibitions_for_edit(proj, diff)
    assert "integer cents" in out


def test_positive_hard_rule_not_surfaced(tmp_path: Path, monkeypatch) -> None:
    """A CONSTRAINT_HARD with NO negative marker is a block fact, not an
    action-point prohibition — it must not fire even on vocabulary overlap
    (this is what drove the ~83% surface rate before the marker filter)."""
    proj, pid, db = _project(tmp_path, monkeypatch)
    _insert(db, pid, "TargetConfig is a frozen dataclass with name, weight and url fields",
            etype="CONSTRAINT_HARD", subject="config schema")
    diff = "class TargetConfig:  # name weight url frozen dataclass config fields"
    assert surface_prohibitions_for_edit(proj, diff) == ""


def test_priority_surfaces_architecture_over_impldetail(tmp_path: Path, monkeypatch) -> None:
    """#56 core fix: when a diff matches BOTH a token-dense impl-detail prohibition
    AND a lower-overlap architecture prohibition, the architecture one must surface
    (authority+scope ranking beats raw overlap). This is the live D5/D16 miss."""
    proj, pid, db = _project(tmp_path, monkeypatch)
    # impl-detail graveyard: HIGH overlap with the diff, no architecture scope
    _insert(db, pid, "RPM check sits outside the attempt loop — one count per request "
                     "with rate limit counter increment",
            etype="APPROACH_ABANDONED_DO_NOT_RETRY", subject="rpm loop", h="impl")
    # architecture graveyard: lower overlap, but about WHERE state lives
    _insert(db, pid, "Local counters don't work — with N instances each enforcing the "
                     "full limit independently the shared rate limit breaks",
            etype="APPROACH_ABANDONED_DO_NOT_RETRY", subject="rate limiting", h="arch")
    diff = ("class RateLimiter:  # sliding-window rpm/tpm limiter, one instance per "
            "deployment; counters live in a dict, rate limit count per request loop")
    out = surface_prohibitions_for_edit(proj, diff)
    assert "Local counters don't work" in out   # architecture prohibition wins
    assert "instances" in out


def test_term_overlap_floor(tmp_path: Path, monkeypatch) -> None:
    """An unrelated edit that happens to rank first in a tiny store stays silent
    (< pretool_min_term_overlap shared content terms)."""
    proj, pid, db = _project(tmp_path, monkeypatch)
    _insert(db, pid, "never store money values as floating point; use integer cents",
            etype="CONSTRAINT_HARD", subject="money type")
    assert surface_prohibitions_for_edit(proj, "logger.info('startup complete')") == ""


def test_ledger_dedup_across_session(tmp_path: Path, monkeypatch) -> None:
    """Already surfaced this session (any channel) → not repeated; a fresh
    session still gets it."""
    proj, pid, db = _project(tmp_path, monkeypatch)
    eid = _insert(db, pid,
                  "do not use in-process rate limit counters; use Redis shared budget",
                  etype="APPROACH_ABANDONED_DO_NOT_RETRY", subject="rate limiting")
    from memlora.storage.connection import get_connection
    from memlora.storage.render_ledger import record_rendered

    with get_connection(db) as conn:
        record_rendered(conn, pid, "sess-A", [eid], "block")

    diff = "self._counter = 0  # in-process rate limit counter budget"
    assert surface_prohibitions_for_edit(proj, diff, session_id="sess-A") == ""
    assert "Redis" in surface_prohibitions_for_edit(proj, diff, session_id="sess-B")


def test_surface_recorded_on_pretool_channel(tmp_path: Path, monkeypatch) -> None:
    """A surface is itself exposure: same diff twice fires only once."""
    proj, pid, db = _project(tmp_path, monkeypatch)
    _insert(db, pid,
            "do not use in-process rate limit counters; use Redis shared budget",
            etype="APPROACH_ABANDONED_DO_NOT_RETRY", subject="rate limiting")
    diff = "self._counter = 0  # in-process rate limit counter budget"
    first = surface_prohibitions_for_edit(proj, diff, session_id="sess-A")
    second = surface_prohibitions_for_edit(proj, diff, session_id="sess-A")
    assert "Redis" in first
    assert second == ""


def test_disabled_by_config(tmp_path: Path, monkeypatch) -> None:
    proj, pid, db = _project(tmp_path, monkeypatch)
    _insert(db, pid, "do not use in-process counters; use Redis",
            etype="APPROACH_ABANDONED_DO_NOT_RETRY", subject="rate limiting")
    from memlora.config import Config

    cfg = Config.load(project_path=proj)
    object.__setattr__(cfg, "pretool_prohibition_surface_enabled", False)
    diff = "self._counter = 0  # in-process rate limit counter budget"
    assert surface_prohibitions_for_edit(proj, diff, config=cfg) == ""
