"""CK-1 — recall_for_prompt: per-prompt injection candidate."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from memlora.integration.query import recall_for_prompt


def test_recall_for_prompt_returns_empty_for_missing_project(tmp_path: Path) -> None:
    result = recall_for_prompt(str(tmp_path / "no_project"), "any query")
    assert result == ""


def test_recall_for_prompt_returns_empty_when_no_hits(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path))
    monkeypatch.setattr("memlora.embedding.model.is_available", lambda: False)
    from memlora.integration.session import init_project
    proj = str(tmp_path / "proj")
    Path(proj).mkdir()
    init_project(proj)
    # Empty DB → no events → no hits → silence.
    result = recall_for_prompt(proj, "what database should we use")
    assert result == ""


def test_recall_for_prompt_returns_empty_on_exception() -> None:
    """Never raises — silence on any error path."""
    with patch("memlora.integration.query._resolve", side_effect=RuntimeError("boom")):
        assert recall_for_prompt("/any/path", "query") == ""


def test_recall_for_prompt_skips_always_injected_types(tmp_path: Path, monkeypatch) -> None:
    """CONSTRAINT_HARD and DO_NOT_RETRY are always in the static block; skip them."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path))
    monkeypatch.setattr("memlora.embedding.model.is_available", lambda: False)

    from memlora.config import Config
    from memlora.integration.session import init_project
    from memlora.storage.connection import get_connection, get_db_path, hash_project_path

    proj = str(tmp_path / "proj")
    Path(proj).mkdir()
    init_project(proj)
    pid = hash_project_path(proj)
    db = get_db_path(Config.load(project_path=proj), pid)

    with get_connection(db) as conn:
        conn.execute(
            "INSERT INTO events (project_id, session_id, created_at, event_type, "
            "payload, content_hash, weight, mention_count) VALUES (?,?,1,'CONSTRAINT_HARD',?,?,1.0,1)",
            (pid, "s", json.dumps({"description": "never delete the production database"}), "h1"),
        )
        conn.commit()

    # Regardless of lexical match: CONSTRAINT_HARD is always injected → filtered out.
    import dataclasses
    cfg = dataclasses.replace(Config.load(project_path=proj), query_injection_threshold=0.0)
    result = recall_for_prompt(proj, "delete the database", config=cfg)
    assert result == ""


def test_recall_for_prompt_respects_threshold_gate(tmp_path: Path, monkeypatch) -> None:
    """Threshold=1.0 → nothing can pass → silence even with matching content."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path))
    monkeypatch.setattr("memlora.embedding.model.is_available", lambda: False)

    from memlora.config import Config
    from memlora.integration.session import init_project
    from memlora.storage.connection import get_connection, get_db_path, hash_project_path

    proj = str(tmp_path / "proj")
    Path(proj).mkdir()
    init_project(proj)
    pid = hash_project_path(proj)
    db = get_db_path(Config.load(project_path=proj), pid)

    with get_connection(db) as conn:
        conn.execute(
            "INSERT INTO events (project_id, session_id, created_at, event_type, "
            "payload, content_hash, weight, mention_count) VALUES (?,?,1,'DECISION',?,?,1.0,1)",
            (pid, "s", json.dumps({"description": "use PostgreSQL for the database"}), "h1"),
        )
        conn.commit()

    import dataclasses
    cfg = dataclasses.replace(Config.load(project_path=proj), query_injection_threshold=1.0)
    result = recall_for_prompt(proj, "which database", config=cfg)
    assert result == ""
