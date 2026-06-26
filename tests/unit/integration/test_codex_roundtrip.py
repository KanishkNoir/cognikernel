"""End-to-end cross-platform round-trip (Sprint L / L6).

A decision made in a Codex session must reach the Claude SessionStart block:
  rollout (cwd=project) -> codex-sync -> worker extract -> handle_session_start.
The reverse direction (Codex reading Claude memory) already works via the shared
DB + the get_session_state MCP tool, so this gates the half L1-L4 added.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from memlora.config import Config


@pytest.fixture
def initialized(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMLORA_DISABLE_AUTO_WARM", "1")
    from memlora.integration.session import init_project

    proj = tmp_path / "proj"
    proj.mkdir()
    init_project(str(proj))
    codex_home = tmp_path / "codex"
    cfg = dataclasses.replace(Config.load(project_path=str(proj)), codex_home=codex_home)
    return proj, codex_home, cfg


def _write_rollout(codex_home: Path, cwd: str, sid: str, text: str) -> None:
    d = codex_home / "sessions" / "2026" / "06" / "21"
    d.mkdir(parents=True, exist_ok=True)
    recs = [
        {"type": "session_meta", "payload": {"id": sid, "cwd": cwd}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
                                              "content": [{"type": "input_text", "text": text}]}},
    ]
    (d / f"rollout-{sid}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in recs), encoding="utf-8"
    )


def test_codex_decision_reaches_claude_block(initialized) -> None:
    from memlora.integration.codex_sync import sync_codex_rollouts
    from memlora.integration.session import process_jobs
    from memlora.integration.session_start import handle_session_start

    proj, codex_home, cfg = initialized
    _write_rollout(codex_home, str(proj), "sid-pg",
                   "We decided to use Postgres for the primary datastore")

    # Codex -> store
    assert sync_codex_rollouts(str(proj), cfg)["captured"] == 1
    assert process_jobs(str(proj), config=cfg)["failed"] == 0

    # store -> Claude SessionStart block (handle_session_start does NOT re-sync,
    # so the temp codex_home is not needed here; the decision is already persisted)
    block = handle_session_start(str(proj), config=cfg)
    assert "Postgres" in block, f"Codex decision missing from the block:\n{block}"
