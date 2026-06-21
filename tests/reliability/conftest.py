"""Shared fixtures for the reliability / failure-injection suite (audit P2 / #65).

This category exists because the rest of the suite runs the happy path in-process
against a fresh tmp DB — it never kills a worker mid-merge, crashes a migration,
or feeds the pipeline malformed input. Every audit P1 lived in exactly those
paths. These fixtures stand up a real, initialized project so tests can drive the
actual worker (process_jobs), the real lock, and real evidence through their
failure modes.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from memlora.config import Config
from memlora.storage.connection import get_db_path, hash_project_path


@dataclass
class ProjectCtx:
    path: str
    db: Path
    pid: str
    cfg: Config


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> ProjectCtx:
    """A fully-initialized project with an isolated MEMLORA_DIR (no model warm)."""
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "data"))
    os.environ.setdefault("MEMLORA_DISABLE_AUTO_WARM", "1")
    from memlora.integration.session import init_project

    proj = tmp_path / "proj"
    proj.mkdir()
    init_project(str(proj))
    cfg = Config.load(project_path=str(proj))
    pid = hash_project_path(str(proj))
    return ProjectCtx(path=str(proj), db=get_db_path(cfg, pid), pid=pid, cfg=cfg)


@pytest.fixture
def jsonl():
    """Factory: build a JSONL transcript with `n` user decision lines."""
    def _build(n: int) -> str:
        lines = [
            json.dumps({"type": "user",
                        "message": {"content": f"User: We decided to use tool D{i} for subsystem S{i}"}})
            for i in range(n)
        ]
        return "\n".join(lines) + "\n"
    return _build
