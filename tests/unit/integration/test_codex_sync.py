"""codex_sync ingest — cwd matching, fail-open, idempotency (Sprint L / L2)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from memlora.config import Config
from memlora.integration.codex_sync import codex_sessions_root, sync_codex_rollouts
from memlora.storage.connection import get_connection, get_db_path, hash_project_path


def _rollout(cwd: str, sid: str, user: str, assistant: str) -> str:
    lines = [
        {"type": "session_meta", "payload": {"id": sid, "cwd": cwd}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
                                              "content": [{"type": "input_text", "text": user}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                                              "content": [{"type": "output_text", "text": assistant}]}},
    ]
    return "\n".join(json.dumps(o) for o in lines)


def _write_rollout(sessions_root: Path, name: str, body: str) -> None:
    d = sessions_root / "2026" / "06" / "21"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(body, encoding="utf-8")


@pytest.fixture
def codex_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MEMLORA_DISABLE_AUTO_WARM", "1")
    codex_home = tmp_path / "codex"
    (codex_home / "sessions").mkdir(parents=True)
    project = tmp_path / "proj"
    project.mkdir()
    cfg = Config.load(project_path=str(project))
    cfg = __import__("dataclasses").replace(cfg, codex_home=codex_home)
    return project, codex_home / "sessions", cfg


class TestCodexSync:
    def test_root_resolution_prefers_config(self, codex_env):
        _, sessions, cfg = codex_env
        assert codex_sessions_root(cfg) == sessions

    def test_matching_cwd_is_captured(self, codex_env):
        project, sessions, cfg = codex_env
        _write_rollout(sessions, "rollout-a.jsonl",
                       _rollout(str(project), "sid-a", "Use Redis for caching.", "Agreed, Redis it is."))
        stats = sync_codex_rollouts(str(project), cfg)
        assert stats["matched"] == 1 and stats["captured"] == 1

        # evidence landed with the codex source_type
        db = get_db_path(cfg, hash_project_path(str(project)))
        with get_connection(db) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM raw_evidence WHERE source_type='codex_rollout'"
            ).fetchone()[0]
        assert n == 1

    def test_foreign_cwd_is_ignored(self, codex_env):
        project, sessions, cfg = codex_env
        _write_rollout(sessions, "rollout-x.jsonl",
                       _rollout("/some/other/project", "sid-x", "unrelated", "noted"))
        stats = sync_codex_rollouts(str(project), cfg)
        assert stats["scanned"] == 1 and stats["matched"] == 0 and stats["captured"] == 0

    def test_matching_project_identity_is_captured_across_checkout_paths(self, codex_env):
        project, sessions, cfg = codex_env
        cfg = __import__("dataclasses").replace(cfg, project_identity="shared-project")
        other_checkout = project.parent / "other-checkout"
        (other_checkout / ".memlora").mkdir(parents=True)
        (other_checkout / ".memlora" / "config.toml").write_text(
            'project_identity = "shared-project"\n',
            encoding="utf-8",
        )

        _write_rollout(sessions, "rollout-shared.jsonl",
                       _rollout(str(other_checkout), "sid-shared", "Use SQLite.", "ok"))
        stats = sync_codex_rollouts(str(project), cfg)
        assert stats["matched"] == 1 and stats["captured"] == 1

    def test_missing_sessions_dir_is_failopen(self, codex_env, tmp_path):
        project, _, cfg = codex_env
        cfg = __import__("dataclasses").replace(cfg, codex_home=tmp_path / "nonexistent")
        stats = sync_codex_rollouts(str(project), cfg)
        assert stats == {"scanned": 0, "matched": 0, "captured": 0, "jobs": 0, "enabled": True}

    def test_disabled_is_noop(self, codex_env):
        project, sessions, cfg = codex_env
        cfg = __import__("dataclasses").replace(cfg, codex_sync_enabled=False)
        _write_rollout(sessions, "rollout-a.jsonl",
                       _rollout(str(project), "sid-a", "Use Redis.", "ok"))
        stats = sync_codex_rollouts(str(project), cfg)
        assert stats["enabled"] is False and stats["captured"] == 0

    def test_resync_is_idempotent(self, codex_env):
        from memlora.integration.session import process_jobs

        project, sessions, cfg = codex_env
        _write_rollout(sessions, "rollout-a.jsonl",
                       _rollout(str(project), "sid-a", "Use Redis for caching.", "Agreed."))
        first = sync_codex_rollouts(str(project), cfg)
        process_jobs(str(project), config=cfg)   # worker extracts + advances the cursor
        second = sync_codex_rollouts(str(project), cfg)
        assert first["captured"] == 1
        # cursor advanced past the rollout -> the unchanged file yields no new content
        assert second["captured"] == 0
        # durable dedup: exactly one evidence row regardless of sync count
        db = get_db_path(cfg, hash_project_path(str(project)))
        with get_connection(db) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM raw_evidence WHERE source_type='codex_rollout'"
            ).fetchone()[0]
        assert n == 1

    def test_malformed_rollout_does_not_raise(self, codex_env):
        project, sessions, cfg = codex_env
        _write_rollout(sessions, "rollout-bad.jsonl", "not json at all\n{partial")
        stats = sync_codex_rollouts(str(project), cfg)  # must not raise
        assert stats["captured"] == 0
