"""Codex rollout capture must survive malformed input and stay idempotent through
the REAL worker (Sprint L / L5).

The unit tests prove the converter + sync in isolation; this drives a genuinely
initialized project through claim -> slice -> extract -> merge with the Codex
source_type, the path that only exists once cross-platform capture is wired.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from cognikernel.storage.connection import get_connection


def _event_stats(db) -> tuple[int, int, float]:
    with get_connection(db) as conn:
        row = conn.execute(
            "SELECT COUNT(*) c, COALESCE(SUM(mention_count),0) m, "
            "COALESCE(SUM(weight),0.0) w FROM events"
        ).fetchone()
    return row["c"], row["m"], round(row["w"], 6)


def _write_rollout(codex_home: Path, cwd: str, sid: str, user_lines, *, prepend_garbage=False) -> None:
    d = codex_home / "sessions" / "2026" / "06" / "21"
    d.mkdir(parents=True, exist_ok=True)
    recs = [json.dumps({"type": "session_meta", "payload": {"id": sid, "cwd": cwd}})]
    for u in user_lines:
        recs.append(json.dumps({"type": "response_item", "payload": {
            "type": "message", "role": "user",
            "content": [{"type": "input_text", "text": u}]}}))
    body = "\n".join(recs)
    if prepend_garbage:
        body = "this is not json\n{partial json\n" + body + "\n{also broken"
    (d / f"rollout-{sid}.jsonl").write_text(body, encoding="utf-8")


class TestCodexRolloutReliability:
    def test_malformed_rollout_drains_without_crash(self, project, tmp_path) -> None:
        from cognikernel.integration.codex_sync import sync_codex_rollouts
        from cognikernel.integration.session import process_jobs

        codex_home = tmp_path / "codex"
        cfg = dataclasses.replace(project.cfg, codex_home=codex_home)
        _write_rollout(
            codex_home, project.path, "sid-bad",
            ["We decided to use tool D1 for subsystem S1"],
            prepend_garbage=True,
        )
        captured = sync_codex_rollouts(project.path, cfg)
        assert captured["captured"] == 1                 # garbage tolerated, decision kept
        summary = process_jobs(project.path, config=cfg)
        assert summary["failed"] == 0                    # extraction did not choke
        assert _event_stats(project.db)[0] > 0           # the valid decision survived

    def test_resync_through_real_worker_is_idempotent(self, project, tmp_path) -> None:
        from cognikernel.integration.codex_sync import sync_codex_rollouts
        from cognikernel.integration.session import process_jobs

        codex_home = tmp_path / "codex"
        cfg = dataclasses.replace(project.cfg, codex_home=codex_home)
        _write_rollout(codex_home, project.path, "sid-1",
                       ["We decided to use tool D2 for subsystem S2"])

        sync_codex_rollouts(project.path, cfg)
        process_jobs(project.path, config=cfg)
        baseline = _event_stats(project.db)
        assert baseline[0] > 0

        # Re-sync the unchanged rollout + drain again: cursor + provenance guard
        # make it a no-op — no extra events, no mention_count/decay drift.
        sync_codex_rollouts(project.path, cfg)
        process_jobs(project.path, config=cfg)
        assert _event_stats(project.db) == baseline, "codex re-sync drifted state"
