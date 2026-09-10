"""Subsystem health checks + doctor --strict (audit P3 / #66).

These make fail-open degradation legible: a probe reports unhealthy instead of a
subsystem silently degrading, and `doctor --strict` turns that into a non-zero
exit for pre-flight/CI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from cognikernel.config import Config
from cognikernel.integration.health import (
    check_config,
    check_embedding,
    check_salience_head,
    check_schema_version,
    check_supersession_head,
    check_worker_queue,
    run_health_checks,
)
from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
from cognikernel.storage.evidence import store_evidence
from cognikernel.storage.jobs import enqueue_extraction, fail_job
from cognikernel.storage.migrations import run_migrations


def _migrated_db(tmp_path: Path) -> Path:
    db = tmp_path / "h.db"
    with get_connection(db) as conn:
        run_migrations(conn)
    return db


class TestHealthChecks:
    def test_fresh_db_is_all_healthy(self, tmp_path: Path) -> None:
        db = _migrated_db(tmp_path)
        with get_connection(db) as conn:
            checks = {c.name: c for c in run_health_checks(conn, "proj", Config())}
        assert checks["schema"].ok, checks["schema"].detail
        assert checks["worker_queue"].ok
        assert checks["embedding"].ok and "disabled" in checks["embedding"].detail
        # Declared deps — present in a properly provisioned environment.
        assert checks["fts5"].ok, checks["fts5"].detail
        assert checks["symbols"].ok, checks["symbols"].detail
        # Config() defaults to extractor="legacy" / cross_encoder_supersession=False,
        # so both encoder-head checks report "not requested" and are healthy.
        assert checks["salience_head"].ok and "not requested" in checks["salience_head"].detail
        assert checks["supersession_head"].ok and "not requested" in checks["supersession_head"].detail

    def test_salience_head_requested_but_not_installed_is_still_healthy(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Status-only (H-heads): a project asking for v2-broad before anyone has
        run `install-heads` (the default state for every freshly-init'd project)
        must never read as unhealthy — only the legacy-vs-fine-tuned status
        differs."""
        monkeypatch.setenv("COGNIKERNEL_V2_BODY_DIR", str(tmp_path / "nowhere"))
        check = check_salience_head(Config(extractor="v2-broad"))
        assert check.ok
        assert "not installed" in check.detail
        assert "install-heads" in check.detail

    def test_supersession_head_requested_but_not_installed_is_still_healthy(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("COGNIKERNEL_XENC_BODY_DIR", str(tmp_path / "nowhere"))
        check = check_supersession_head(Config(cross_encoder_supersession=True))
        assert check.ok
        assert "not installed" in check.detail
        assert "install-heads" in check.detail

    def test_salience_head_artifacts_present_reports_installed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Artifact detection is independent of whether onnxruntime/tokenizers
        (the `embedding` extra) happen to be importable in this environment —
        assert only on the branch that's environment-invariant."""
        body_dir = tmp_path / "salience_v2"
        body_dir.mkdir()
        (body_dir / "body.onnx").write_bytes(b"")
        (body_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("COGNIKERNEL_V2_BODY_DIR", str(body_dir))
        check = check_salience_head(Config(extractor="v2-broad"))
        assert check.ok
        assert "not installed" not in check.detail

    def test_schema_mismatch_is_unhealthy(self, tmp_path: Path) -> None:
        db = _migrated_db(tmp_path)
        with get_connection(db) as conn:
            conn.execute("UPDATE meta SET value='1' WHERE key='schema_version'")
            conn.commit()
            check = check_schema_version(conn)
        assert not check.ok
        assert "expected" in check.detail

    def test_dead_letter_is_unhealthy(self, tmp_path: Path) -> None:
        db = _migrated_db(tmp_path)
        with get_connection(db) as conn:
            ev = store_evidence(conn, "proj", "s", "transcript", b"x")
            jid = enqueue_extraction(conn, "proj", "s", ev, "extract.transcript")
            for _ in range(3):  # -> dead_lettered
                fail_job(conn, jid, "EXTRACTOR_BUG", "boom")
            check = check_worker_queue(conn, "proj")
        assert not check.ok
        assert "dead-letter" in check.detail

    def test_embedding_disabled_is_healthy(self) -> None:
        check = check_embedding(Config())  # embedding_enabled defaults False
        assert check.ok
        assert "disabled" in check.detail

    def test_config_typo_is_unhealthy(self, tmp_path: Path, monkeypatch) -> None:
        """H1: a typo'd project config degrades hooks silently (Config.load is
        fail-open) — the config check is what makes that degradation visible."""
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
        proj = tmp_path / "proj"
        (proj / ".cognikernel").mkdir(parents=True)
        (proj / ".cognikernel" / "config.toml").write_text(
            'hook_policy = "Strict"\n',  # case typo — not a valid policy
            encoding="utf-8",
        )
        check = check_config(str(proj))
        assert not check.ok
        assert "hook_policy" in check.detail

    def test_config_clean_is_healthy(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
        proj = tmp_path / "proj"
        (proj / ".cognikernel").mkdir(parents=True)
        (proj / ".cognikernel" / "config.toml").write_text(
            'hook_policy = "strict"\n', encoding="utf-8",
        )
        check = check_config(str(proj))
        assert check.ok


class TestDoctorStrict:
    def _init(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("COGNIKERNEL_DISABLE_AUTO_WARM", "1")
        from cognikernel.integration.session import init_project

        proj = tmp_path / "proj"
        proj.mkdir()
        init_project(str(proj))
        cfg = Config.load()
        pid = hash_project_path(str(proj))
        return proj, get_db_path(cfg, pid), pid

    def test_strict_exits_nonzero_when_degraded(self, tmp_path: Path, monkeypatch) -> None:
        from cognikernel.integration.cli import _cmd_doctor

        proj, db, pid = self._init(tmp_path, monkeypatch)
        with get_connection(db) as conn:
            ev = store_evidence(conn, pid, "s", "transcript", b"x")
            jid = enqueue_extraction(conn, pid, "s", ev, "extract.transcript")
            for _ in range(3):
                fail_job(conn, jid, "EXTRACTOR_BUG", "boom")

        with pytest.raises(SystemExit) as exc_info:
            _cmd_doctor(argparse.Namespace(project_path=str(proj), strict=True))
        assert exc_info.value.code == 1

    def test_strict_healthy_does_not_exit(self, tmp_path: Path, monkeypatch) -> None:
        from cognikernel.integration.cli import _cmd_doctor

        proj, _, _ = self._init(tmp_path, monkeypatch)
        # Fresh project: no dead-letters, embedding disabled by config — healthy.
        _cmd_doctor(argparse.Namespace(project_path=str(proj), strict=True))  # no SystemExit

    def test_non_strict_never_exits_even_when_degraded(self, tmp_path: Path, monkeypatch) -> None:
        from cognikernel.integration.cli import _cmd_doctor

        proj, db, pid = self._init(tmp_path, monkeypatch)
        with get_connection(db) as conn:
            ev = store_evidence(conn, pid, "s", "transcript", b"x")
            jid = enqueue_extraction(conn, pid, "s", ev, "extract.transcript")
            for _ in range(3):
                fail_job(conn, jid, "EXTRACTOR_BUG", "boom")
        # Default doctor is informational: reports DEGRADED but exits 0.
        _cmd_doctor(argparse.Namespace(project_path=str(proj), strict=False))

    def test_strict_exits_nonzero_on_config_typo(self, tmp_path: Path, monkeypatch) -> None:
        """The full H1 chain: a typo'd project config no longer kills the hooks
        (fail-open per key) AND doctor --strict now sees it and fails."""
        from cognikernel.integration.cli import _cmd_doctor

        proj, _, _ = self._init(tmp_path, monkeypatch)
        (proj / ".cognikernel").mkdir(exist_ok=True)
        (proj / ".cognikernel" / "config.toml").write_text(
            'extractor = "v3"\n',  # not a valid backend
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as exc_info:
            _cmd_doctor(argparse.Namespace(project_path=str(proj), strict=True))
        assert exc_info.value.code == 1


class TestDoctorRoundTripTelemetry:
    """S4 T-403: doctor surfaces the round-trips CogniKernel's own tools add, and
    names telemetry rows that were counted per transcript line instead of mixing
    them silently into corrected figures."""

    def _init(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("COGNIKERNEL_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("COGNIKERNEL_DISABLE_AUTO_WARM", "1")
        from cognikernel.integration.session import init_project

        proj = tmp_path / "proj"
        proj.mkdir()
        init_project(str(proj))
        pid = hash_project_path(str(proj))
        return proj, get_db_path(Config.load(), pid), pid

    @staticmethod
    def _row(pid: str, sid: str, **extra) -> dict:
        return {
            "project_id": pid, "session_id": sid, "input_tokens": 10,
            "cache_creation_tokens": 0, "cache_read_tokens": 100, "output_tokens": 5,
            **extra,
        }

    @staticmethod
    def _insert_legacy(conn, pid: str, sid: str) -> None:
        conn.execute(
            "INSERT INTO api_telemetry (project_id, session_id, input_tokens, "
            "cache_creation_tokens, cache_read_tokens, output_tokens, ingested_at, usage_basis) "
            "VALUES (?, ?, 30, 0, 9000, 15, 1, ?)",
            (pid, sid, "per_line_legacy"),
        )
        conn.commit()

    def test_prints_the_round_trip_share_and_its_breakdown(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        from cognikernel.integration.cli import _cmd_doctor
        from cognikernel.telemetry.ingest import store_telemetry

        proj, db, pid = self._init(tmp_path, monkeypatch)
        with get_connection(db) as conn:
            store_telemetry(conn, self._row(pid, "a", responses=10, memory_tool_responses=2,
                                            denied_responses=1, retried_denials=1))
            store_telemetry(conn, self._row(pid, "b", responses=30, memory_tool_responses=3,
                                            denied_responses=2, retried_denials=1))

        _cmd_doctor(argparse.Namespace(project_path=str(proj), strict=False))
        out = capsys.readouterr().out
        assert "API responses      : 40" in out
        assert "added by CK tools  : 25.0%" in out
        assert "memory-tool-only 5, denied 3, retried 2" in out
        assert "legacy rows" not in out

    def test_legacy_rows_are_named_and_kept_out_of_the_figures(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        from cognikernel.integration.cli import _cmd_doctor
        from cognikernel.telemetry.ingest import store_telemetry

        proj, db, pid = self._init(tmp_path, monkeypatch)
        with get_connection(db) as conn:
            store_telemetry(conn, self._row(pid, "fresh", responses=4))
            self._insert_legacy(conn, pid, "old")

        _cmd_doctor(argparse.Namespace(project_path=str(proj), strict=False))
        out = capsys.readouterr().out
        assert "sessions with data : 1" in out
        assert "cache reads served : 100 tok" in out
        assert "legacy rows        : 1 session(s)" in out

    def test_a_store_holding_only_legacy_rows_still_names_them(
        self, tmp_path: Path, monkeypatch, capsys
    ) -> None:
        """Otherwise doctor would report 'no telemetry' for a store that has rows,
        which reads like the ingest never ran."""
        from cognikernel.integration.cli import _cmd_doctor

        proj, db, pid = self._init(tmp_path, monkeypatch)
        with get_connection(db) as conn:
            self._insert_legacy(conn, pid, "old-1")
            self._insert_legacy(conn, pid, "old-2")

        _cmd_doctor(argparse.Namespace(project_path=str(proj), strict=False))
        out = capsys.readouterr().out
        assert "legacy rows        : 2 session(s)" in out
        assert "re-run 'cognikernel telemetry <project_path>'" in out
