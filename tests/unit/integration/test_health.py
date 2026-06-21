"""Subsystem health checks + doctor --strict (audit P3 / #66).

These make fail-open degradation legible: a probe reports unhealthy instead of a
subsystem silently degrading, and `doctor --strict` turns that into a non-zero
exit for pre-flight/CI.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from memlora.config import Config
from memlora.integration.health import (
    check_embedding,
    check_schema_version,
    check_worker_queue,
    run_health_checks,
)
from memlora.storage.connection import get_connection, get_db_path, hash_project_path
from memlora.storage.evidence import store_evidence
from memlora.storage.jobs import enqueue_extraction, fail_job
from memlora.storage.migrations import run_migrations


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


class TestDoctorStrict:
    def _init(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "data"))
        monkeypatch.setenv("MEMLORA_DISABLE_AUTO_WARM", "1")
        from memlora.integration.session import init_project

        proj = tmp_path / "proj"
        proj.mkdir()
        init_project(str(proj))
        cfg = Config.load()
        pid = hash_project_path(str(proj))
        return proj, get_db_path(cfg, pid), pid

    def test_strict_exits_nonzero_when_degraded(self, tmp_path: Path, monkeypatch) -> None:
        from memlora.integration.cli import _cmd_doctor

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
        from memlora.integration.cli import _cmd_doctor

        proj, _, _ = self._init(tmp_path, monkeypatch)
        # Fresh project: no dead-letters, embedding disabled by config — healthy.
        _cmd_doctor(argparse.Namespace(project_path=str(proj), strict=True))  # no SystemExit

    def test_non_strict_never_exits_even_when_degraded(self, tmp_path: Path, monkeypatch) -> None:
        from memlora.integration.cli import _cmd_doctor

        proj, db, pid = self._init(tmp_path, monkeypatch)
        with get_connection(db) as conn:
            ev = store_evidence(conn, pid, "s", "transcript", b"x")
            jid = enqueue_extraction(conn, pid, "s", ev, "extract.transcript")
            for _ in range(3):
                fail_job(conn, jid, "EXTRACTOR_BUG", "boom")
        # Default doctor is informational: reports DEGRADED but exits 0.
        _cmd_doctor(argparse.Namespace(project_path=str(proj), strict=False))
