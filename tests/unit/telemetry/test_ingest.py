"""Tests for cognikernel.telemetry.ingest — JSONL usage ingestion."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from cognikernel.config import Config
from cognikernel.integration.session import init_project, session_end
from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
from cognikernel.telemetry.ingest import (
    whole_session_rollup,
    ingest_session_jsonl,
    store_telemetry,
    get_cache_stats,
    find_and_ingest_telemetry,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _assistant_line(
    input_t: int = 100,
    cache_create: int = 0,
    cache_read: int = 0,
    output_t: int = 50,
) -> str:
    return json.dumps({
        "type": "assistant",
        "message": {
            "usage": {
                "input_tokens": input_t,
                "cache_creation_input_tokens": cache_create,
                "cache_read_input_tokens": cache_read,
                "output_tokens": output_t,
            }
        },
    })


def _make_jsonl(tmp_path: Path, name: str, lines: list[str]) -> Path:
    p = tmp_path / f"{name}.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(cognikernel_dir=tmp_path / "cognikernel")


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    p = tmp_path / "myproject"
    p.mkdir()
    return p


@pytest.fixture
def db_conn(project_path: Path, cfg: Config):
    init_project(project_path, config=cfg)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(cfg, project_id)
    with get_connection(db_path) as conn:
        yield conn, project_id


# ── ingest_session_jsonl ──────────────────────────────────────────────────────

class TestIngestSessionJsonl:
    def test_parses_single_message(self, tmp_path: Path) -> None:
        f = _make_jsonl(tmp_path, "s1", [_assistant_line(100, 0, 0, 50)])
        result = ingest_session_jsonl(f, "s1", "proj1")
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert result["cache_creation_tokens"] == 0
        assert result["cache_read_tokens"] == 0

    def test_sums_multiple_messages(self, tmp_path: Path) -> None:
        lines = [
            _assistant_line(100, 500, 0, 50),
            _assistant_line(5, 0, 2000, 30),
        ]
        f = _make_jsonl(tmp_path, "s1", lines)
        result = ingest_session_jsonl(f, "s1", "proj1")
        assert result["input_tokens"] == 105
        assert result["cache_creation_tokens"] == 500
        assert result["cache_read_tokens"] == 2000
        assert result["output_tokens"] == 80

    def test_ignores_non_assistant_lines(self, tmp_path: Path) -> None:
        lines = [
            json.dumps({"type": "user", "message": {"usage": {"input_tokens": 999}}}),
            _assistant_line(50, 0, 0, 25),
        ]
        f = _make_jsonl(tmp_path, "s1", lines)
        result = ingest_session_jsonl(f, "s1", "proj1")
        assert result["input_tokens"] == 50

    def test_skips_malformed_lines(self, tmp_path: Path) -> None:
        lines = ["not-json", _assistant_line(10, 0, 0, 5)]
        f = _make_jsonl(tmp_path, "s1", lines)
        result = ingest_session_jsonl(f, "s1", "proj1")
        assert result["input_tokens"] == 10

    def test_empty_file_returns_zeros(self, tmp_path: Path) -> None:
        f = _make_jsonl(tmp_path, "s1", [])
        result = ingest_session_jsonl(f, "s1", "proj1")
        assert result["input_tokens"] == 0
        assert result["cache_read_tokens"] == 0

    def test_missing_usage_field_skipped(self, tmp_path: Path) -> None:
        f = _make_jsonl(tmp_path, "s1", [json.dumps({"type": "assistant", "message": {}})])
        result = ingest_session_jsonl(f, "s1", "proj1")
        assert result["input_tokens"] == 0

    def test_returns_correct_ids(self, tmp_path: Path) -> None:
        f = _make_jsonl(tmp_path, "my-session", [_assistant_line(10, 0, 0, 5)])
        result = ingest_session_jsonl(f, "my-session", "my-project")
        assert result["session_id"] == "my-session"
        assert result["project_id"] == "my-project"

    def test_blank_lines_skipped(self, tmp_path: Path) -> None:
        lines = ["", "   ", _assistant_line(20, 0, 100, 10)]
        f = _make_jsonl(tmp_path, "s1", lines)
        result = ingest_session_jsonl(f, "s1", "proj1")
        assert result["input_tokens"] == 20
        assert result["cache_read_tokens"] == 100


# ── store_telemetry ───────────────────────────────────────────────────────────

class TestStoreTelemetry:
    def test_inserts_row(self, db_conn) -> None:
        conn, project_id = db_conn
        row = {
            "project_id": project_id,
            "session_id": "sess-abc",
            "input_tokens": 500,
            "cache_creation_tokens": 1000,
            "cache_read_tokens": 2000,
            "output_tokens": 100,
        }
        store_telemetry(conn, row)
        count = conn.execute(
            "SELECT COUNT(*) FROM api_telemetry WHERE session_id = 'sess-abc'"
        ).fetchone()[0]
        assert count == 1

    def test_upsert_replaces_on_duplicate(self, db_conn) -> None:
        conn, project_id = db_conn
        row = {
            "project_id": project_id,
            "session_id": "sess-dup",
            "input_tokens": 100,
            "cache_creation_tokens": 0,
            "cache_read_tokens": 0,
            "output_tokens": 50,
        }
        store_telemetry(conn, row)
        row2 = {**row, "input_tokens": 999, "cache_read_tokens": 5000}
        store_telemetry(conn, row2)
        saved = conn.execute(
            "SELECT input_tokens, cache_read_tokens FROM api_telemetry WHERE session_id='sess-dup'"
        ).fetchone()
        assert saved[0] == 999
        assert saved[1] == 5000

    def test_sets_ingested_at(self, db_conn) -> None:
        conn, project_id = db_conn
        before = int(time.time() * 1000)
        store_telemetry(conn, {
            "project_id": project_id, "session_id": "s-ts",
            "input_tokens": 1, "cache_creation_tokens": 0,
            "cache_read_tokens": 0, "output_tokens": 1,
        })
        after = int(time.time() * 1000)
        ts = conn.execute(
            "SELECT ingested_at FROM api_telemetry WHERE session_id='s-ts'"
        ).fetchone()[0]
        assert before <= ts <= after


# ── get_cache_stats ───────────────────────────────────────────────────────────

class TestGetCacheStats:
    def test_returns_zeros_for_empty_table(self, db_conn) -> None:
        conn, project_id = db_conn
        stats = get_cache_stats(conn, project_id)
        assert stats["sessions_with_data"] == 0
        assert stats["avg_cache_hit_rate"] == 0.0
        assert stats["total_cache_read_tokens"] == 0
        assert stats["effective_tokens_saved"] == 0

    def test_computes_cache_hit_rate(self, db_conn) -> None:
        conn, project_id = db_conn
        # 200 input, 800 cache_read → 80% hit rate
        store_telemetry(conn, {
            "project_id": project_id, "session_id": "s1",
            "input_tokens": 200, "cache_creation_tokens": 0,
            "cache_read_tokens": 800, "output_tokens": 50,
        })
        stats = get_cache_stats(conn, project_id)
        assert stats["sessions_with_data"] == 1
        assert abs(stats["avg_cache_hit_rate"] - 0.80) < 0.01

    def test_cache_creation_counts_against_hit_rate(self, db_conn) -> None:
        # 200 input, 200 cache_creation, 600 read → 600/1000 = 0.60 (creation is
        # NOT a cache hit). The old formula read/(input+read) would report 0.75.
        conn, project_id = db_conn
        store_telemetry(conn, {
            "project_id": project_id, "session_id": "s1",
            "input_tokens": 200, "cache_creation_tokens": 200,
            "cache_read_tokens": 600, "output_tokens": 50,
        })
        stats = get_cache_stats(conn, project_id)
        assert abs(stats["avg_cache_hit_rate"] - 0.60) < 0.01

    def test_cache_read_and_effective_saved(self, db_conn) -> None:
        conn, project_id = db_conn
        for i, (inp, read) in enumerate([(100, 500), (200, 1000)]):
            store_telemetry(conn, {
                "project_id": project_id, "session_id": f"s{i}",
                "input_tokens": inp, "cache_creation_tokens": 0,
                "cache_read_tokens": read, "output_tokens": 50,
            })
        stats = get_cache_stats(conn, project_id)
        assert stats["total_cache_read_tokens"] == 1500
        # cache_read billed ~0.1x → ~90% effective saving.
        assert stats["effective_tokens_saved"] == 1350

    def test_recent_sessions_limited_to_ten(self, db_conn) -> None:
        conn, project_id = db_conn
        for i in range(15):
            store_telemetry(conn, {
                "project_id": project_id, "session_id": f"s{i:03d}",
                "input_tokens": 100, "cache_creation_tokens": 0,
                "cache_read_tokens": 50, "output_tokens": 30,
            })
        stats = get_cache_stats(conn, project_id)
        assert len(stats["recent_sessions"]) <= 10


# ── whole_session_rollup ──────────────────────────────────────────────────────

class TestWholeSessionRollup:
    def test_empty(self, db_conn) -> None:
        conn, project_id = db_conn
        roll = whole_session_rollup(conn, project_id)
        assert roll["sessions_with_data"] == 0
        assert roll["totals"] == {"input": 0, "cache_creation": 0, "cache_read": 0, "output": 0}

    def test_sums_and_billed_equivalent(self, db_conn) -> None:
        conn, project_id = db_conn
        store_telemetry(conn, {
            "project_id": project_id, "session_id": "s1",
            "input_tokens": 1000, "cache_creation_tokens": 400,
            "cache_read_tokens": 2000, "output_tokens": 300,
        })
        roll = whole_session_rollup(conn, project_id)
        assert roll["sessions_with_data"] == 1
        assert roll["totals"]["cache_read"] == 2000
        # 1000 + round(1.25*400) + round(0.1*2000) = 1000 + 500 + 200 = 1700
        assert roll["billed_equivalent_input_tokens"] == 1700
        assert len(roll["sessions"]) == 1


# ── find_and_ingest_telemetry ─────────────────────────────────────────────────

class TestFindAndIngestTelemetry:
    """Tests for the high-level scan-and-ingest function.

    Uses the injectable claude_projects_dir parameter so tests never touch ~/.claude/.
    """

    def _seed_session(
        self,
        project_path: Path,
        cfg: Config,
        session_id: str,
    ) -> None:
        """Insert a minimal event row so session_id appears in the events table."""
        from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
        from cognikernel.storage.events import Event, insert_event

        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            insert_event(conn, Event(
                project_id=project_id,
                session_id=session_id,
                event_type="DECISION",
                payload={"description": "seed"},
                content_hash=f"hash-{session_id}",
            ))

    def test_db_does_not_exist_returns_zeros(self, tmp_path: Path) -> None:
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        project_path = tmp_path / "proj"
        project_path.mkdir()
        # DB never initialised — should return zeros, not raise
        result = find_and_ingest_telemetry(project_path, config=cfg)
        assert result == {"ingested": 0, "skipped": 0, "total_sessions_known": 0}

    def test_no_events_returns_zeros(self, tmp_path: Path) -> None:
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        project_path = tmp_path / "proj"
        project_path.mkdir()
        init_project(project_path, config=cfg)
        # DB exists but has no events
        result = find_and_ingest_telemetry(project_path, config=cfg)
        assert result == {"ingested": 0, "skipped": 0, "total_sessions_known": 0}

    def test_claude_projects_dir_missing_all_skipped(self, tmp_path: Path) -> None:
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        project_path = tmp_path / "proj"
        project_path.mkdir()
        init_project(project_path, config=cfg)
        self._seed_session(project_path, cfg, "sess-aaa")
        # Point to a non-existent dir — nothing to scan
        missing_dir = tmp_path / "nonexistent_claude_projects"
        result = find_and_ingest_telemetry(
            project_path, config=cfg, claude_projects_dir=missing_dir
        )
        assert result["total_sessions_known"] == 1
        assert result["ingested"] == 0
        assert result["skipped"] == 1

    def test_matching_jsonl_is_ingested(self, tmp_path: Path) -> None:
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        project_path = tmp_path / "proj"
        project_path.mkdir()
        init_project(project_path, config=cfg)
        session_id = "sess-match-123"
        self._seed_session(project_path, cfg, session_id)

        # Create a fake Claude projects dir with a matching JSONL
        claude_dir = tmp_path / "claude_projects" / "proj_hash"
        claude_dir.mkdir(parents=True)
        jsonl_content = "\n".join([
            _assistant_line(200, 1000, 800, 100),
            _assistant_line(50, 0, 200, 30),
        ])
        (claude_dir / f"{session_id}.jsonl").write_text(jsonl_content, encoding="utf-8")

        result = find_and_ingest_telemetry(
            project_path, config=cfg, claude_projects_dir=tmp_path / "claude_projects"
        )
        assert result["total_sessions_known"] == 1
        assert result["ingested"] == 1
        assert result["skipped"] == 0

        # Verify data was written to the DB
        from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            row = conn.execute(
                "SELECT input_tokens, cache_read_tokens FROM api_telemetry WHERE session_id=?",
                (session_id,),
            ).fetchone()
        assert row is not None
        assert row["input_tokens"] == 250   # 200 + 50
        assert row["cache_read_tokens"] == 1000  # 800 + 200

    def test_mixed_some_match_some_skip(self, tmp_path: Path) -> None:
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        project_path = tmp_path / "proj"
        project_path.mkdir()
        init_project(project_path, config=cfg)
        self._seed_session(project_path, cfg, "sess-found")
        self._seed_session(project_path, cfg, "sess-missing")

        claude_dir = tmp_path / "claude_projects" / "any_hash"
        claude_dir.mkdir(parents=True)
        # Only provide JSONL for sess-found
        (claude_dir / "sess-found.jsonl").write_text(
            _assistant_line(100, 0, 500, 50), encoding="utf-8"
        )

        result = find_and_ingest_telemetry(
            project_path, config=cfg, claude_projects_dir=tmp_path / "claude_projects"
        )
        assert result["total_sessions_known"] == 2
        assert result["ingested"] == 1
        assert result["skipped"] == 1

    def test_idempotent_reingest(self, tmp_path: Path) -> None:
        cfg = Config(cognikernel_dir=tmp_path / "cognikernel")
        project_path = tmp_path / "proj"
        project_path.mkdir()
        init_project(project_path, config=cfg)
        session_id = "sess-idem"
        self._seed_session(project_path, cfg, session_id)

        claude_dir = tmp_path / "claude_projects" / "h"
        claude_dir.mkdir(parents=True)
        (claude_dir / f"{session_id}.jsonl").write_text(
            _assistant_line(100, 0, 400, 50), encoding="utf-8"
        )

        kwargs = {"config": cfg, "claude_projects_dir": tmp_path / "claude_projects"}
        find_and_ingest_telemetry(project_path, **kwargs)
        find_and_ingest_telemetry(project_path, **kwargs)  # second call

        from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
        project_id = hash_project_path(project_path)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM api_telemetry WHERE project_id=?", (project_id,)
            ).fetchone()[0]
        assert count == 1  # upsert, not duplicate insert
