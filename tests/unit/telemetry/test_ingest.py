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


def _block_line(
    message_id: str,
    block_type: str = "text",
    input_t: int = 100,
    cache_create: int = 0,
    cache_read: int = 0,
    output_t: int = 50,
) -> str:
    """One JSONL line as Claude Code writes it: a SINGLE content block of a response.

    A response with several blocks (text, thinking, tool_use) is written as several
    lines that share `message.id` and repeat the response's full `usage` on each —
    the usage is per response, not per block.
    """
    return json.dumps({
        "type": "assistant",
        "message": {
            "id": message_id,
            "content": [{"type": block_type}],
            "usage": {
                "input_tokens": input_t,
                "cache_creation_input_tokens": cache_create,
                "cache_read_input_tokens": cache_read,
                "output_tokens": output_t,
            },
        },
    })


def _tool_use_line(message_id: str, name: str, tool_id: str, tool_input: dict) -> str:
    """One response line carrying a single tool_use block."""
    return json.dumps({
        "type": "assistant",
        "message": {
            "id": message_id,
            "content": [{"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}],
            "usage": {"input_tokens": 1, "cache_creation_input_tokens": 0,
                      "cache_read_input_tokens": 100, "output_tokens": 10},
        },
    })


def _tool_result_line(tool_id: str, content, is_error: bool = False) -> str:
    """The user line Claude Code writes with a tool's result."""
    return json.dumps({
        "type": "user",
        "message": {"content": [{
            "type": "tool_result", "tool_use_id": tool_id,
            "content": content, "is_error": is_error,
        }]},
    })


_DENIAL = ("[CogniKernel] app/main.py signatures are listed in the Codebase skeleton "
           "section of your session context. Use them.")
_RECALL = "mcp__cognikernel__recall"


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


class TestIngestCountsEachResponseOnce:
    """Claude Code writes one JSONL line per content block and repeats the response's
    usage on every one of them. Summing per line counted a response once per block —
    2-3x on real sessions, and by a different factor per session depending on how many
    blocks responses had — which inflated every api_telemetry row and every token
    figure derived from transcripts. Verified on real transcripts: 1,077 of 1,077
    multi-line responses carried identical usage on every line.
    """

    def test_usage_repeated_on_every_block_line_counts_once(self, tmp_path: Path) -> None:
        lines = [
            _block_line("msg_1", "text", 3, 1200, 40000, 800),
            _block_line("msg_1", "thinking", 3, 1200, 40000, 800),
            _block_line("msg_1", "tool_use", 3, 1200, 40000, 800),
        ]
        result = ingest_session_jsonl(_make_jsonl(tmp_path, "s1", lines), "s1", "p")
        assert result["input_tokens"] == 3
        assert result["cache_creation_tokens"] == 1200
        assert result["cache_read_tokens"] == 40000
        assert result["output_tokens"] == 800

    def test_distinct_responses_are_summed(self, tmp_path: Path) -> None:
        lines = [
            _block_line("msg_1", "text", 10, 100, 1000, 50),
            _block_line("msg_1", "tool_use", 10, 100, 1000, 50),
            _block_line("msg_2", "text", 20, 200, 2000, 60),
        ]
        result = ingest_session_jsonl(_make_jsonl(tmp_path, "s1", lines), "s1", "p")
        assert result["input_tokens"] == 30
        assert result["cache_creation_tokens"] == 300
        assert result["cache_read_tokens"] == 3000
        assert result["output_tokens"] == 110

    def test_lines_without_a_message_id_each_count(self, tmp_path: Path) -> None:
        """Older or hand-built transcripts may carry no id. With nothing to group
        on, each line is its own response — the previous behaviour, kept."""
        lines = [_assistant_line(100, 0, 0, 50), _assistant_line(100, 0, 0, 50)]
        result = ingest_session_jsonl(_make_jsonl(tmp_path, "s1", lines), "s1", "p")
        assert result["input_tokens"] == 200
        assert result["output_tokens"] == 100

    def test_reports_the_number_of_api_responses(self, tmp_path: Path) -> None:
        lines = [
            _block_line("msg_1", "text"),
            _block_line("msg_1", "thinking"),
            _block_line("msg_1", "tool_use"),
            _block_line("msg_2", "text"),
            _assistant_line(1, 0, 0, 1),
        ]
        result = ingest_session_jsonl(_make_jsonl(tmp_path, "s1", lines), "s1", "p")
        assert result["responses"] == 3

    def test_interleaved_user_lines_do_not_split_a_response(self, tmp_path: Path) -> None:
        """A tool_result user line lands between blocks of the same response in
        real transcripts; grouping is by id, not by adjacency."""
        lines = [
            _block_line("msg_1", "tool_use", 5, 0, 500, 40),
            json.dumps({"type": "user", "message": {"content": [{"type": "tool_result"}]}}),
            _block_line("msg_1", "text", 5, 0, 500, 40),
        ]
        result = ingest_session_jsonl(_make_jsonl(tmp_path, "s1", lines), "s1", "p")
        assert result["output_tokens"] == 40
        assert result["responses"] == 1


class TestIngestClassifiesRoundTrips:
    """Round-trips CogniKernel's own tool surface adds (S4 T-403).

    Measured on the four-project benchmark, these explained more than all of
    Relay's +23% cost over an agent with no memory: memory-tool-only responses,
    reads denied by the PreToolUse gate, and retries of those denied reads. An
    agent with no memory makes none of them, so they are the G1 instrument.
    Each response lands in at most one class.
    """

    def _ingest(self, tmp_path: Path, lines: list[str]) -> dict:
        return ingest_session_jsonl(_make_jsonl(tmp_path, "s1", lines), "s1", "p")

    def test_usage_basis_is_per_response(self, tmp_path: Path) -> None:
        assert self._ingest(tmp_path, [_block_line("m1")])["usage_basis"] == "per_response"

    def test_memory_tool_only_response_is_counted(self, tmp_path: Path) -> None:
        r = self._ingest(tmp_path, [_tool_use_line("m1", _RECALL, "t1", {"query": "auth"})])
        assert r["memory_tool_responses"] == 1

    def test_a_response_mixing_memory_and_other_tools_is_not_memory_only(
        self, tmp_path: Path
    ) -> None:
        lines = [
            _tool_use_line("m1", _RECALL, "t1", {"query": "auth"}),
            _tool_use_line("m1", "Read", "t2", {"file_path": "a.py"}),
        ]
        assert self._ingest(tmp_path, lines)["memory_tool_responses"] == 0

    def test_a_response_with_no_tools_is_not_memory_only(self, tmp_path: Path) -> None:
        assert self._ingest(tmp_path, [_block_line("m1", "text")])["memory_tool_responses"] == 0

    def test_a_multi_line_memory_response_is_counted_once(self, tmp_path: Path) -> None:
        lines = [_block_line("m1", "text"), _tool_use_line("m1", _RECALL, "t1", {"query": "x"})]
        assert self._ingest(tmp_path, lines)["memory_tool_responses"] == 1

    def test_a_cognikernel_denial_is_counted(self, tmp_path: Path) -> None:
        lines = [
            _tool_use_line("m1", "Read", "t1", {"file_path": "app/main.py"}),
            _tool_result_line("t1", _DENIAL, is_error=True),
        ]
        r = self._ingest(tmp_path, lines)
        assert r["denied_responses"] == 1
        assert r["retried_denials"] == 0

    def test_a_denial_with_list_shaped_content_is_counted(self, tmp_path: Path) -> None:
        lines = [
            _tool_use_line("m1", "Read", "t1", {"file_path": "app/main.py"}),
            _tool_result_line("t1", [{"type": "text", "text": _DENIAL}], is_error=True),
        ]
        assert self._ingest(tmp_path, lines)["denied_responses"] == 1

    def test_an_ordinary_tool_error_is_not_a_denial(self, tmp_path: Path) -> None:
        lines = [
            _tool_use_line("m1", "Read", "t1", {"file_path": "missing.py"}),
            _tool_result_line("t1", "File does not exist.", is_error=True),
        ]
        assert self._ingest(tmp_path, lines)["denied_responses"] == 0

    def test_a_failed_memory_tool_call_is_not_a_denial(self, tmp_path: Path) -> None:
        """Only the PreToolUse gate's marked message is a denial. A cognikernel MCP
        tool that errors is still a memory-tool round-trip, not a denied read."""
        lines = [
            _tool_use_line("m1", _RECALL, "t1", {"query": "x"}),
            _tool_result_line("t1", "Error calling cognikernel recall: timeout", is_error=True),
        ]
        r = self._ingest(tmp_path, lines)
        assert r["denied_responses"] == 0
        assert r["memory_tool_responses"] == 1

    def test_reissuing_a_denied_call_is_a_retry(self, tmp_path: Path) -> None:
        lines = [
            _tool_use_line("m1", "Read", "t1", {"file_path": "app/main.py"}),
            _tool_result_line("t1", _DENIAL, is_error=True),
            _tool_use_line("m2", "Read", "t2", {"file_path": "app/main.py"}),
            _tool_result_line("t2", "def go(): ..."),
        ]
        r = self._ingest(tmp_path, lines)
        assert r["denied_responses"] == 1
        assert r["retried_denials"] == 1
        assert r["responses"] == 2

    def test_a_different_call_after_a_denial_is_not_a_retry(self, tmp_path: Path) -> None:
        lines = [
            _tool_use_line("m1", "Read", "t1", {"file_path": "app/main.py"}),
            _tool_result_line("t1", _DENIAL, is_error=True),
            _tool_use_line("m2", "Read", "t2", {"file_path": "app/other.py"}),
        ]
        assert self._ingest(tmp_path, lines)["retried_denials"] == 0


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


# ── round-trip persistence and doctor stats (S4 T-403) ────────────────────────

class TestRoundTripPersistenceAndStats:
    def test_store_telemetry_persists_round_trip_columns(self, db_conn) -> None:
        conn, project_id = db_conn
        store_telemetry(conn, {
            "project_id": project_id, "session_id": "s-rt",
            "input_tokens": 1, "cache_creation_tokens": 0, "cache_read_tokens": 0,
            "output_tokens": 1, "responses": 10, "memory_tool_responses": 2,
            "denied_responses": 1, "retried_denials": 1, "usage_basis": "per_response",
        })
        row = conn.execute(
            "SELECT responses, memory_tool_responses, denied_responses, retried_denials, "
            "usage_basis FROM api_telemetry WHERE session_id = 's-rt'"
        ).fetchone()
        assert tuple(row) == (10, 2, 1, 1, "per_response")

    def test_an_old_shaped_row_defaults_to_zero_counts_and_per_response(self, db_conn) -> None:
        """The only producer of rows is the fixed ingest, so a caller that omits
        the new keys is still counting per response."""
        conn, project_id = db_conn
        store_telemetry(conn, {
            "project_id": project_id, "session_id": "s-old",
            "input_tokens": 1, "cache_creation_tokens": 0,
            "cache_read_tokens": 0, "output_tokens": 1,
        })
        row = conn.execute(
            "SELECT responses, memory_tool_responses, denied_responses, retried_denials, "
            "usage_basis FROM api_telemetry WHERE session_id = 's-old'"
        ).fetchone()
        assert tuple(row) == (0, 0, 0, 0, "per_response")

    def test_empty_table_reports_zero_round_trips(self, db_conn) -> None:
        conn, project_id = db_conn
        stats = get_cache_stats(conn, project_id)
        assert stats["legacy_sessions"] == 0
        assert stats["round_trips"] == {
            "responses": 0, "memory_tool_responses": 0,
            "denied_responses": 0, "retried_denials": 0,
        }
        assert stats["induced_share"] == 0.0

    def test_legacy_rows_are_excluded_and_reported(self, db_conn) -> None:
        """A row ingested before usage was counted per response is inflated 2-3x.
        It must not be silently averaged in with corrected rows."""
        conn, project_id = db_conn
        store_telemetry(conn, {
            "project_id": project_id, "session_id": "s-new",
            "input_tokens": 200, "cache_creation_tokens": 0,
            "cache_read_tokens": 800, "output_tokens": 10,
        })
        conn.execute(
            "INSERT INTO api_telemetry (project_id, session_id, input_tokens, "
            "cache_creation_tokens, cache_read_tokens, output_tokens, ingested_at, usage_basis) "
            "VALUES (?, ?, 600, 0, 90000, 30, 1, ?)",
            (project_id, "s-legacy", "per_line_legacy"),
        )
        conn.commit()
        stats = get_cache_stats(conn, project_id)
        assert stats["sessions_with_data"] == 1
        assert stats["total_cache_read_tokens"] == 800
        assert stats["legacy_sessions"] == 1

    def test_induced_share_spans_all_three_round_trip_classes(self, db_conn) -> None:
        conn, project_id = db_conn
        for sid, resp, mem, den, ret in (("a", 10, 2, 1, 1), ("b", 30, 3, 2, 1)):
            store_telemetry(conn, {
                "project_id": project_id, "session_id": sid,
                "input_tokens": 1, "cache_creation_tokens": 0, "cache_read_tokens": 0,
                "output_tokens": 1, "responses": resp, "memory_tool_responses": mem,
                "denied_responses": den, "retried_denials": ret,
            })
        stats = get_cache_stats(conn, project_id)
        assert stats["round_trips"] == {
            "responses": 40, "memory_tool_responses": 5,
            "denied_responses": 3, "retried_denials": 2,
        }
        assert abs(stats["induced_share"] - 0.25) < 1e-9
