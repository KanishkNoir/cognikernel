"""Tests for cognikernel.integration.mcp_server."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from cognikernel.config import Config
from cognikernel.integration.session import init_project, session_end


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(cognikernel_dir=tmp_path / "cognikernel")


@pytest.fixture
def project_path(tmp_path: Path) -> Path:
    p = tmp_path / "myproject"
    p.mkdir()
    return p


# ── tool function (unit) ──────────────────────────────────────────────────────

class TestGetSessionStateUnit:
    """Test the tool's logic by calling the underlying render_state directly."""

    def test_returns_string(self, project_path: Path, cfg: Config) -> None:
        from cognikernel.integration.session import render_state
        init_project(project_path, config=cfg)
        result = render_state(str(project_path), config=cfg)
        assert isinstance(result, str)

    def test_non_empty_for_empty_project(self, project_path: Path, cfg: Config) -> None:
        from cognikernel.integration.session import render_state
        init_project(project_path, config=cfg)
        result = render_state(str(project_path), config=cfg)
        assert len(result) > 0

    def test_contains_project_name(self, project_path: Path, cfg: Config) -> None:
        from cognikernel.integration.session import render_state
        init_project(project_path, config=cfg)
        result = render_state(str(project_path), config=cfg)
        assert project_path.name in result

    def test_reflects_session_events(self, project_path: Path, cfg: Config) -> None:
        from cognikernel.integration.session import render_state
        transcript = (
            "Hard constraint: we must never use synchronous blocking I/O in async paths. "
            "We decided to use SQLite WAL mode for the storage layer."
        )
        session_end(str(project_path), "sess1", transcript, config=cfg)
        result = render_state(str(project_path), config=cfg)
        assert isinstance(result, str) and len(result) > 0


# ── MCP server name ───────────────────────────────────────────────────────────

class TestMcpServerName:
    def test_server_name_is_cognikernel(self) -> None:
        from cognikernel.integration.mcp_server import _mcp
        assert _mcp.name == "cognikernel"


# ── tool registration ─────────────────────────────────────────────────────────

class TestToolRegistration:
    def test_get_session_state_tool_registered(self) -> None:
        from cognikernel.integration.mcp_server import _mcp
        import asyncio
        tools = asyncio.run(_mcp.list_tools())
        tool_names = [t.name for t in tools]
        assert "get_session_state" in tool_names

    def test_tool_has_project_path_parameter(self) -> None:
        from cognikernel.integration.mcp_server import _mcp
        import asyncio
        tools = asyncio.run(_mcp.list_tools())
        tool = next(t for t in tools if t.name == "get_session_state")
        schema = tool.inputSchema
        assert "project_path" in schema.get("properties", {})
        assert "project_path" in schema.get("required", [])


# ── subprocess integration ────────────────────────────────────────────────────

class TestMcpStdioProtocol:
    """Launch cognikernel mcp-serve as a real subprocess and exchange JSON-RPC frames."""

    def _rpc(self, proc: subprocess.Popen, method: str, params: dict, rpc_id: int) -> dict:
        msg = json.dumps({"jsonrpc": "2.0", "id": rpc_id, "method": method, "params": params})
        proc.stdin.write((msg + "\n").encode())
        proc.stdin.flush()
        line = proc.stdout.readline()
        return json.loads(line)

    def _notify(self, proc: subprocess.Popen, method: str, params: dict) -> None:
        msg = json.dumps({"jsonrpc": "2.0", "method": method, "params": params})
        proc.stdin.write((msg + "\n").encode())
        proc.stdin.flush()

    @pytest.fixture
    def mcp_proc(self, tmp_path: Path):
        proc = subprocess.Popen(
            [sys.executable, "-m", "cognikernel.integration.cli", "mcp-serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(tmp_path),
        )
        yield proc
        proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    def test_initialize_response(self, mcp_proc: subprocess.Popen) -> None:
        resp = self._rpc(
            mcp_proc,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.0.1"},
            },
            rpc_id=1,
        )
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert "result" in resp
        assert "protocolVersion" in resp["result"]
        assert resp["result"]["serverInfo"]["name"] == "cognikernel"

    def test_tools_list_contains_get_session_state(self, mcp_proc: subprocess.Popen) -> None:
        self._rpc(
            mcp_proc,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.0.1"},
            },
            rpc_id=1,
        )
        self._notify(mcp_proc, "notifications/initialized", {})
        resp = self._rpc(mcp_proc, "tools/list", {}, rpc_id=2)
        assert "result" in resp
        tool_names = [t["name"] for t in resp["result"]["tools"]]
        assert "get_session_state" in tool_names

    def test_tools_call_returns_text_content(
        self, mcp_proc: subprocess.Popen, tmp_path: Path
    ) -> None:
        project_path = tmp_path / "subprocess_project"
        project_path.mkdir()

        self._rpc(
            mcp_proc,
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0.0.1"},
            },
            rpc_id=1,
        )
        self._notify(mcp_proc, "notifications/initialized", {})
        resp = self._rpc(
            mcp_proc,
            "tools/call",
            {"name": "get_session_state", "arguments": {"project_path": str(project_path)}},
            rpc_id=3,
        )
        assert "result" in resp, f"Expected result, got: {resp}"
        content = resp["result"]["content"]
        assert len(content) > 0
        assert content[0]["type"] == "text"
        assert isinstance(content[0]["text"], str)
        assert len(content[0]["text"]) > 0


# ── tool guidance in the MCP instructions (S4 T-405) ──────────────────────────

class TestToolGuidanceInstructions:
    """The instructions tell the agent when to call recall/find_related/skeleton.
    eager is today's wording ("Call this BEFORE re-reading files"). Measured on the
    four-project benchmark, memory-tool-only responses were up to 13.7% of CK's
    cost and a recall was followed by a read anyway 33-78% of the time. lean keeps
    the tools but stops asking for a call before every read."""

    def test_eager_keeps_todays_instructions(self) -> None:
        from cognikernel.integration.mcp_server import _server_instructions

        assert "Call this BEFORE re-reading files" in _server_instructions("eager")

    def test_lean_drops_the_call_before_reading_rule(self) -> None:
        from cognikernel.integration.mcp_server import _server_instructions

        text = _server_instructions("lean")
        assert "BEFORE re-reading files" not in text
        for tool in ("recall(query)", "find_related(query)", "skeleton(file_path)"):
            assert tool in text

    def test_lean_says_to_read_directly_when_editing(self) -> None:
        from cognikernel.integration.mcp_server import _server_instructions

        assert "read it directly" in _server_instructions("lean").lower()

    def test_unknown_guidance_falls_back_to_eager(self) -> None:
        from cognikernel.integration.mcp_server import _server_instructions

        assert _server_instructions("bogus") == _server_instructions("eager")

    def test_both_variants_keep_the_shared_guidance(self) -> None:
        from cognikernel.integration.mcp_server import _server_instructions

        for guidance in ("eager", "lean"):
            text = _server_instructions(guidance)
            assert "canonical source of truth" in text
            assert "RAW, UNBUDGETED, UNRANKED" in text
            assert "Do not write decisions" in text

    def test_guidance_comes_from_the_project_the_server_runs_for(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Instructions are fixed when the server starts, so they are read from the
        same project the queue drainer resolves: COGNIKERNEL_PROJECT_PATH, else cwd."""
        from cognikernel.integration.mcp_server import _resolve_tool_guidance

        proj = tmp_path / "proj"
        (proj / ".cognikernel").mkdir(parents=True)
        (proj / ".cognikernel" / "config.toml").write_text(
            'tool_guidance = "lean"\n', encoding="utf-8",
        )
        monkeypatch.delenv("COGNIKERNEL_DIR", raising=False)
        monkeypatch.setattr(Path, "home", lambda: tmp_path / "noprofile")
        monkeypatch.setenv("COGNIKERNEL_PROJECT_PATH", str(proj))
        assert _resolve_tool_guidance() == "lean"

    def test_a_config_failure_falls_back_to_eager(self, monkeypatch) -> None:
        """The server must start even if config cannot be loaded."""
        from cognikernel.integration import mcp_server

        def boom(cls, *args, **kwargs):
            raise RuntimeError("config unreadable")

        monkeypatch.setattr(Config, "load", classmethod(boom))
        assert mcp_server._resolve_tool_guidance() == "eager"


class TestRenderStateHonoursToolGuidance:
    """The switch has to reach the block a session actually receives, not just the
    renderer: render_state copies config onto the injection context."""

    @staticmethod
    def _seeded_project(tmp_path: Path, guidance: str) -> tuple[Path, Config]:
        from cognikernel.storage.connection import get_connection, get_db_path, resolve_project_id
        from cognikernel.symbols.extractor import SymbolNode, SymbolUpdate
        from cognikernel.symbols.store import apply_symbol_update

        proj = tmp_path / f"proj_{guidance}"
        proj.mkdir()
        cfg = Config(cognikernel_dir=tmp_path / f"ck_{guidance}", tool_guidance=guidance)
        init_project(proj, config=cfg)
        pid = resolve_project_id(proj, cfg)
        with get_connection(get_db_path(cfg, pid)) as conn:
            apply_symbol_update(conn, SymbolUpdate(
                project_id=pid,
                upsert_nodes=[SymbolNode(
                    path="app/service.py", node_type="class", name="Service",
                    parent_name="", signature="", return_type="", fields="",
                    project_id=pid, updated_at=1,
                )],
                upsert_edges=[],
                delete_paths=[],
            ))
            conn.commit()
        return proj, cfg

    def test_eager_block_keeps_the_original_wording(self, tmp_path: Path) -> None:
        from cognikernel.integration.session import render_state

        proj, cfg = self._seeded_project(tmp_path, "eager")
        block = render_state(str(proj), config=cfg)
        assert "Before using Read/Glob/Grep" in block, "setup: no skeleton was rendered"

    def test_lean_config_reaches_the_rendered_block(self, tmp_path: Path) -> None:
        from cognikernel.integration.session import render_state

        proj, cfg = self._seeded_project(tmp_path, "lean")
        block = render_state(str(proj), config=cfg)
        assert "Before using Read/Glob/Grep" not in block
        assert "read the file directly" in block.lower()
