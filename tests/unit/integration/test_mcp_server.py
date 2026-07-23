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
