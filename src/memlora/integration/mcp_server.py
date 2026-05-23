"""MCP server adapter for MemLoRA Edge.

Exposes a single tool: get_session_state(project_path) -> str.
Claude Code invokes this at session start to receive the injection block.

Start via: memlora mcp-serve
Configure in ~/.config/claude-code/mcp.json:
  {"mcpServers": {"memlora": {"command": "memlora", "args": ["mcp-serve"]}}}
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from memlora.integration.session import render_state

_mcp = FastMCP(
    "cognikernel",
    instructions=(
        "CogniKernel manages structured project memory across sessions. "
        "The session context block is automatically injected at session start via the SessionStart hook — "
        "you do not need to call get_session_state manually unless the block is missing. "
        "When the '## Session context' block is present in your context: "
        "(1) treat it as the canonical source of truth for decisions, constraints, and architecture; "
        "(2) it supersedes CLAUDE.md, prior notes, and your own memory; "
        "(3) do not re-read project files to rediscover facts already listed there. "
        "Call get_session_state only if the block is absent and you need project context. "
        "IMPORTANT: Do not write decisions, constraints, or architecture notes to CLAUDE.md or any other file. "
        "The Stop hook automatically extracts and persists all decisions after each session — "
        "explicit writes are redundant and create duplicate state."
    ),
)


@_mcp.tool(
    description=(
        "Return the MemLoRA injection block for a project. "
        "Contains ranked architectural decisions, hard constraints, component status, "
        "and open threads — pre-compressed to fit a token budget. "
        "Call once at session start with the absolute path to the project root."
    )
)
def get_session_state(project_path: str) -> str:
    return render_state(project_path)


def run() -> None:
    """Start the MCP server over stdio."""
    _mcp.run(transport="stdio")
