"""MCP server adapter for MemLoRA Edge.

Exposes three tools:
  - get_session_state — return the injection block (fallback entrypoint when the
                        SessionStart hook's auto-injection is missing)
  - recall            — PULL: rank prior decisions relevant to a query (no file reads)
  - find_related      — PULL: decisions + code areas related to a topic/file
                        (semantic ∪ import-graph), query-seeded

Start via: memlora mcp-serve
Configure in <project>/.mcp.json:
  {"mcpServers": {"cognikernel": {"command": "memlora", "args": ["mcp-serve"]}}}
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from memlora.integration.query import find_related_memory, recall_memory
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
        "Return the CogniKernel memory block for a project (ranked decisions, "
        "constraints, component status, open threads). Call with the absolute "
        "project root path only if the session-start block is missing."
    )
)
def get_session_state(project_path: str) -> str:
    return render_state(project_path)


@_mcp.tool(
    description=(
        "Recall prior project decisions/constraints relevant to a question, ranked "
        "by relevance — WITHOUT reading files. Use when you need a past decision "
        "(e.g. 'which auth scheme did we choose?') and it isn't already in the "
        "session context block. Returns a short ranked list, or a 'none found' note."
    )
)
def recall(project_path: str, query: str, limit: int = 8) -> str:
    return recall_memory(project_path, query, limit)


@_mcp.tool(
    description=(
        "Find decisions and code areas related to a topic or file — semantic "
        "neighbours UNION import-graph-adjacent events. Query-seeded. Use to scope "
        "'what else does changing X touch?' without grepping the codebase."
    )
)
def find_related(project_path: str, query: str, limit: int = 8) -> str:
    return find_related_memory(project_path, query, limit)


def run() -> None:
    """Start the MCP server over stdio."""
    _mcp.run(transport="stdio")
