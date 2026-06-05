"""MCP server adapter for MemLoRA Edge.

Three tools (model-controlled pull):
  - get_session_state — return the injection block (fallback when SessionStart hook missing)
  - recall            — rank prior decisions relevant to a query (no file reads)
  - find_related      — decisions + code areas related to a topic/file

Seven resources (client-discoverable structured memory, CK-5):
  Static:
    cognikernel://projects                         → all known projects + resource URIs
  Template (substitute {project_id}):
    cognikernel://project/{project_id}/state       → full session-start block
    cognikernel://project/{project_id}/constraints → CONSTRAINT_HARD events
    cognikernel://project/{project_id}/decisions   → ranked DECISION events
    cognikernel://project/{project_id}/graveyard   → APPROACH_ABANDONED_DO_NOT_RETRY
    cognikernel://project/{project_id}/skeleton    → AST symbol graph
    cognikernel://project/{project_id}/threads     → THREAD_OPEN events

  project_id = SHA-256(resolved_path)[:16] — discoverable via cognikernel://projects.
  Resources work with ANY MCP-capable client (Cursor, Copilot, Codex, etc.) without
  Claude Code hooks — the extraction path keeps the DB updated; resources deliver it.

Start via: memlora mcp-serve
Configure in <project>/.mcp.json:
  {"mcpServers": {"cognikernel": {"command": "memlora", "args": ["mcp-serve"]}}}
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from memlora.integration.query import find_related_memory, recall_memory
from memlora.integration.resources import (
    list_projects,
    render_section,
)
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
        "For targeted queries, use the recall or find_related tools, or read a specific "
        "resource (e.g. cognikernel://project/{id}/constraints) for structured typed memory. "
        "If a decision or constraint seems missing from the block, call recall BEFORE "
        "re-reading files, Globbing, or asking the user to rediscover it — the memory "
        "likely has it. Use find_related before changing a subsystem to surface related "
        "decisions and import-graph-adjacent code (the skeleton is an AST symbol "
        "graph ranked by PageRank centrality). "
        "IMPORTANT: Do not write decisions, constraints, or architecture notes to CLAUDE.md "
        "or any other file. The Stop hook automatically extracts and persists all decisions."
    ),
)


# ── Tools ─────────────────────────────────────────────────────────────────────


@_mcp.tool(
    description=(
        "Return the full CogniKernel memory block for a project (constraints, decisions, "
        "skeleton, threads). Call with the absolute project root path only if the "
        "session-start block is missing from your context."
    )
)
def get_session_state(project_path: str) -> str:
    return render_state(project_path)


@_mcp.tool(
    description=(
        "Recall prior project decisions/constraints relevant to a question, ranked by "
        "relevance — WITHOUT reading files. Use when you need a past decision and it "
        "isn't already in the session context block."
    )
)
def recall(project_path: str, query: str, limit: int = 8) -> str:
    return recall_memory(project_path, query, limit)


@_mcp.tool(
    description=(
        "Find decisions and code areas related to a topic or file — semantic neighbours "
        "UNION import-graph-adjacent events. Use to scope impact before changing a module."
    )
)
def find_related(project_path: str, query: str, limit: int = 8) -> str:
    return find_related_memory(project_path, query, limit)


# ── Resources (CK-5) ──────────────────────────────────────────────────────────
# Static resource: discover all projects + their IDs and section URIs.

@_mcp.resource(
    "cognikernel://projects",
    name="cognikernel-projects",
    title="CogniKernel — all managed projects",
    description=(
        "JSON array of all CogniKernel-managed projects on this machine. "
        "Each entry includes the project_id needed to construct section resource URIs, "
        "the project path, and pre-built URIs for every section."
    ),
    mime_type="application/json",
)
def projects_resource() -> str:
    return list_projects()


# Template resources: one per section, keyed by project_id (hex, no path issues).

@_mcp.resource(
    "cognikernel://project/{project_id}/state",
    name="cognikernel-state",
    title="CogniKernel — full session state",
    description="Complete session-start memory block: constraints, decisions, skeleton, threads.",
)
def state_resource(project_id: str) -> str:
    return render_section(project_id, "state")


@_mcp.resource(
    "cognikernel://project/{project_id}/constraints",
    name="cognikernel-constraints",
    title="CogniKernel — hard constraints",
    description=(
        "CONSTRAINT_HARD events — decisions that must never be violated, ranked by weight. "
        "These are protected from decay and always present if established."
    ),
)
def constraints_resource(project_id: str) -> str:
    return render_section(project_id, "constraints")


@_mcp.resource(
    "cognikernel://project/{project_id}/decisions",
    name="cognikernel-decisions",
    title="CogniKernel — key decisions",
    description="DECISION events ranked by composite weight (recency × repetition × centrality).",
)
def decisions_resource(project_id: str) -> str:
    return render_section(project_id, "decisions")


@_mcp.resource(
    "cognikernel://project/{project_id}/graveyard",
    name="cognikernel-graveyard",
    title="CogniKernel — rejected approaches",
    description=(
        "APPROACH_ABANDONED_DO_NOT_RETRY events — explicitly rejected approaches "
        "that must not be re-suggested. Protected from decay."
    ),
)
def graveyard_resource(project_id: str) -> str:
    return render_section(project_id, "graveyard")


@_mcp.resource(
    "cognikernel://project/{project_id}/skeleton",
    name="cognikernel-skeleton",
    title="CogniKernel — codebase skeleton",
    description=(
        "AST-extracted symbol graph: classes, methods, imports per file. "
        "PageRank-ranked by architectural centrality."
    ),
)
def skeleton_resource(project_id: str) -> str:
    return render_section(project_id, "skeleton")


@_mcp.resource(
    "cognikernel://project/{project_id}/threads",
    name="cognikernel-threads",
    title="CogniKernel — open threads",
    description="THREAD_OPEN events — active work items and open questions.",
)
def threads_resource(project_id: str) -> str:
    return render_section(project_id, "threads")


def run() -> None:
    """Start the MCP server over stdio."""
    # Kick the embedding model load in the background as soon as the (long-lived)
    # server starts, so by the time a `recall`/`find_related` call arrives the model
    # is ready and the answer is semantic — without ever blocking the first call on
    # the cold-start download (which falls back to lexical until the load finishes).
    try:
        from memlora.embedding.model import warm
        warm()
    except Exception:
        pass
    _mcp.run(transport="stdio")
