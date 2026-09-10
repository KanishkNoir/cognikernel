"""MCP server adapter for CogniKernel.

Three tools (model-controlled pull):
  - get_session_state — return the injection block (fallback when SessionStart hook missing)
  - recall            — rank prior decisions relevant to a query (no file reads)
  - find_related      — decisions + code areas related to a topic/file

Seven resources (client-discoverable structured memory, CK-5):
  Static:
    cognikernel://projects                         → the calling project (+ resource URIs);
                                                        falls back to every known project only
                                                        if the current one can't be resolved
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

Start via: cognikernel mcp-serve
Configure in <project>/.mcp.json:
  {"mcpServers": {"cognikernel": {"command": "cognikernel", "args": ["mcp-serve"]}}}
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from cognikernel.integration.query import find_related_memory, recall_memory
from cognikernel.integration.resources import (
    list_projects,
    render_section,
)
from cognikernel.integration.session import render_state

# ── server instructions ──────────────────────────────────────────────────────
# Split so the tool-guidance wording can vary (S4 T-405) while everything else
# stays byte-identical between the two variants.

_BLOCK_GUIDANCE = (
    "CogniKernel manages structured project memory across sessions. "
    "The session context block is automatically injected at session start via "
    "the SessionStart hook — you do not need to call get_session_state manually "
    "unless the block is missing. When the '## Session context' block is "
    "present in your context: (1) treat it as the canonical source of truth "
    "for decisions, constraints, and architecture; (2) it supersedes CLAUDE.md, "
    "prior notes, and your own memory; (3) do not re-read project files to "
    "rediscover facts already listed there. Call get_session_state only if "
    "the block is absent.\n\n"
)

# eager: the original wording — call a memory tool before re-reading or globbing.
_EAGER_TOOL_GUIDANCE = (
    "PREFER THE TOOLS below over the raw resources for anything the block "
    "doesn't already answer — they are query-scoped and respect the same "
    "ranking/budget discipline the block uses, so they stay cheap:\n"
    "  - recall(query) — a prior decision/constraint relevant to a question, "
    "ranked, WITHOUT reading files. Call this BEFORE re-reading files, "
    "Globbing, or asking the user to rediscover something — the memory "
    "likely already has it.\n"
    "  - find_related(query) — decisions plus import-graph-adjacent code for "
    "a topic or file. Call before changing a subsystem to scope impact.\n"
    "  - skeleton(file_path) — full, uncapped public signatures for ONE file "
    "WITHOUT reading it. This is the correct escape hatch when the block's "
    "skeleton section omitted or compressed a file you need — reach for this, "
    "not a raw resource, when the block feels incomplete for a specific file.\n\n"
)

# lean: keep the tools, stop asking for a call before every read. Measured on the
# four-project benchmark, a recall was followed by a read anyway 33-78% of the time
# and a skeleton call by a Read of the same file 41-67%; memory-tool-only responses
# were up to 13.7% of CogniKernel's cost.
_LEAN_TOOL_GUIDANCE = (
    "USE THE TOOLS below for what the block doesn't already answer — they are "
    "query-scoped and respect the same ranking/budget discipline the block uses:\n"
    "  - recall(query) — a prior decision/constraint relevant to a question, ranked. "
    "Use it when the answer depends on a past decision the block doesn't list; if "
    "you are about to read or edit the file anyway, read it directly instead.\n"
    "  - find_related(query) — decisions plus import-graph-adjacent code for a topic "
    "or file. Use it to scope impact before changing a subsystem.\n"
    "  - skeleton(file_path) — full public signatures for ONE file. Use it when you "
    "need signatures only; if you need the implementation or are about to edit the "
    "file, read it directly — calling skeleton first just adds a round trip.\n\n"
)

_RESOURCE_GUIDANCE = (
    "The cognikernel://project/{id}/... resources (constraints, decisions, "
    "threads, graveyard, skeleton, state) return RAW, UNBUDGETED, UNRANKED "
    "dumps of the whole store — up to 50 constraints or 20 decisions in one "
    "read, none of the drop-to-fit budgeting the block and tools apply. They "
    "exist for non-Claude-Code MCP clients that have no hook-injected block, "
    "not as a bigger version of recall/find_related. If a tool result feels "
    "incomplete, that's a signal to narrow the recall/find_related query or "
    "use skeleton for the specific file — not to read the raw resource "
    "instead.\n\n"
    "IMPORTANT: Do not write decisions, constraints, or architecture notes to "
    "CLAUDE.md or any other file. The Stop hook automatically extracts and "
    "persists all decisions."
)

_TOOL_GUIDANCE = {"eager": _EAGER_TOOL_GUIDANCE, "lean": _LEAN_TOOL_GUIDANCE}


def _server_instructions(guidance: str) -> str:
    """MCP server instructions for a tool-guidance setting; unknown values get eager."""
    return (
        _BLOCK_GUIDANCE
        + _TOOL_GUIDANCE.get(guidance, _EAGER_TOOL_GUIDANCE)
        + _RESOURCE_GUIDANCE
    )


def _resolve_tool_guidance() -> str:
    """tool_guidance for the project this server runs for; "eager" on any failure.

    FastMCP fixes its instructions at construction (the property has no setter), so
    they are chosen at import, from the same project the queue drainer resolves:
    COGNIKERNEL_PROJECT_PATH, else the working directory. A config problem must not
    stop the server starting, so every failure falls back to the original wording.
    An invalid value is already reported by `cognikernel doctor`'s config check.
    """
    import logging
    import os

    try:
        from cognikernel.config import Config

        project_path = os.environ.get("COGNIKERNEL_PROJECT_PATH") or os.getcwd()
        return Config.load(project_path=project_path).tool_guidance
    except Exception as exc:
        logging.getLogger("cognikernel.mcp").warning(
            "tool_guidance unavailable, using eager instructions: %s", exc
        )
        return "eager"


_mcp = FastMCP(
    "cognikernel",
    instructions=_server_instructions(_resolve_tool_guidance()),
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


@_mcp.tool(
    description=(
        "Full AST skeleton (public classes/functions/imports) for files matching "
        "file_path — WITHOUT reading the file. The session-context skeleton section is "
        "budget-capped and may omit or compress files; this serves the complete "
        "signatures for a specific file. Use when the Read gate denies a file or the "
        "block's skeleton lacks the detail you need. Empty file_path = whole skeleton "
        "(budget-capped)."
    )
)
def skeleton(project_path: str, file_path: str = "") -> str:
    from cognikernel.integration.resources import render_skeleton
    from cognikernel.config import Config
    from cognikernel.storage.connection import resolve_project_id
    config = Config.load(project_path=project_path)
    return render_skeleton(resolve_project_id(project_path, config), path_filter=file_path)


# ── Resources (CK-5) ──────────────────────────────────────────────────────────
# Static resource: discover all projects + their IDs and section URIs.

@_mcp.resource(
    "cognikernel://projects",
    name="cognikernel-projects",
    title="CogniKernel — current project",
    description=(
        "JSON array with the calling project's entry (falls back to every "
        "CogniKernel-managed project on this machine only if the current one "
        "can't be resolved). Each entry includes the project_id needed to "
        "construct section resource URIs, the project path, and pre-built "
        "URIs for every section."
    ),
    mime_type="application/json",
)
def projects_resource() -> str:
    import os
    current_path = os.environ.get("COGNIKERNEL_PROJECT_PATH") or os.getcwd()
    return list_projects(current_path=current_path)


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


def _start_queue_drainer() -> None:
    """Background thread: drain queued extraction jobs for this project (I4/I7).

    The MCP server is the only long-lived CogniKernel process in a Claude Code
    session, which makes it the one reliable host for queue processing: hook
    subprocesses are killed with their Job Object on hook exit (Windows), so a
    worker detached from a hook may die at birth. This thread polls the
    project's queue and runs process_jobs in-process; the worker single-flight
    lock makes it safe alongside any CLI/hook-spawned workers.
    """
    import os
    import threading
    import time as _time

    project_path = os.environ.get("COGNIKERNEL_PROJECT_PATH") or os.getcwd()

    def _loop() -> None:
        from cognikernel.config import Config
        from cognikernel.integration.codex_sync import sync_codex_rollouts
        from cognikernel.integration.session import _worker_log, process_jobs
        from cognikernel.storage.connection import get_connection, get_db_path, resolve_project_id
        config = Config.load(project_path=project_path)
        project_id = resolve_project_id(project_path, config)
        _worker_log(project_id, f"mcp-drainer started (cwd={project_path})")
        try:
            sync_codex_rollouts(project_path, config=config, spawn_worker=False)
        except Exception as exc:
            _worker_log(project_id, f"mcp-drainer codex-sync error: {exc}")
        _time.sleep(10.0)  # let session start settle before first drain
        while True:
            try:
                config = Config.load(project_path=project_path)
                project_id = resolve_project_id(project_path, config)
                sync_codex_rollouts(project_path, config=config, spawn_worker=False)
                db_path = get_db_path(config, project_id)
                if db_path.exists():
                    with get_connection(db_path) as conn:
                        # Requeue jobs whose claimant died (hook drains are
                        # killed at the hook ceiling); otherwise they'd be
                        # invisible to the queued-count below for the whole
                        # 10-min stuck-grace window (GAMMA_CK_TEST lag bug).
                        from cognikernel.integration.session import _pid_alive
                        from cognikernel.storage.jobs import recover_orphaned_jobs
                        recover_orphaned_jobs(conn, _pid_alive)
                        n = conn.execute(
                            "SELECT COUNT(*) FROM extraction_jobs "
                            "WHERE state IN ('queued','retryable_failure')"
                        ).fetchone()[0]
                    if n > 0:
                        _worker_log(project_id, f"mcp-drainer: {n} queued — draining")
                        process_jobs(project_path, config=config)
            except Exception as exc:
                _worker_log(project_id, f"mcp-drainer error: {exc}")
            _time.sleep(15.0)

    threading.Thread(target=_loop, daemon=True, name="cognikernel-queue-drainer").start()


def run() -> None:
    """Start the MCP server over stdio."""
    # Kick the embedding model load in the background as soon as the (long-lived)
    # server starts, so by the time a `recall`/`find_related` call arrives the model
    # is ready and the answer is semantic — without ever blocking the first call on
    # the cold-start download (which falls back to lexical until the load finishes).
    try:
        from cognikernel.embedding.model import warm
        warm()
    except Exception:
        pass
    try:
        _start_queue_drainer()
    except Exception:
        pass
    _mcp.run(transport="stdio")
