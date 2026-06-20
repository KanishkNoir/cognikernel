"""SessionStart hook logic — injects CogniKernel state at every session start.

Called by scripts/memlora_session_start_hook.py for all SessionStart sources
(startup, resume, compact, clear). Returns the injection block so the hook
surfaces it via additionalContext before Claude's first response.
"""
from __future__ import annotations

# The full directive contract lives HERE (dynamic, always present when the hook
# runs, sitting right above the facts/skeleton it governs) — CLAUDE.md carries
# only a minimal fallback. Principle (research/claude_md_design.md): instruct
# TRUST IN THE OUTPUT, never explain the mechanism. The "captured silently —
# never ask to save / announce" line fixes the observed "lock this into
# CogniKernel?" narration.
_TRUST_HEADER = """\
> **CogniKernel — your project memory.** The session-context block below is the canonical source \
of truth for this project's decisions, constraints, rejected approaches, and architecture; it \
supersedes CLAUDE.md and your own recollection — resolve any conflict in its favor. Do not Read or \
Glob a file shown in the skeleton below unless you need a function body to edit it (strict mode may \
deny the Read; repeat it within 60s for the body). For anything not shown here, call `recall` / \
`find_related` BEFORE exploring files or assuming greenfield. Decisions are captured automatically \
and silently at session end — never ask to save, record, or "lock in" a decision, and never \
announce writing to memory.\
"""

# Shown when extraction jobs are still queued at session start (async ingest
# can lag the previous session's final turns). Without this, an agent reading
# a thin block concludes "greenfield" and re-derives decisions blind — the
# exact failure observed in the gamma-CK S2 run.
_PENDING_NOTICE = """\
> ⏳ **Memory ingestion in progress: {n} extraction job(s) queued.** Recent \
decisions may not be loaded into this block yet. They will land shortly — \
use the `recall` MCP tool for targeted queries, and do NOT assume an empty \
block means no prior decisions exist.\
"""


def _pending_jobs_count(db_path) -> int:
    """ALL non-terminal extraction jobs — best-effort, never raises.

    Includes claimed/running: a job stuck mid-flight (e.g. its hook-drain
    worker was killed) is still pending memory the block doesn't contain yet —
    counting only queued under-reported exactly when it mattered most.
    """
    try:
        from memlora.storage.connection import get_connection
        with get_connection(db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM extraction_jobs "
                "WHERE state IN ('queued','retryable_failure','claimed','running')"
            ).fetchone()[0]
    except Exception:
        return 0


def handle_session_start(cwd: str | None, config=None, session_id: str | None = None) -> str:
    """Return the rendered injection block for any SessionStart source.

    Returns "" if cwd is falsy or the project has not been initialised.
    The hook script passes this as additionalContext in the hook JSON response.

    Always carries the recall affordance (in the trust header) and an explicit
    pending-ingestion notice when the queue is non-empty — a thin block must
    never read as "this project has no memory".
    """
    if not cwd:
        return ""

    from memlora.config import Config
    from memlora.storage.connection import get_db_path, hash_project_path

    config = config or Config.load(project_path=cwd)
    project_id = hash_project_path(cwd)
    db_path = get_db_path(config, project_id)

    if not db_path.exists():
        return ""

    from memlora.integration.session import render_state
    block = render_state(cwd, config=config, session_id=session_id)
    if not block:
        return ""

    parts = [_TRUST_HEADER]
    pending = _pending_jobs_count(db_path)
    if pending > 0:
        parts.append(_PENDING_NOTICE.format(n=pending))
    parts.append(block)
    return "\n\n".join(parts)


# Backwards-compat alias used by older hook scripts
handle_compact_event = handle_session_start
