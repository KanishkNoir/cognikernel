"""SessionStart hook logic — injects CogniKernel state at every session start.

Called by scripts/memlora_session_start_hook.py for all SessionStart sources
(startup, resume, compact, clear). Returns the injection block so the hook
surfaces it via additionalContext before Claude's first response.
"""
from __future__ import annotations

_TRUST_HEADER = """\
> **CogniKernel is active.** The session context block below is the canonical \
source of truth for this project's decisions, constraints, and architecture. \
It supersedes CLAUDE.md and any prior session notes. Do not re-read project \
files to rediscover decisions already listed here. For anything not in the \
block, query memory with the `recall` / `find_related` MCP tools BEFORE \
exploring files or assuming greenfield.\
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
    """Queued/retryable extraction jobs — best-effort, never raises."""
    try:
        from memlora.storage.connection import get_connection
        with get_connection(db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM extraction_jobs "
                "WHERE state IN ('queued','retryable_failure')"
            ).fetchone()[0]
    except Exception:
        return 0


def handle_session_start(cwd: str | None, config=None) -> str:
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
    block = render_state(cwd, config=config)
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
