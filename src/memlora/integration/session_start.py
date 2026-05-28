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
files to rediscover decisions already listed here.\
"""


def handle_session_start(cwd: str | None, config=None) -> str:
    """Return the rendered injection block for any SessionStart source.

    Returns "" if cwd is falsy or the project has not been initialised.
    The hook script passes this as additionalContext in the hook JSON response.
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
    return f"{_TRUST_HEADER}\n\n{block}"


# Backwards-compat alias used by older hook scripts
handle_compact_event = handle_session_start
