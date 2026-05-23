"""Claude Code SessionStart hook — injects CogniKernel state at every session start.

Fires on: startup, resume, compact, clear.
All sources receive the injection block via additionalContext so Claude never
needs to call get_session_state manually. This makes injection reliable even
when Claude reads CLAUDE.md or project files before consulting the MCP tool.
"""
from __future__ import annotations

import json
import sys

_REINJECT_SOURCES = frozenset({"startup", "resume", "compact", "clear"})


def main() -> None:
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        return

    source = payload.get("source", "")
    if source not in _REINJECT_SOURCES:
        return

    cwd = payload.get("cwd", "")
    if not cwd:
        return

    try:
        from memlora.integration.session_start import handle_session_start
        context = handle_session_start(cwd)
    except Exception:
        return

    if not context:
        return

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # never block Claude
