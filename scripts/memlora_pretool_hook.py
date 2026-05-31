"""Back-compat shim for the PreToolUse hook (CK-6a).

Logic lives in memlora.integration.hooks.pretool_main (Read gate + optional Grep
cache). New projects register `python -m memlora hook-pretool`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    try:
        from memlora.integration.hooks import pretool_main
        pretool_main()
    except Exception:
        # Fail open — never block a Read on a broken hook.
        import json
        print(json.dumps({
            "hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}
        }))
