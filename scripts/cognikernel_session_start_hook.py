"""Back-compat shim for the SessionStart hook (CK-6a).

The logic now lives in cognikernel.integration.hooks.session_start_main. New projects
register `python -m cognikernel hook-session-start`; this shim keeps older
settings.json entries (absolute script path) working.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    try:
        from cognikernel.integration.hooks import session_start_main
        session_start_main()
    except Exception:
        pass  # never block Claude
