"""Back-compat shim for the PostToolUse:Grep hook (CK-3a / CK-6a).

Logic lives in cognikernel.integration.hooks.posttool_grep_main. New projects
register `python -m cognikernel hook-posttool-grep`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    try:
        from cognikernel.integration.hooks import posttool_grep_main
        posttool_grep_main()
    except Exception:
        pass  # never block Claude
