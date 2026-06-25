"""Back-compat shim for the PostToolUse (Write/Edit) hook (CK-6a).

Logic lives in memlora.integration.hooks.posttool_main (symbol-graph update). New
projects register `python -m memlora hook-posttool`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    try:
        from memlora.integration.hooks import posttool_main
        posttool_main()
    except Exception:
        pass  # never block Claude
