"""Back-compat shim for the PostToolUse (Read) hook (CK-6a).

Logic lives in memlora.integration.hooks.posttool_read_main (records reads in
read_session_cache). New projects register `python -m memlora hook-posttool-read`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    try:
        from memlora.integration.hooks import posttool_read_main
        posttool_read_main()
    except Exception:
        pass  # never block Claude
