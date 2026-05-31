"""Back-compat shim for the Stop hook (CK-6a).

Logic lives in memlora.integration.hooks.stop_main (session-end extraction). New
projects register `python -m memlora hook-stop`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

if __name__ == "__main__":
    try:
        from memlora.integration.hooks import stop_main
        stop_main()
    except Exception:
        pass  # never block session teardown
