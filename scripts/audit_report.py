"""Phase A statement-quality audit — compute rates from completed human labels.

Reads a labeled pool (pool_<stamp>.jsonl with `labels` filled in) plus its
sidecar meta, and reports the human-labeled defect rate with Wilson confidence
intervals, per-code and per-event-type breakdowns, inter-labeler agreement when
a second labeler file is supplied, and the heuristic's measured miss rate.

The human number is the result. The heuristic number is reported only as a
proxy, with its miss rate attached.

Usage: uv run python scripts/audit_report.py research/statement_audit/pool_<stamp>.jsonl \
           [--labeler-b path.jsonl] [--gate-types DECISION,APPROACH_ABANDONED_DO_NOT_RETRY]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("research/statement_audit")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Correct at the boundaries, unlike normal approx."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohens_kappa(a: list[set[str]], b: list[set[str]],
                 codes: tuple[str, ...]) -> dict:
    """Per-code binary Cohen's kappa plus the mean over defined codes.

    Returns None for a code neither labeler ever applied (kappa undefined
    rather than zero — reporting 0.0 there would understate agreement).
    """
    out: dict = {}
    defined = []
    for code in codes:
        n = len(a)
        both = sum(1 for x, y in zip(a, b) if code in x and code in y)
        neither = sum(1 for x, y in zip(a, b) if code not in x and code not in y)
        pa = (both + neither) / n if n else 0.0
        pa_yes = sum(1 for x in a if code in x) / n if n else 0.0
        pb_yes = sum(1 for x in b if code in x) / n if n else 0.0
        pe = pa_yes * pb_yes + (1 - pa_yes) * (1 - pb_yes)
        if pe >= 1.0:
            out[code] = None
            continue
        out[code] = (pa - pe) / (1 - pe)
        defined.append(out[code])
    out["_mean"] = sum(defined) / len(defined) if defined else None
    return out
