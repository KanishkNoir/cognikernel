"""Read-only defect sweep over every local CogniKernel store.

Research tooling — NOT part of the shipped package.

Produces the prevalence baseline that the paper's "before" column and the CI
regression gate both read. It runs the SAME detector functions the live
admission gate runs (cognikernel.quality.detectors), which is the point:
measurement and enforcement cannot drift apart.

Usage:
    python scripts/injection_defect_audit.py
    python scripts/injection_defect_audit.py --out docs/metrics/injection_defect_baseline.json

Never writes to a store: every connection is opened read-only.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from cognikernel.quality.detectors import (
    detect_boilerplate,
    detect_junk_constraint,
    detect_subject_less,
    normalized_key,
)

STATEMENT_TYPES = (
    "DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
    "APPROACH_ABANDONED", "APPROACH_ABANDONED_DO_NOT_RETRY",
)


def wilson_interval(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation because defect rates here are small
    and the normal interval misbehaves (and can go negative) near zero.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def sweep_store(db_path: Path) -> tuple[Counter, Counter]:
    """Return (hits_by_rule, denominators_by_rule) for one store. Read-only."""
    hits: Counter = Counter()
    denom: Counter = Counter()
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT event_type, payload FROM events WHERE archived = 0"
        ).fetchall()
    except Exception:
        return hits, denom

    keys_by_norm: dict[str, set[str]] = defaultdict(set)
    for event_type, raw in rows:
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        description = (payload.get("description") or "").strip()
        if event_type not in STATEMENT_TYPES or not description:
            continue

        denom["D7"] += 1
        if detect_subject_less(description):
            hits["D7"] += 1

        denom["D4"] += 1
        if detect_boilerplate(description):
            hits["D4"] += 1

        if event_type.startswith("CONSTRAINT"):
            denom["D2"] += 1
            if detect_junk_constraint(description, event_type):
                hits["D2"] += 1

        key = normalized_key(description)
        if key:
            keys_by_norm[key].add(event_type)

    for _key, types in keys_by_norm.items():
        denom["D5"] += 1
        if len(types) > 1:
            hits["D5"] += 1

    conn.close()
    return hits, denom


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projects-dir",
        default=str(Path.home() / ".cognikernel" / "projects"),
    )
    parser.add_argument("--out", default="docs/metrics/injection_defect_baseline.json")
    args = parser.parse_args()

    stores = sorted(Path(args.projects_dir).glob("*.db"))
    total_hits: Counter = Counter()
    total_denom: Counter = Counter()
    stores_affected: dict[str, set] = defaultdict(set)

    for db in stores:
        hits, denom = sweep_store(db)
        total_hits.update(hits)
        total_denom.update(denom)
        for rule in hits:
            stores_affected[rule].add(db.name)

    classes = {}
    for rule in sorted(set(total_denom) | set(total_hits)):
        n = total_denom.get(rule, 0)
        h = total_hits.get(rule, 0)
        lo, hi = wilson_interval(h, n)
        classes[rule] = {
            "hits": h,
            "denominator": n,
            "rate": (h / n) if n else 0.0,
            "wilson_95": [lo, hi],
            "stores_affected": len(stores_affected.get(rule, ())),
        }

    report = {
        "generated_at": int(time.time()),
        "stores_scanned": len(stores),
        "classes": classes,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"stores scanned: {len(stores)}")
    print(f"{'rule':6} {'hits':>7} {'denom':>8} {'rate':>8}  {'95% CI':>18} stores")
    print("-" * 62)
    for rule, c in classes.items():
        lo, hi = c["wilson_95"]
        ci = f"[{100*lo:.2f}%, {100*hi:.2f}%]"
        print(f"{rule:6} {c['hits']:>7} {c['denominator']:>8} "
              f"{100*c['rate']:>7.2f}%  {ci:>18} {c['stores_affected']:>6}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
