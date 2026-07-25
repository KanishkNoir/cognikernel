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


CODES = ("DANGLING_REFERENCE", "NOT_A_STATEMENT", "WRONG_TYPE", "NOT_DURABLE",
         "META_TALK", "COMPOUND", "MISSING_SUBJECT", "OTHER")


def load_labeled(path: Path) -> list[dict]:
    """Load a labeled pool, rejecting incomplete or contradictory rows loudly.

    A silently-skipped unlabeled row would bias the rate toward whatever the
    labeler happened to finish first, so this raises instead.
    """
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unlabeled, mixed, unnoted, unknown = [], [], [], []
    for r in rows:
        labels = r.get("labels") or []
        if not labels:
            unlabeled.append(r["id"])
        if "CLEAN" in labels and len(labels) > 1:
            mixed.append(r["id"])
        if "OTHER" in labels and not (r.get("notes") or "").strip():
            unnoted.append(r["id"])
        bad = [c for c in labels if c != "CLEAN" and c not in CODES]
        if bad:
            unknown.append(f"{r['id']}:{','.join(bad)}")
    problems = []
    if unlabeled:
        problems.append(f"unlabeled: {', '.join(unlabeled)}")
    if mixed:
        problems.append(f"CLEAN mixed with other codes: {', '.join(mixed)}")
    if unnoted:
        problems.append(f"OTHER without a note: {', '.join(unnoted)}")
    if unknown:
        problems.append(f"unknown codes: {', '.join(unknown)}")
    if problems:
        raise ValueError("; ".join(problems))
    return rows


def defect_rate(rows: list[dict]) -> tuple[int, int]:
    """(defective, total). Defective = carries any code other than CLEAN."""
    bad = sum(1 for r in rows
              if any(c != "CLEAN" for c in (r.get("labels") or [])))
    return bad, len(rows)


def heuristic_miss_rate(rows: list[dict], meta: dict) -> dict:
    """How often the surface heuristic missed a human-labeled defect."""
    human_bad = [r for r in rows
                 if any(c != "CLEAN" for c in (r.get("labels") or []))]
    missed = [r for r in human_bad if not meta.get(r["id"], {}).get("heuristic")]
    return {
        "human_defective": len(human_bad),
        "missed_by_heuristic": len(missed),
        "miss_rate": len(missed) / len(human_bad) if human_bad else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("--labeler-b", type=Path, default=None)
    ap.add_argument("--gate-types", default="DECISION,APPROACH_ABANDONED_DO_NOT_RETRY")
    ap.add_argument("--gate-threshold", type=float, default=0.20)
    args = ap.parse_args()

    rows = load_labeled(args.pool)
    meta_path = args.pool.with_name(args.pool.name.replace("pool_", "pool_meta_"))
    meta = {}
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                meta[rec["id"]] = rec

    bad, n = defect_rate(rows)
    lo, hi = wilson_ci(bad, n)
    print(f"labeled            : {n}")
    print(f"defective (human)  : {bad}  = {100 * bad / n:.1f}%  "
          f"[95% CI {100 * lo:.1f}-{100 * hi:.1f}%]")

    print("\n-- per code --")
    code_counts = collections.Counter(
        c for r in rows for c in (r.get("labels") or []) if c != "CLEAN")
    for c, v in code_counts.most_common():
        print(f"  {c:20s} {v:4d}  {100 * v / n:5.1f}%")

    print("\n-- per event type --")
    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r.get("event_type", "?")].append(r)
    for t, trows in sorted(by_type.items()):
        tb, tn = defect_rate(trows)
        tlo, thi = wilson_ci(tb, tn)
        print(f"  {t:32s} n={tn:4d}  {100 * tb / tn:5.1f}%  "
              f"[{100 * tlo:.1f}-{100 * thi:.1f}%]")

    miss = heuristic_miss_rate(rows, meta) if meta else None
    if miss:
        rate = miss["miss_rate"]
        print(f"\nheuristic miss rate: {miss['missed_by_heuristic']}/"
              f"{miss['human_defective']}"
              + (f" = {100 * rate:.1f}%" if rate is not None else " (n/a)"))

    kappa = None
    if args.labeler_b:
        b_rows = load_labeled(args.labeler_b)
        b_by_id = {r["id"]: set(r.get("labels") or []) for r in b_rows}
        shared = [r for r in rows if r["id"] in b_by_id]
        if shared:
            kappa = cohens_kappa([set(r["labels"]) for r in shared],
                                 [b_by_id[r["id"]] for r in shared], CODES)
            mean = kappa["_mean"]
            shown = f"{mean:.2f}" if mean is not None else "undefined"
            print(f"\ninter-labeler kappa (n={len(shared)}): mean {shown}")

    gate_types = [t.strip() for t in args.gate_types.split(",") if t.strip()]
    gate_rows = [r for r in rows if r.get("event_type") in gate_types]
    verdict = None
    if gate_rows:
        gb, gn = defect_rate(gate_rows)
        glo, ghi = wilson_ci(gb, gn)
        rate = gb / gn
        verdict = "PROCEED" if rate >= args.gate_threshold else "STOP"
        print(f"\n-- pre-registered gate ({'+'.join(gate_types)}) --")
        print(f"  rate {100 * rate:.1f}% [{100 * glo:.1f}-{100 * ghi:.1f}%] "
              f"vs threshold {100 * args.gate_threshold:.0f}%  ->  {verdict}")
        print("  NOTE: the point estimate decides the gate as pre-registered; "
              "the CI is reported for honesty, not to move the line after the fact.")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"results_{stamp}.json").write_text(json.dumps({
        "stamp": stamp, "pool": str(args.pool), "n": n, "defective": bad,
        "rate": bad / n, "ci95": [lo, hi], "per_code": dict(code_counts),
        "heuristic_miss": miss, "kappa": kappa,
        "gate": {"types": gate_types, "threshold": args.gate_threshold,
                 "verdict": verdict},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / f'results_{stamp}.json'}")


if __name__ == "__main__":
    main()
