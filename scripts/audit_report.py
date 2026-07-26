"""Phase A statement-quality audit — compute rates from completed human labels.

Reads a labeled pool (pool_<stamp>.jsonl with `labels` filled in) plus its
sidecar meta, and reports the human-labeled defect rate with Wilson confidence
intervals, per-code and per-event-type breakdowns, inter-labeler agreement when
a second labeler file is supplied, and the heuristic's measured miss rate.

The human number is the result. The heuristic number is reported only as a
proxy, with its miss rate attached.

Usage: uv run python scripts/audit_report.py research/statement_audit/pool_<stamp>.jsonl \
           [--meta path.jsonl] [--labeler-b path.jsonl] \
           [--gate-types DECISION,APPROACH_ABANDONED_DO_NOT_RETRY] \
           [--gate-threshold 0.20] [--gate {A,A0}]

--meta is only auto-derived when the pool filename starts with "pool_"; for
any other input filename it must be given explicitly, or the heuristic miss
rate is reported unavailable rather than guessed.
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


def resolve_meta_path(pool: Path, explicit: Path | None) -> Path | None:
    """The pool_meta_<stamp>.jsonl sidecar to read, or None if unavailable.

    An explicit --meta always wins. Otherwise the default is derived ONLY
    when the pool filename actually starts with "pool_" (i.e. is itself a
    pool_<stamp>.jsonl written by audit_statement_quality.py); for any other
    filename (a verify_*.jsonl, a copy, a renamed file, ...) guessing a
    sidecar path is unsafe — a naive `.replace("pool_", "pool_meta_")` on a
    name without that prefix is a no-op and silently resolves back to the
    input file itself, which then gets read as its own "meta", fabricating a
    100% heuristic-miss-rate. Never fall back to the input path."""
    if explicit is not None:
        return explicit
    if not pool.name.startswith("pool_"):
        return None
    candidate = pool.with_name(pool.name.replace("pool_", "pool_meta_", 1))
    assert candidate != pool  # guaranteed by the startswith check above
    return candidate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("--labeler-b", type=Path, default=None)
    ap.add_argument("--meta", type=Path, default=None,
                    help="path to the pool_meta_<stamp>.jsonl sidecar. Only "
                         "auto-derived when the pool filename starts with "
                         "'pool_'; otherwise it must be given explicitly or "
                         "the heuristic miss rate is reported unavailable.")
    ap.add_argument("--gate-types", default="DECISION,APPROACH_ABANDONED_DO_NOT_RETRY")
    ap.add_argument("--gate-threshold", type=float, default=0.20)
    ap.add_argument("--gate", choices=("A", "A0"), default="A",
                    help="A: pre-registered Phase A gate — binary PROCEED/STOP "
                         "at >= --gate-threshold on --gate-types. "
                         "A0: pre-registered Phase A0 pilot gate — three-way "
                         "verdict (>=25%% run full Phase A / <15%% abandon / "
                         "between extend to n=150) on the whole labeled pool.")
    args = ap.parse_args()

    rows = load_labeled(args.pool)
    if not rows:
        sys.exit(f"no labeled rows in {args.pool} — label the pool first")

    meta_path = resolve_meta_path(args.pool, args.meta)
    if meta_path is not None and not meta_path.exists():
        if args.meta is not None:
            # The user named this file explicitly — a missing file is their
            # mistake to see and fix, not something to silently work around.
            sys.exit(f"meta sidecar not found: {meta_path}")
        # Derived default (pool_ prefix) that just isn't on disk yet/anymore
        # — not an error, just "no sidecar available". Same as if no meta
        # were ever supplied: falls through to the "unavailable" reporting
        # path below, never fatal.
        meta_path = None
    meta = {}
    if meta_path is not None:
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

    stratum = None
    if meta:
        strata = {rec.get("stratum") for rec in meta.values()}
        stratum = strata.pop() if len(strata) == 1 else None

    circular = False
    miss = None
    coverage = None
    if meta_path is None:
        print("\nheuristic miss rate: unavailable (no meta sidecar given; "
              "pass --meta explicitly — it is never guessed from the input "
              "filename)")
    else:
        # An id absent from meta is NOT "the heuristic found nothing for it"
        # — it means this sidecar has nothing to say about that row at all
        # (e.g. --meta points at the wrong file, or a stale/partial sidecar).
        # heuristic_miss_rate's `not meta.get(id, {}).get("heuristic")` would
        # otherwise treat "absent" and "heuristic found nothing" identically,
        # which is the same fabrication as the CRITICAL bug wearing a
        # different hat — this time reachable via --meta pointing anywhere,
        # not just the auto-derived path. Restrict to rows meta actually
        # covers, and refuse outright if it covers none of them.
        covered_rows = [r for r in rows if r["id"] in meta]
        coverage = len(covered_rows)
        if coverage == 0:
            print(f"\nheuristic miss rate: unavailable — {meta_path} shares "
                  "no ids with this pool (wrong --meta file?)")
        else:
            miss = heuristic_miss_rate(covered_rows, meta)
            rate = miss["miss_rate"]
            coverage_note = ("" if coverage == n else
                             f"  [meta covers {coverage}/{n} pool rows; "
                             "rate computed over those only]")
            if stratum == "clean":
                circular = True
                shown = f"{100 * rate:.1f}%" if rate is not None else "n/a"
                print(f"\nheuristic miss rate: {miss['missed_by_heuristic']}/"
                      f"{miss['human_defective']} = {shown} BY CONSTRUCTION — "
                      "this pool was sampled from stratum=clean "
                      "(heuristic-negative rows only), so every "
                      "human-labeled defect is necessarily one the heuristic "
                      f"missed. Not an informative measurement of the "
                      f"heuristic's real miss rate.{coverage_note}")
                if rate is not None and rate != 1.0:
                    print(f"  WARNING: expected exactly 100% on a "
                          f"clean-stratum pool but got {shown} — some row's "
                          "meta.heuristic is non-empty despite "
                          "stratum=clean. That contradicts the pool's own "
                          "sampling and should be investigated before "
                          "trusting any number here.")
            else:
                print(f"\nheuristic miss rate: {miss['missed_by_heuristic']}/"
                      f"{miss['human_defective']}"
                      + (f" = {100 * rate:.1f}%" if rate is not None else " (n/a)")
                      + coverage_note)

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
    verdict = None
    gate_record: dict = {"gate": args.gate}

    if args.gate == "A" and stratum == "clean":
        print(f"\n-- pre-registered gate ({args.gate}) --")
        print("  REFUSED: this pool's meta sidecar reports stratum=clean "
              "(heuristic-negative rows only). The Phase A gate is defined "
              "over an unfiltered corpus sample; a clean-stratum pool cannot "
              "answer it. Use --gate A0 for the Phase A0 pilot rule, or run "
              "Phase A on a stratum=all pool.")
        gate_record.update({"types": gate_types, "threshold": args.gate_threshold,
                            "verdict": None, "refused": "stratum_mismatch"})
    elif args.gate == "A":
        gate_rows = [r for r in rows if r.get("event_type") in gate_types]
        if gate_rows:
            gb, gn = defect_rate(gate_rows)
            glo, ghi = wilson_ci(gb, gn)
            rate = gb / gn
            verdict = "PROCEED" if rate >= args.gate_threshold else "STOP"
            print(f"\n-- pre-registered gate A ({'+'.join(gate_types)}) --")
            print(f"  rate {100 * rate:.1f}% [{100 * glo:.1f}-{100 * ghi:.1f}%] "
                  f"vs threshold {100 * args.gate_threshold:.0f}%  ->  {verdict}")
            print("  NOTE: the point estimate decides the gate as pre-registered; "
                  "the CI is reported for honesty, not to move the line after the fact.")
        gate_record.update({"types": gate_types, "threshold": args.gate_threshold,
                            "verdict": verdict})
    else:  # A0
        rate0 = bad / n
        lo0, hi0 = wilson_ci(bad, n)
        if rate0 >= 0.25:
            verdict = "RUN_FULL_PHASE_A"
        elif rate0 < 0.15:
            verdict = "ABANDON"
        else:
            verdict = "EXTEND_TO_150"
        print("\n-- pre-registered gate A0 (pilot) --")
        print(f"  clean-bucket defect rate {100 * rate0:.1f}% "
              f"[{100 * lo0:.1f}-{100 * hi0:.1f}%]  ->  {verdict}")
        print("  rule: >=25% run full Phase A / <15% abandon / "
              "between extend the pilot to n=150")
        gate_record.update({"threshold_run_full": 0.25, "threshold_abandon": 0.15,
                            "verdict": verdict})

    stamp = time.strftime("%Y%m%d-%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"results_{stamp}.json").write_text(json.dumps({
        "stamp": stamp, "pool": str(args.pool),
        "meta": str(meta_path) if meta_path is not None else None,
        "stratum": stratum, "n": n, "defective": bad,
        "rate": bad / n, "ci95": [lo, hi], "per_code": dict(code_counts),
        "heuristic_miss": miss, "heuristic_miss_circular_by_construction": circular,
        "heuristic_miss_meta_coverage": coverage,
        "kappa": kappa, "gate": gate_record,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / f'results_{stamp}.json'}")


if __name__ == "__main__":
    main()
