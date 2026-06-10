#!/usr/bin/env python3
"""Ingest-cost gate — measures extraction latency growth and write amplification.

Reports per-firing wall time vs transcript size for every Stop hook invocation
recorded in a project DB. Used as a gate: the O(n²) growth curve and 5× write
amplification are the Sprint-I baselines to beat. Every improvement must be
validated against this script (benchmark only goes up, cost only goes down).

Metrics:
  firing latency  — job created_at to last ack, per firing, grouped by session
  transcript size — original_size_bytes from raw_evidence (uncompressed)
  write amp       — sum(original_size_bytes) / approx on-disk transcript size
                    (true amp = total evidence bytes / unique transcript bytes)
  growth trend    — per-session linear fit: ms/KB slope (the O(n) coefficient)

Thresholds (--check mode, override with --max-latency / --max-amp):
  MAX_LATENCY_S = 90   any single firing >= 90s fails (timeout cliff risk at 120s kill)
  MAX_WRITE_AMP = 4.0  write amp = total_orig / unique_content. Pre-I2 baseline ~5x.
                       After I2 (delta extraction) target is ~1.0x. Gate at 4.0 catches
                       regressions back toward the old O(n^2) behavior.

Usage:
    python scripts/eval_ingest_cost.py --db ~/.memlora/projects/<id>.db
    python scripts/eval_ingest_cost.py --db <db> --check
    python scripts/eval_ingest_cost.py --db <db> --max-latency 30 --max-amp 1.5
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

DEFAULT_MAX_LATENCY_S = 90.0
DEFAULT_MAX_AMP       = 4.0


def _ack_time_col(conn: sqlite3.Connection) -> str:
    cols = [r[1] for r in conn.execute("pragma table_info(extraction_job_acks)").fetchall()]
    for c in ("acked_at", "completed_at", "created_at", "ts", "timestamp"):
        if c in cols:
            return c
    raise RuntimeError(f"Cannot find timestamp column in extraction_job_acks; cols={cols}")


def load_firings(db_path: Path) -> list[dict]:
    """Return one dict per extraction job with timing + size."""
    conn = sqlite3.connect(str(db_path)); conn.row_factory = sqlite3.Row
    ts_col = _ack_time_col(conn)
    rows = conn.execute(f"""
        SELECT j.id, j.session_id, j.created_at,
               MAX(a.{ts_col}) AS done_at,
               e.original_size_bytes AS orig_bytes,
               e.stored_size_bytes   AS stored_bytes
        FROM   extraction_jobs j
        JOIN   extraction_job_acks a ON a.job_id = j.id
        JOIN   raw_evidence         e ON e.id     = j.evidence_id
        GROUP  BY j.id
        ORDER  BY j.session_id, j.created_at
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def report(firings: list[dict], max_latency_s: float, max_amp: float, check: bool) -> bool:
    if not firings:
        print("No firings found — DB may be empty or schema mismatch."); return True

    all_ok  = True
    flagged = []

    # ── per-session breakdown ───────────────────────────────────────────────
    sessions: dict[str, list[dict]] = {}
    for f in firings:
        sessions.setdefault(f["session_id"], []).append(f)

    print(f"{'session':<14}  {'#fire':>5}  {'KB range':>16}  {'s range':>12}  {'ms/KB':>7}  status")
    print("-" * 70)

    total_orig = total_stored = 0
    for sid, flist in sessions.items():
        lats  = [(f["done_at"] - f["created_at"]) / 1000 for f in flist]
        sizes = [f["orig_bytes"] / 1e3 for f in flist]
        total_orig   += sum(f["orig_bytes"] for f in flist)
        total_stored += sum(f["stored_bytes"] for f in flist)

        # slope (ms/KB) via simple linear fit: rate of latency growth with size
        if len(flist) >= 2:
            n   = len(flist)
            sx  = sum(sizes); sy  = sum(lats)
            sxx = sum(x*x for x in sizes); sxy = sum(x*y for x,y in zip(sizes, lats))
            denom = n*sxx - sx*sx
            slope_ms_kb = (n*sxy - sx*sy) / denom * 1000 if denom else 0.0
        else:
            slope_ms_kb = 0.0

        over = [lat for lat in lats if lat >= max_latency_s]
        status = "OK"
        if over:
            status = f"SLOW({len(over)})"
            flagged.append((sid, max(over)))
            all_ok = False

        print(f"{sid[:12]:<14}  {len(flist):>5}  "
              f"{min(sizes):>6.0f}-{max(sizes):>6.0f}KB  "
              f"{min(lats):>5.1f}-{max(lats):>4.1f}s  "
              f"{slope_ms_kb:>6.1f}  {status}")

    # ── aggregate ───────────────────────────────────────────────────────────
    # Write amplification = total evidence bytes / unique content bytes.
    # Unique content ≈ the LARGEST (final) evidence per session — the last
    # firing contains all previous content, so unique_bytes = sum of per-session
    # maxima. This measures "how many times was the same content re-ingested."
    # After I2 (delta extraction) each session's unique bytes == total_orig, so
    # write_amp drops to ~1.0.
    unique_bytes = sum(max(f["orig_bytes"] for f in flist) for flist in sessions.values())
    write_amp = total_orig / max(unique_bytes, 1)

    all_lats = [(f["done_at"] - f["created_at"]) / 1000 for f in firings]

    print()
    print(f"Total firings:          {len(firings)}")
    print(f"Max single firing:      {max(all_lats):.1f}s  (threshold {max_latency_s}s)")
    print(f"Total evidence orig:    {total_orig/1e6:.1f} MB")
    print(f"Unique content (est.):  {unique_bytes/1e6:.1f} MB  (sum of per-session final sizes)")
    print(f"Total evidence stored:  {total_stored/1e6:.1f} MB (compressed)")
    print(f"Write amplification:    {write_amp:.1f}x  (threshold {max_amp}x)  [unique→1.0x after I2]")

    amp_ok = write_amp < max_amp
    if not amp_ok:
        all_ok = False
        print(f"  FAIL: write amp {write_amp:.1f}x exceeds {max_amp}x")

    if flagged:
        print(f"\nSLOW firings (>= {max_latency_s}s):")
        for sid, lat in flagged:
            print(f"  {sid[:12]}  {lat:.1f}s  <- approaching 120s kill timeout")

    verdict = "OK" if all_ok else "REGRESSION"
    print(f"\nINGEST COST: {verdict}")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="project .db path")
    ap.add_argument("--check", action="store_true", help="exit 1 on REGRESSION")
    ap.add_argument("--max-latency", type=float, default=DEFAULT_MAX_LATENCY_S,
                    dest="max_latency", metavar="S",
                    help=f"max acceptable single-firing latency in seconds (default {DEFAULT_MAX_LATENCY_S})")
    ap.add_argument("--max-amp", type=float, default=DEFAULT_MAX_AMP,
                    dest="max_amp", metavar="X",
                    help=f"max acceptable write amplification ratio (default {DEFAULT_MAX_AMP})")
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"DB not found: {db}", file=sys.stderr); return 1

    firings = load_firings(db)
    ok = report(firings, args.max_latency, args.max_amp, args.check)
    return 0 if (ok or not args.check) else 1


if __name__ == "__main__":
    sys.exit(main())
