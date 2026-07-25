"""Phase A statement-quality audit — sweep stores, emit a blinded labeling pool.

Reads every CogniKernel project store read-only, buckets stored memory
statements by a crude surface heuristic, and writes a stratified, blinded sample
for human labeling plus a sidecar of the metadata the labeler must not see.

The heuristic is a PROXY used to stratify sampling. The reported defect rate
comes from human labels only; scripts/audit_report.py measures how badly the
heuristic misses.

Writes:
  research/statement_audit/pool_<stamp>.jsonl        (labeler opens this)
  research/statement_audit/pool_meta_<stamp>.jsonl   (blinded metadata, keyed by id)
  research/statement_audit/heuristic_<stamp>.json    (corpus-wide sweep summary)

Usage: uv run python scripts/audit_statement_quality.py [--n 60] [--stratum clean]
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("research/statement_audit")
STORE_DIR = Path.home() / ".cognikernel" / "projects"
TYPES = ("DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
         "APPROACH_ABANDONED_DO_NOT_RETRY")

DEFECT_CODES = (
    "DANGLING_REFERENCE", "NOT_A_STATEMENT", "WRONG_TYPE", "NOT_DURABLE",
    "META_TALK", "COMPOUND", "MISSING_SUBJECT", "OTHER",
)

# Surface heuristics only. WRONG_TYPE, COMPOUND, MISSING_SUBJECT and OTHER are
# deliberately absent — they are semantic and no regex detects them, which is
# precisely why human labeling is required.
_RX = (
    ("DANGLING_REFERENCE", re.compile(
        r"^(with (it|this|that)\b|it\s|this\s|that\s|these\s|those\s|they\s"
        r"|but\s|however\b|the (above|latter|former|old|new)\b|old\s|new\s)", re.I)),
    ("NOT_DURABLE", re.compile(
        r"^(now\s|let me\b|let's\b|i'll\b|i will\b|i need to\b|next[,\s]|then\s"
        r"|first[,\s]|going to\b|update\s|add\s|read\s|check\s)", re.I)),
    ("META_TALK", re.compile(
        r"\b(stop hook|session context|cognikernel|claude\.md|memory block"
        r"|the recall|injected block)\b", re.I)),
    ("NOT_A_STATEMENT", re.compile(
        r"(\|\s*\.?\s*$|^[-+|=\s]+$|^\W{0,3}$|\.\.\.$|[─-╿]|—\s*$)")),
)


def classify_heuristic(text: str) -> set[str]:
    """Surface-detectable defect codes for one statement. Empty = no flag."""
    hits = {code for code, rx in _RX if rx.search(text)}
    if len(text.split()) < 5:
        hits.add("NOT_A_STATEMENT")
    return hits


def iter_statements(store_dir: Path) -> Iterator[dict]:
    """Yield active typed statements from every store, read-only."""
    placeholders = ",".join("?" * len(TYPES))
    for db in sorted(glob.glob(str(store_dir / "*.db"))):
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        except sqlite3.Error:
            continue  # a store that won't open is skipped, not fatal
        try:
            rows = conn.execute(
                f"SELECT event_type, payload FROM events "
                f"WHERE event_type IN ({placeholders}) AND archived=0", TYPES
            ).fetchall()
        except sqlite3.Error:
            continue  # older schema without this table/column
        finally:
            conn.close()
        for event_type, payload in rows:
            try:
                text = (json.loads(payload).get("description") or "").strip()
            except (json.JSONDecodeError, AttributeError):
                continue
            if text:
                yield {"store": Path(db).stem, "event_type": event_type, "text": text}


def statement_id(store: str, text: str) -> str:
    """Stable id for one statement. Store-scoped so identical text in two
    projects stays two rows — cross-project repetition is itself a finding."""
    return hashlib.sha256(f"{store}\x00{text}".encode("utf-8")).hexdigest()[:12]


def stratified_sample(rows: list[dict], n: int, stratum: str, seed: int) -> list[dict]:
    """Deterministic sample balanced across event_type.

    stratum: "clean" (heuristic found nothing) | "flagged" | "all".
    Returns fewer than n when the pool is smaller. Round-robins across types so
    a rare type (APPROACH_ABANDONED) is not swamped by DECISION.
    """
    if stratum == "clean":
        pool = [r for r in rows if not classify_heuristic(r["text"])]
    elif stratum == "flagged":
        pool = [r for r in rows if classify_heuristic(r["text"])]
    else:
        pool = list(rows)

    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for r in pool:
        by_type[r["event_type"]].append(r)

    rng = random.Random(seed)
    for bucket in by_type.values():
        bucket.sort(key=lambda r: (r["store"], r["text"]))  # stable pre-shuffle order
        rng.shuffle(bucket)

    out: list[dict] = []
    types = sorted(by_type)
    while len(out) < n and any(by_type[t] for t in types):
        for t in types:
            if by_type[t] and len(out) < n:
                out.append(by_type[t].pop())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="statements to sample")
    ap.add_argument("--stratum", choices=("clean", "flagged", "all"), default="clean")
    ap.add_argument("--seed", type=int, default=1106)
    ap.add_argument("--store-dir", type=Path, default=STORE_DIR)
    args = ap.parse_args()

    rows = list(iter_statements(args.store_dir))
    if not rows:
        sys.exit(f"no statements found under {args.store_dir}")

    counts = collections.Counter()
    per_type = collections.defaultdict(collections.Counter)
    for r in rows:
        hits = classify_heuristic(r["text"])
        for h in hits:
            counts[h] += 1
            per_type[r["event_type"]][h] += 1
        counts["_any" if hits else "_none"] += 1
        per_type[r["event_type"]]["_any" if hits else "_none"] += 1

    sample = stratified_sample(rows, args.n, args.stratum, args.seed)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pool_path = OUT_DIR / f"pool_{stamp}.jsonl"
    meta_path = OUT_DIR / f"pool_meta_{stamp}.jsonl"
    with pool_path.open("w", encoding="utf-8") as pf, \
         meta_path.open("w", encoding="utf-8") as mf:
        for r in sample:
            sid = statement_id(r["store"], r["text"])
            # BLINDED: labeler sees only id, type, text. No store, no heuristic.
            pf.write(json.dumps({"id": sid, "event_type": r["event_type"],
                                 "text": r["text"], "labels": [], "notes": ""},
                                ensure_ascii=False) + "\n")
            mf.write(json.dumps({"id": sid, "store": r["store"],
                                 "heuristic": sorted(classify_heuristic(r["text"])),
                                 "stratum": args.stratum, "seed": args.seed},
                                ensure_ascii=False) + "\n")

    summary = {
        "stamp": stamp, "stores_scanned": len(set(r["store"] for r in rows)),
        "statements_total": len(rows), "sampled": len(sample),
        "stratum": args.stratum, "seed": args.seed,
        "heuristic_counts": dict(counts),
        "heuristic_by_type": {k: dict(v) for k, v in per_type.items()},
    }
    (OUT_DIR / f"heuristic_{stamp}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"statements: {len(rows)} across {summary['stores_scanned']} stores")
    print(f"heuristic flagged: {counts['_any']} "
          f"({100 * counts['_any'] / len(rows):.1f}%)")
    print(f"\nwrote {pool_path}  ({len(sample)} to label)")
    print(f"wrote {meta_path}  (do NOT open before labeling)")


if __name__ == "__main__":
    main()
