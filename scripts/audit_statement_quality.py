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
