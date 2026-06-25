"""J5 prep — scan benchmark DBs for meta-narration leaks the current regex misses."""
import glob
import json
import re
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"src")
from memlora.extraction.pipeline import _MEMORY_META_RE

# Candidate shapes of memory-system self-reference leaking into descriptions.
CANDIDATES = re.compile(
    r"APPROACH_ABANDONED|CONSTRAINT_HARD|CONSTRAINT_SOFT|THREAD_OPEN|DECISION\b|"
    r"\bentry recording\b|\bsuperseded?\b|\bwill persist\b|\bextract(?:ion|s)? (?:is|will)\b|"
    r"\bcommitted to memory\b|\bhasn'?t been committed\b|\brecord(?:ing|ed)? (?:now|this decision)\b|"
    r"\bmemory (?:has|says|is clear|returned|surfaces)\b|\breplacement constraint\b",
    re.IGNORECASE,
)

for db_path in glob.glob(r"C:\Users\Admin\.memlora\projects\*.db"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, event_type, weight, json_extract(payload,'$.description') d "
            "FROM events WHERE archived=0 AND superseded_by IS NULL"
        ).fetchall()
    except sqlite3.OperationalError:
        continue
    hits = []
    for r in rows:
        d = r["d"] or ""
        if CANDIDATES.search(d) and not _MEMORY_META_RE.search(d):
            hits.append((r["id"], r["event_type"], r["weight"], d))
    if hits:
        print(f"\n=== {db_path.split(chr(92))[-1]} ({len(hits)} new-leak candidates) ===")
        for i, et, w, d in hits[:15]:
            print(f"  #{i} [{et}] w={w:.2f}  {d[:130]}")
    conn.close()
