"""J2/J3 verification — migrate the gamma COPY, rebuild projection, render block.

Reports: schema version, consolidation stats, lineage samples, block token count.
Uses a sandboxed MEMLORA_DIR so the real store is untouched.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

PID = "79e384f12a00c402"
SRC = Path(r".bench_dbs\gamma_cogni.db")

home = Path(tempfile.mkdtemp(prefix="memlora_j2_"))
(home / "projects").mkdir()
shutil.copy(SRC, home / "projects" / f"{PID}.db")
os.environ["MEMLORA_DIR"] = str(home)
os.environ["MEMLORA_DISABLE_AUTO_WARM"] = "1"

conn = sqlite3.connect(home / "projects" / f"{PID}.db")
conn.row_factory = sqlite3.Row

from memlora.storage.migrations import run_migrations

run_migrations(conn)
print("schema_version:", conn.execute(
    "SELECT value FROM meta WHERE key='schema_version'").fetchone()[0])

from memlora.storage.projections import rebuild_projection

proj = rebuild_projection(conn, PID)
n_hard = len(proj.hard_constraints)
n_dec = len(proj.ranked_decisions)
consolidated = [
    r for r in proj.hard_constraints + proj.ranked_decisions
    if r["payload"].get("provenance_count", 0) > 1
]
print(f"projection: hard={n_hard} decisions={n_dec} "
      f"golden-records={len(consolidated)}")
for r in consolidated:
    print(f"  #{r['id']} (×{r['payload']['provenance_count']}) "
          f"key={r.get('decision_key')!r}  {r['payload']['description'][:80]}")
    for li in r["payload"].get("lineage", []):
        print(f"      previously: {li['description'][:70]}")

keyed = conn.execute(
    "SELECT COUNT(*) FROM events WHERE decision_key IS NOT NULL AND decision_key != ''"
).fetchone()[0]
print(f"\nbackfilled keys (non-empty): {keyed}")

# Render the block through the real path.
from memlora.config import Config
from memlora.integration.session import render_state

config = Config.load()
block = render_state(r"C:\Users\Admin\OneDrive\Desktop\GAMMA_COGNI", config)
from memlora.injection.template import count_tokens_accurate

try:
    toks = count_tokens_accurate(block)
except Exception:
    toks = None
print(f"\nblock chars: {len(block)}  tokens: {toks}")
print("---- block head ----")
print(block[:2200])
