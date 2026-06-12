"""J2 rehearsal — migrate copies of ALL real project DBs to v16 + rebuild."""
import glob
import os
import shutil
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")
os.environ.setdefault("MEMLORA_DISABLE_AUTO_WARM", "1")

from memlora.storage.migrations import run_migrations
from memlora.storage.projections import rebuild_projection

os.makedirs(".bench_dbs/all", exist_ok=True)
ok = fail = 0
for src in glob.glob(r"C:\Users\Admin\.memlora\projects\*.db"):
    dst = os.path.join(".bench_dbs/all", os.path.basename(src))
    shutil.copy(src, dst)
    try:
        c = sqlite3.connect(dst)
        c.row_factory = sqlite3.Row
        run_migrations(c)
        v = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()[0]
        assert v == "16", f"schema_version={v}"
        pid = c.execute("SELECT DISTINCT project_id FROM events LIMIT 1").fetchone()
        if pid:
            p = rebuild_projection(c, pid[0])
            gr = sum(
                1 for r in p.hard_constraints + p.ranked_decisions
                if r["payload"].get("provenance_count", 0) > 1
            )
            print(f"{os.path.basename(src)}: v16 OK, golden-records={gr}")
        else:
            print(f"{os.path.basename(src)}: v16 OK (no events)")
        c.close()
        ok += 1
    except Exception as e:
        fail += 1
        print(f"{os.path.basename(src)}: FAIL — {e}")
print(f"\nmigrated: {ok} ok, {fail} failed")
