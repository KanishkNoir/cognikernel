"""One-off: remove test-residue project DBs (pytest tmp paths) from ~/.memlora."""
import glob
import os
import sqlite3

removed = 0
for p in glob.glob(os.path.expanduser(r"~\.memlora\projects\*.db")):
    path = None
    try:
        c = sqlite3.connect(p)
        row = c.execute("SELECT value FROM meta WHERE key='project_path'").fetchone()
        c.close()
        path = row[0] if row else None
    except Exception:
        continue
    if path and ("pytest_tmp" in path or "pytest-of" in path or "\\.pytest_tmp" in path):
        print("removing test residue:", os.path.basename(p), "->", path)
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(p + suffix)
            except OSError:
                pass
        removed += 1
print("removed:", removed)
