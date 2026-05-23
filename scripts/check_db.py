import sqlite3, sys

db_path = r"C:\Users\Admin\.memlora\projects\7a32983b975f9023.db"

try:
    conn = sqlite3.connect(db_path, timeout=3.0)
    print("journal_mode:", conn.execute("PRAGMA journal_mode").fetchone()[0])
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    print("tables:", tables)
    if "meta" in tables:
        print("meta rows:", conn.execute("SELECT key, value FROM meta").fetchall())
    if "events" in tables:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        print("event count:", count)
    conn.close()
    print("DB is accessible - no lock")
except Exception as e:
    print("ERROR:", e)
