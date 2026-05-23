import sqlite3, json
from pathlib import Path

db_path = r"C:\Users\Admin\.memlora\projects\3ffe6ec7f4f75c42.db"

if not Path(db_path).exists():
    print("DB DOES NOT EXIST")
else:
    conn = sqlite3.connect(db_path, timeout=5.0)
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    sessions = [r[0] for r in conn.execute("SELECT DISTINCT session_id FROM events").fetchall()]
    print(f"event count: {count}")
    print(f"sessions: {sessions}")
    print()
    rows = conn.execute(
        "SELECT event_type, session_id, payload FROM events ORDER BY id"
    ).fetchall()
    for event_type, session_id, payload_json in rows:
        payload = json.loads(payload_json)
        desc = payload.get("description", "")[:120].encode("ascii", "replace").decode()
        rationale = payload.get("rationale", "")[:80].encode("ascii", "replace").decode()
        print(f"[{event_type}] {desc}")
        if rationale:
            print(f"         why: {rationale}")
    conn.close()
