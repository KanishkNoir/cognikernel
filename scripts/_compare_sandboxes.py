import sys, sqlite3
sys.path.insert(0, "src")
from memlora.config import Config
from memlora.storage.connection import get_db_path, hash_project_path

config = Config.load()

sandboxes = [
    ("sandbox_cognikernel", r"C:\Users\Admin\OneDrive\Desktop\sandbox_cognikernel"),
    ("sandbox_baseline",    r"C:\Users\Admin\OneDrive\Desktop\sandbox_baseline"),
    ("CogniKernel (self)",  r"C:\Users\Admin\OneDrive\Desktop\CogniKernel"),
]

for label, path in sandboxes:
    pid = hash_project_path(path)
    db  = get_db_path(config, pid)
    try:
        conn = sqlite3.connect(str(db))
        conn.row_factory = sqlite3.Row
        events   = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM events").fetchone()[0]
        has_sym  = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='symbol_nodes'").fetchone()[0]
        if has_sym:
            nodes = conn.execute("SELECT COUNT(*) FROM symbol_nodes").fetchone()[0]
            edges = conn.execute("SELECT COUNT(*) FROM symbol_edges").fetchone()[0]
            ntypes = conn.execute(
                "SELECT node_type, COUNT(*) as c FROM symbol_nodes GROUP BY node_type"
            ).fetchall()
            type_str = ", ".join(f"{r['node_type']}={r['c']}" for r in ntypes)
        else:
            nodes = edges = 0
            type_str = "pre-migration schema"
        conn.close()
        print(f"\n[{label}]")
        print(f"  events={events}  sessions={sessions}")
        print(f"  symbol_nodes={nodes}  symbol_edges={edges}")
        if type_str:
            print(f"  breakdown: {type_str}")
    except Exception as e:
        print(f"\n[{label}]  ERROR: {e}")
