"""
For each orientation Read in the real sessions, check whether the skeleton
now covers the information that read was fetching.
Prints a file-by-file coverage table.
"""
import sys, sqlite3
sys.path.insert(0, "src")
from memlora.config import Config
from memlora.storage.connection import get_db_path, hash_project_path
from memlora.storage.connection import get_connection
from memlora.storage.migrations import run_migrations
from memlora.symbols.store import load_symbol_nodes, load_symbol_edges
from memlora.symbols.projection import compress_to_skeleton
from memlora.symbols.render import render_skeleton_section

config = Config.load()
project_path = r"C:\Users\Admin\OneDrive\Desktop\sandbox_cognikernel"
pid = hash_project_path(project_path)
db  = get_db_path(config, pid)

with get_connection(db) as conn:
    run_migrations(conn)
    nodes = load_symbol_nodes(conn, pid)
    edges = load_symbol_edges(conn, pid)

skeleton = compress_to_skeleton(nodes, edges, budget_tokens=200)

# Build a set of skeleton-covered file basenames
covered_paths = {e.path for e in skeleton}

# The orientation reads from the real sessions (extracted from previous script)
orientation_reads = [
    # session-1
    ("src/models.py",              "Quote class fields, Author class, SQLAlchemy Base"),
    ("src/database.py",            "get_db(), init_db(), Base declaration"),
    ("src/app.py",                 "FastAPI app setup, router mounts"),
    ("requirements.txt",           "installed packages (not code)"),
    ("tests/test_quotes.py",       "test structure (not code)"),
    ("src/api/quotes.py",          "route handlers, CRUD logic"),
    ("src/api/authors.py",         "author routes"),
    # session-2 extras
    ("src/__init__.py",            "empty / package marker"),
    ("src/api/__init__.py",        "empty / package marker"),
]

# Build skeleton file index
skeleton_index = {e.path: e for e in skeleton}

print("ORIENTATION READ COVERAGE — what skeleton covers vs what still needs a Read")
print("=" * 78)
print(f"{'File read':<35}  {'Covered?':^8}  {'What skeleton provides'}")
print("-" * 78)

for rel_path, reason in orientation_reads:
    entry = skeleton_index.get(rel_path)
    if entry:
        classes = ", ".join(
            f"{c.name}({c.bases})" if c.bases else c.name
            for c in entry.classes
        )
        fns = ", ".join(f.name for f in entry.functions)
        detail = " | ".join(filter(None, [classes, fns]))
        covered = "YES"
    else:
        detail = "not in skeleton"
        covered = "no"
    print(f"  {rel_path:<33}  {covered:^8}  {detail}")

print()
print(f"Skeleton entries: {len(skeleton)}")
for e in skeleton:
    classes = [f"{c.name}" for c in e.classes]
    fns = [f.name for f in e.functions]
    token_est = e.token_estimate
    print(f"  {e.path:<35}  classes={classes}  fns={fns}  ~{token_est}tok")

total = sum(e.token_estimate for e in skeleton)
print(f"\nTotal skeleton tokens: {total}")
print(f"Orientation reads that skeleton eliminates: "
      f"{sum(1 for p,_ in orientation_reads if p in skeleton_index)}/{len(orientation_reads)}")
