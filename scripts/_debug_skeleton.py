"""Debug: show all nodes in DB and skeleton at multiple budgets."""
import sys
sys.path.insert(0, "src")
from memlora.config import Config
from memlora.storage.connection import get_db_path, hash_project_path, get_connection
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

print(f"Total nodes: {len(nodes)}  edges: {len(edges)}")
print(f"Config: token_budget={config.token_budget}  skeleton_budget={config.skeleton_budget}")

for budget, label in [(200, "old default"), (400, "doubled"), (800, "NEW default")]:
    sk = compress_to_skeleton(nodes, edges, budget_tokens=budget)
    total = sum(e.token_estimate for e in sk)
    files_covered = len(sk)
    print(f"\n--- Skeleton at {budget} tokens ({label}) ---")
    for e in sk:
        cls_names = [c.name for c in e.classes]
        fn_names  = [f.name for f in e.functions]
        print(f"  {e.path:<35} ~{e.token_estimate}tok  cls={cls_names}  fns={fn_names}")
    print(f"  TOTAL: {total} tokens  ({files_covered} files covered)")

print()
print("--- Rendered output at new default (800 tok) ---")
sk800 = compress_to_skeleton(nodes, edges, budget_tokens=800)
rendered = render_skeleton_section(sk800)
print(rendered.encode("ascii", errors="replace").decode("ascii"))
