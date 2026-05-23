"""Trigger symbol extraction on a real project (no transcript, filesystem walk)."""
import sys, time
sys.path.insert(0, "src")
from memlora.integration.session import session_end
from memlora.config import Config

path = sys.argv[1]
config = Config.load()
stats = session_end(
    project_path=path,
    session_id=f"symbol-backfill-{int(time.time())}",
    transcript="",
    config=config,
    git_diff=None,
)
print(f"events extracted: {stats['extracted']}, inserted: {stats['inserted']}")
