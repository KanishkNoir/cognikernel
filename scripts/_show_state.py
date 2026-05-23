"""Show render_state() output for a project path, ASCII-safe."""
import sys
sys.path.insert(0, "src")
from memlora.integration.session import render_state

path = sys.argv[1]
block = render_state(path)
print(block.encode("ascii", errors="replace").decode("ascii"))
