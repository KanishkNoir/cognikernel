"""Start the real CogniKernel MCP server over stdio and require it to list its tools.

Used by the fresh-install CI job. An import check is not enough: the server has to
answer `initialize` and `tools/list` the way Claude Code talks to it. mcp 2.0
removed `mcp.server.fastmcp`, and because uv.lock pinned mcp 1.x every other job
stayed green while a fresh `pip install` got a server that exited at startup.

Run from an empty directory; exits non-zero with the server's stderr on failure.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from importlib.metadata import version

EXPECTED_TOOLS = {"recall", "find_related", "skeleton", "get_session_state"}
TIMEOUT_S = 120


def main() -> int:
    env = dict(os.environ, COGNIKERNEL_DIR=tempfile.mkdtemp(prefix="ck-mcp-check-"),
               COGNIKERNEL_DISABLE_AUTO_WARM="1")
    subprocess.run([sys.executable, "-m", "cognikernel", "init", os.getcwd()],
                   env=env, check=True, capture_output=True)

    proc = subprocess.Popen([sys.executable, "-m", "cognikernel", "mcp-serve"], env=env,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True)
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()

    def send(message: dict) -> None:
        proc.stdin.write(json.dumps(message) + "\n")
        proc.stdin.flush()

    def receive(request_id: int) -> dict | None:
        deadline = time.time() + TIMEOUT_S
        while time.time() < deadline:
            try:
                message = json.loads(lines.get(timeout=2))
            except queue.Empty:
                if proc.poll() is not None:
                    return None
                continue
            if message.get("id") == request_id:
                return message
        return None

    def fail(reason: str) -> int:
        proc.kill()
        stderr = proc.stderr.read()
        print(f"MCP server check FAILED (mcp {version('mcp')}): {reason}\n--- server stderr ---\n{stderr[-3000:]}")
        return 1

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
              "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                         "clientInfo": {"name": "ci", "version": "0"}}})
    except (BrokenPipeError, OSError):
        return fail("server exited before accepting initialize")
    if not (reply := receive(1)) or "result" not in reply:
        return fail("no initialize result")

    send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    reply = receive(2)
    tools = {tool["name"] for tool in ((reply or {}).get("result") or {}).get("tools", [])}
    if missing := EXPECTED_TOOLS - tools:
        return fail(f"tools/list is missing {sorted(missing)} (got {sorted(tools)})")

    proc.kill()
    print(f"MCP server OK with mcp {version('mcp')}: {sorted(tools)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
