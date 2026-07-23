"""End-to-end smoke test for the three Tier-1 fixes.

Builds an isolated synthetic project, seeds events that reproduce all three
failure modes from the Arm-C-v2 analysis (Tool policy absent, T1 mis-ranked,
bare basenames in injection), then runs render_state and asserts on the
rendered text.

Not a pytest module — invoked directly to keep the temp dirs visible.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

# Make the source tree importable as `cognikernel.*`
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cognikernel.config import Config
from cognikernel.extraction.authority import (
    ASSISTANT_DECIDED,
    USER_STATED,
)
from cognikernel.extraction.hashing import compute_content_hash
from cognikernel.integration.session import init_project, render_state
from cognikernel.storage.connection import get_connection, get_db_path, hash_project_path
from cognikernel.storage.events import Event, insert_event


def _seed(conn: sqlite3.Connection, project_id: str, session_id: str) -> None:
    """Seed events that reproduce the three failure modes."""

    # Fix 2 reproducer: T1 user_stated low-weight vs assistant_decided high-weight.
    insert_event(conn, Event(
        project_id=project_id, session_id=session_id,
        event_type="THREAD_OPEN",
        payload={
            "description": "We need to implement JWT authentication end-to-end.",
            "rationale": "",
            "authority": USER_STATED,
            "subject": "JWT auth thread",
        },
        content_hash=compute_content_hash("THREAD_OPEN", "jwt-thread"),
        weight=0.89,
    ))
    insert_event(conn, Event(
        project_id=project_id, session_id=session_id,
        event_type="THREAD_OPEN",
        payload={
            "description": "Alternative: catch Exception, rollback, re-raise.",
            "rationale": "",
            "authority": ASSISTANT_DECIDED,
            "subject": "transaction management musing",
        },
        content_hash=compute_content_hash("THREAD_OPEN", "musing"),
        weight=2.05,
    ))

    # Fix 3 reproducer: bare-basename COMPONENT_STATUS rows from a legacy DB.
    for bare in ("alembic.ini", "env.py", "config.py", "components.json"):
        try:
            insert_event(conn, Event(
                project_id=project_id, session_id=session_id,
                event_type="COMPONENT_STATUS",
                payload={
                    "path": bare,
                    "intent": bare,
                    "description": f"{bare} modified (legacy bare basename)",
                    "rationale": "",
                    "authority": "inferred_from_code",
                    "provenance": "file_mention",
                },
                content_hash=compute_content_hash("COMPONENT_STATUS", f"bare-{bare}"),
                weight=0.6,
                mention_count=5,
            ))
        except Exception:
            pass  # dedup collisions ok

    # And a qualified path so the section isn't empty.
    insert_event(conn, Event(
        project_id=project_id, session_id=session_id,
        event_type="COMPONENT_STATUS",
        payload={
            "path": "backend/app/core/security.py",
            "intent": "auth module",
            "description": "backend/app/core/security.py modified",
            "rationale": "",
            "authority": "inferred_from_code",
            "provenance": "file_mention",
        },
        content_hash=compute_content_hash("COMPONENT_STATUS", "security-qualified"),
        weight=0.6,
        mention_count=3,
    ))
    conn.commit()


def main() -> int:
    tmp_root = Path(tempfile.mkdtemp(prefix="tier1_smoke_"))
    try:
        cognikernel_dir = tmp_root / "cognikernel_data"
        os.environ["COGNIKERNEL_DIR"] = str(cognikernel_dir)

        project_path = tmp_root / "synthetic_project"
        project_path.mkdir()
        # Fix 1 reproducer: per-project strict-mode overlay.
        (project_path / ".cognikernel").mkdir()
        (project_path / ".cognikernel" / "config.toml").write_text(
            'hook_policy = "strict"\n', encoding="utf-8"
        )

        init_project(str(project_path))

        cfg = Config.load(project_path=str(project_path))
        project_id = hash_project_path(str(project_path))
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            _seed(conn, project_id, "smoke-session-1")

        # The acid test: no config passed — render_state must pick up the
        # per-project hook_policy="strict" overlay via Config.load(project_path=...).
        block = render_state(str(project_path))

        print("=" * 70)
        print("RENDERED INJECTION (synthetic project)")
        print("=" * 70)
        print(block)
        print("=" * 70)

        failures: list[str] = []

        # Fix 1 — Tool policy section present
        if "### Tool policy" not in block:
            failures.append("Fix 1 FAILED: '### Tool policy' section missing")
        else:
            print("\n[OK] Fix 1: Tool policy section rendered")

        # Fix 2 — Active thread is the user-stated JWT thread
        thread_line = ""
        for line in block.splitlines():
            if line.startswith("Working on:"):
                thread_line = line
                break
        if "JWT" not in thread_line and "authentication" not in thread_line:
            failures.append(
                f"Fix 2 FAILED: Active thread is not the user-stated JWT thread.\n"
                f"  Got: {thread_line!r}"
            )
        elif "catch Exception" in thread_line:
            failures.append("Fix 2 FAILED: assistant musing still wins the Active thread slot")
        else:
            print(f"[OK] Fix 2: Active thread shows JWT directive: {thread_line[:80]}")

        # Fix 3 — no bare-basename entries
        bare_in_hot = []
        bare_in_components = []
        in_hot = False
        in_components = False
        for line in block.splitlines():
            if line.startswith("### Most active files"):
                in_hot, in_components = True, False
                continue
            if line.startswith("### Component state"):
                in_hot, in_components = False, True
                continue
            if line.startswith("### "):
                in_hot, in_components = False, False
                continue
            if not line.startswith("- "):
                continue
            body = line[2:]
            path_segment = body.split(" · ", 1)[0]
            if "/" not in path_segment and path_segment.endswith((
                ".py", ".ini", ".ts", ".tsx", ".json", ".yml", ".yaml", ".cfg", ".sql",
            )):
                if in_hot:
                    bare_in_hot.append(path_segment)
                if in_components:
                    bare_in_components.append(path_segment)

        if bare_in_hot:
            failures.append(f"Fix 3 FAILED: bare basenames in Most active files: {bare_in_hot}")
        else:
            print("[OK] Fix 3a: no bare basenames in Most active files")
        if bare_in_components:
            failures.append(f"Fix 3 FAILED: bare basenames in Component state: {bare_in_components}")
        else:
            print("[OK] Fix 3b: no bare basenames in Component state")

        if failures:
            print("\n" + "=" * 70)
            print("FAILURES")
            print("=" * 70)
            for f in failures:
                print(f"\n{f}")
            return 1

        print("\n" + "=" * 70)
        print("ALL THREE TIER-1 FIXES VERIFIED END-TO-END")
        print("=" * 70)
        return 0

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
