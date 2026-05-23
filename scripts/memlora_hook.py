"""Claude Code Stop hook — auto-extracts the completed session into MemLoRA.

Claude Code calls this script (via settings.json hooks.Stop) after every session
ends. It reads the hook payload from stdin (JSON), locates the JSONL transcript,
and runs `memlora extract` with --auto-session-id --jsonl.

Exits 0 on success OR on any recoverable error so it never blocks session teardown.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> None:
    # Claude Code passes hook context as JSON on stdin.
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        payload = {}

    session_id: str = payload.get("session_id", "")
    project_dir: str = payload.get("cwd", payload.get("project_dir", ""))

    if not session_id:
        _warn("memlora_hook: no session_id in payload — skipping extraction")
        return
    if not project_dir:
        _warn("memlora_hook: no cwd/project_dir in payload — skipping extraction")
        return

    # Claude Code stores transcripts at ~/.claude/projects/<project_hash>/<session_id>.jsonl
    # The project hash is computed from the resolved project path, but Claude Code names
    # the projects directory after the path with path separators replaced by dashes.
    claude_projects = Path.home() / ".claude" / "projects"
    project_path = Path(project_dir).resolve()

    # Find matching jsonl: look in all project dirs under ~/.claude/projects/
    jsonl_path: Path | None = None
    for candidate_dir in claude_projects.iterdir():
        candidate = candidate_dir / f"{session_id}.jsonl"
        if candidate.exists():
            jsonl_path = candidate
            break

    if jsonl_path is None:
        _warn(f"memlora_hook: JSONL not found for session {session_id} — skipping")
        return

    # Capture git diff for COMPONENT_STATUS augmentation.
    git_diff_content = ""
    try:
        git_result = subprocess.run(
            ["git", "-C", project_dir, "diff", "HEAD~1..HEAD", "--stat", "-p"],
            capture_output=True, text=True, timeout=30,
        )
        if git_result.returncode == 0 and git_result.stdout.strip():
            git_diff_content = git_result.stdout
    except Exception:
        pass

    cmd = [
        sys.executable, "-m", "memlora",
        "extract",
        str(project_path),
        str(jsonl_path),
        "--auto-session-id",
        "--jsonl",
    ]

    git_diff_file: tempfile.NamedTemporaryFile | None = None
    if git_diff_content:
        try:
            git_diff_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".diff", delete=False, encoding="utf-8"
            )
            git_diff_file.write(git_diff_content)
            git_diff_file.close()
            cmd += ["--git-diff", git_diff_file.name]
        except Exception:
            git_diff_file = None

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            _warn(f"memlora_hook: extract failed (rc={result.returncode}): {result.stderr[:300]}")
        else:
            stats = result.stdout.strip()
            _warn(f"memlora_hook: extracted session {session_id} → {stats[:200]}")
    except subprocess.TimeoutExpired:
        _warn("memlora_hook: extract timed out after 120s")
    except Exception as exc:
        _warn(f"memlora_hook: unexpected error: {exc}")
    finally:
        if git_diff_file is not None:
            try:
                Path(git_diff_file.name).unlink(missing_ok=True)
            except Exception:
                pass


def _warn(msg: str) -> None:
    print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
