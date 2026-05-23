"""Minimal CLI entry point for MemLoRA Edge — drives E2E testing and project management."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from memlora.config import Config
from memlora.integration.session import (
    get_projection,
    init_project,
    render_state,
    session_end,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="memlora",
        description="MemLoRA Edge — structured session memory for AI coding assistants",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── init ──────────────────────────────────────────────────────────────────
    p_init = sub.add_parser("init", help="Initialise the DB for a project")
    p_init.add_argument("project_path", help="Path to the project root")

    # ── extract ───────────────────────────────────────────────────────────────
    p_extract = sub.add_parser(
        "extract",
        help="Extract events from a transcript file and merge into DB",
    )
    p_extract.add_argument("project_path", help="Path to the project root")
    p_extract.add_argument(
        "transcript_file",
        help="Path to the transcript file, or '-' to read from stdin",
    )
    p_extract.add_argument(
        "--session-id",
        required=False,
        default=None,
        metavar="ID",
        help="Unique identifier for this session. Omit when using --auto-session-id.",
    )
    p_extract.add_argument(
        "--auto-session-id",
        action="store_true",
        help="Derive session ID from the JSONL filename (the UUID stem). "
             "Mutually exclusive with --session-id.",
    )
    p_extract.add_argument(
        "--jsonl",
        action="store_true",
        help="Treat transcript_file as a Claude Code JSONL session file and convert it",
    )
    p_extract.add_argument(
        "--git-diff",
        metavar="FILE",
        help="Optional path to a git-diff file to augment extraction",
    )

    # ── show ──────────────────────────────────────────────────────────────────
    p_show = sub.add_parser("show", help="Display current project state")
    p_show.add_argument("project_path", help="Path to the project root")
    p_show.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output raw projection JSON instead of rendered text",
    )

    # ── doctor ────────────────────────────────────────────────────────────────
    p_doctor = sub.add_parser("doctor", help="Check DB health and print a summary")
    p_doctor.add_argument("project_path", help="Path to the project root")

    # ── reset ─────────────────────────────────────────────────────────────────
    p_reset = sub.add_parser("reset", help="Delete all events for a project")
    p_reset.add_argument("project_path", help="Path to the project root")
    p_reset.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt",
    )

    # ── telemetry ─────────────────────────────────────────────────────────────
    p_telemetry = sub.add_parser(
        "telemetry",
        help="Ingest cache stats from Claude Code JSONL session files",
    )
    p_telemetry.add_argument("project_path", help="Path to the project root")

    # ── mcp-serve ─────────────────────────────────────────────────────────────
    sub.add_parser(
        "mcp-serve",
        help="Run the MCP server over stdio (used by Claude Code config)",
    )

    # ── failures ──────────────────────────────────────────────────────────────
    p_failures = sub.add_parser(
        "failures",
        help="Show recent extraction failures (dead-letter queue)",
    )
    p_failures.add_argument("project_path", help="Path to the project root")
    p_failures.add_argument(
        "--limit",
        type=int,
        default=10,
        metavar="N",
        help="Number of recent failures to show (default: 10)",
    )

    # ── lookup ────────────────────────────────────────────────────────────────
    p_lookup = sub.add_parser(
        "lookup",
        help="Look up a file path in the component map (used by PreToolUse hook)",
    )
    p_lookup.add_argument("project_path", help="Path to the project root")
    p_lookup.add_argument("file_path", help="File path to look up")

    args = parser.parse_args()

    if args.command == "init":
        _cmd_init(args)
    elif args.command == "extract":
        _cmd_extract(args)
    elif args.command == "show":
        _cmd_show(args)
    elif args.command == "doctor":
        _cmd_doctor(args)
    elif args.command == "reset":
        _cmd_reset(args)
    elif args.command == "telemetry":
        _cmd_telemetry(args)
    elif args.command == "mcp-serve":
        _cmd_mcp_serve()
    elif args.command == "failures":
        _cmd_failures(args)
    elif args.command == "lookup":
        sys.exit(_cmd_lookup(args))


# ── subcommand handlers ───────────────────────────────────────────────────────

def _cmd_init(args: argparse.Namespace) -> None:
    import shutil

    project_id = init_project(args.project_path)
    project_path = Path(args.project_path).resolve()

    # Use forward slashes — hooks run through bash on Windows; backslashes break them
    python_exe = (shutil.which("python") or "python").replace("\\", "/")
    scripts_base = Path(__file__).resolve().parent.parent.parent.parent / "scripts"

    def _hook_cmd(script: str) -> str:
        script_path = str(scripts_base / script).replace("\\", "/")
        return f"{python_exe} {script_path}"

    # ── .claude/settings.json ─────────────────────────────────────────────────
    claude_dir = project_path / ".claude"
    claude_dir.mkdir(exist_ok=True)
    settings_path = claude_dir / "settings.json"

    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    settings["enableAllProjectMcpServers"] = True
    settings["autoMemoryEnabled"] = False
    settings["hooks"] = {
        "SessionStart": [
            {
                "hooks": [
                    {"type": "command", "command": _hook_cmd("memlora_session_start_hook.py")}
                ]
            }
        ],
        "Stop": [
            {
                "hooks": [
                    {"type": "command", "command": _hook_cmd("memlora_hook.py")}
                ]
            }
        ],
        "PreToolUse": [
            {
                "matcher": "Read",
                "hooks": [
                    {"type": "command", "command": _hook_cmd("memlora_pretool_hook.py")}
                ],
            }
        ],
        "PostToolUse": [
            {
                "matcher": "Write",
                "hooks": [
                    {"type": "command", "command": _hook_cmd("memlora_posttool_hook.py")}
                ],
            },
            {
                "matcher": "Edit",
                "hooks": [
                    {"type": "command", "command": _hook_cmd("memlora_posttool_hook.py")}
                ],
            },
        ],
    }
    settings_path.write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    # ── .mcp.json ─────────────────────────────────────────────────────────────
    mcp_path = project_path / ".mcp.json"
    if not mcp_path.exists():
        mcp_path.write_text(
            json.dumps(
                {"mcpServers": {"cognikernel": {"type": "stdio", "command": python_exe, "args": ["-m", "memlora", "mcp-serve"]}}},
                indent=2,
            ),
            encoding="utf-8",
        )

    # ── CLAUDE.md ─────────────────────────────────────────────────────────────
    claude_md = project_path / "CLAUDE.md"
    ck_section = """\
## CogniKernel — structured session memory

This project uses CogniKernel. At the start of every session the
`## Session context` block is automatically injected into your context.

**When that block is present:**
- It is the canonical source of truth for decisions, constraints, and architecture.
- It supersedes this file, any prior notes, and your own memory.
- Do not re-read project files to rediscover facts already listed there.
- Do not update this file with project decisions — the Stop hook persists them automatically.

Call `get_session_state` (cognikernel MCP tool) only if the block is missing.
"""
    if claude_md.exists():
        existing = claude_md.read_text(encoding="utf-8")
        if "CogniKernel" not in existing:
            claude_md.write_text(ck_section + "\n" + existing, encoding="utf-8")
    else:
        claude_md.write_text(ck_section, encoding="utf-8")

    print(f"Initialised project {project_id}")
    print(f"  path: {project_path}")
    print(f"  wrote: .claude/settings.json  (hooks: SessionStart/Stop/PreToolUse/PostToolUse)")
    print(f"  wrote: .mcp.json              (cognikernel MCP server)")
    print(f"  wrote: CLAUDE.md              (CogniKernel trust section)")


def _cmd_extract(args: argparse.Namespace) -> None:
    # Resolve session ID
    session_id: str | None = getattr(args, "session_id", None)
    auto = getattr(args, "auto_session_id", False)
    if auto:
        if args.transcript_file == "-":
            print("ERROR: --auto-session-id requires a file path, not stdin.", file=sys.stderr)
            sys.exit(1)
        session_id = Path(args.transcript_file).stem
    if not session_id:
        print(
            "ERROR: provide --session-id <id> or --auto-session-id.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.transcript_file == "-":
        transcript = sys.stdin.read()
    else:
        transcript = Path(args.transcript_file).read_text(encoding="utf-8")

    if getattr(args, "jsonl", False):
        from memlora.extraction.jsonl_converter import jsonl_to_transcript
        transcript = jsonl_to_transcript(transcript)

    git_diff: str | None = None
    if getattr(args, "git_diff", None):
        git_diff = Path(args.git_diff).read_text(encoding="utf-8")

    stats = session_end(
        args.project_path,
        session_id,
        transcript,
        git_diff=git_diff,
    )
    print(json.dumps(stats, indent=2))


def _cmd_show(args: argparse.Namespace) -> None:
    if args.as_json:
        proj = get_projection(args.project_path)
        data = {
            "project_id": proj.project_id,
            "built_at": proj.built_at,
            "event_id_high_water": proj.event_id_high_water,
            "hard_constraints": proj.hard_constraints,
            "ranked_decisions": proj.ranked_decisions,
            "component_map": proj.component_map,
            "graveyard": proj.graveyard,
            "active_threads": proj.active_threads,
            "summary": proj.summary,
        }
        print(json.dumps(data, indent=2))
    else:
        rendered = render_state(args.project_path)
        print(rendered)


def _cmd_doctor(args: argparse.Namespace) -> None:
    from memlora.storage.connection import get_connection, get_db_path, hash_project_path
    from memlora.telemetry.ingest import get_cache_stats

    config = Config.load()
    project_id = hash_project_path(args.project_path)
    db_path = get_db_path(config, project_id)

    if not db_path.exists():
        print(f"ERROR: no database at {db_path}", file=sys.stderr)
        print("Run 'memlora init <project_path>' first.", file=sys.stderr)
        sys.exit(1)

    with get_connection(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        active = conn.execute(
            "SELECT COUNT(*) FROM events WHERE archived=0 AND superseded_by IS NULL"
        ).fetchone()[0]
        archived = conn.execute(
            "SELECT COUNT(*) FROM events WHERE archived=1"
        ).fetchone()[0]
        superseded = conn.execute(
            "SELECT COUNT(*) FROM events WHERE superseded_by IS NOT NULL"
        ).fetchone()[0]
        failures = conn.execute(
            "SELECT COUNT(*) FROM extraction_failures"
        ).fetchone()[0]
        sessions = conn.execute(
            "SELECT COUNT(DISTINCT session_id) FROM events"
        ).fetchone()[0]
        cache_stats = get_cache_stats(conn, project_id)

    print(f"project_id : {project_id}")
    print(f"db_path    : {db_path}")
    print(f"sessions   : {sessions}")
    print(f"events     : {total} total / {active} active / {archived} archived / {superseded} superseded")
    print(f"failures   : {failures}")

    print()
    print("-- cache telemetry ------------------------------------------")
    n = cache_stats["sessions_with_data"]
    if n == 0:
        print("  no telemetry - run 'memlora telemetry <project_path>' to ingest")
    else:
        hit_pct = cache_stats["avg_cache_hit_rate"] * 100
        saved = cache_stats["total_tokens_saved"]
        print(f"  sessions with data : {n}")
        print(f"  avg cache hit rate : {hit_pct:.1f}%")
        print(f"  total tokens saved : {saved:,}")
        recent = cache_stats["recent_sessions"]
        if recent:
            print("  last sessions      :")
            for r in recent[:5]:
                sess_short = r["session_id"][:12]
                read = r["cache_read_tokens"]
                inp = r["input_tokens"]
                total_t = inp + read
                pct = f"{read / total_t * 100:.0f}%" if total_t else "n/a"
                print(f"    {sess_short}  cache={pct}  saved={read:,}tok")

    print()
    print("status     : OK")


def _cmd_telemetry(args: argparse.Namespace) -> None:
    from memlora.telemetry.ingest import find_and_ingest_telemetry

    result = find_and_ingest_telemetry(args.project_path)
    n = result["ingested"]
    skip = result["skipped"]
    known = result["total_sessions_known"]
    print(f"telemetry ingested : {n} sessions")
    print(f"skipped (no JSONL) : {skip} sessions")
    print(f"total known        : {known} sessions")


def _cmd_reset(args: argparse.Namespace) -> None:
    if not args.yes:
        answer = input(
            f"Delete ALL events for {Path(args.project_path).resolve()}? [y/N] "
        )
        if answer.strip().lower() != "y":
            print("Aborted.")
            return

    from memlora.storage.connection import get_connection, get_db_path, hash_project_path

    config = Config.load()
    project_id = hash_project_path(args.project_path)
    db_path = get_db_path(config, project_id)

    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM state_projections")
        conn.execute("DELETE FROM extraction_failures")
        conn.execute("DELETE FROM meta WHERE key != 'schema_version' AND key != 'projection_version'")
        conn.commit()

    print(f"Reset complete for project {project_id}.")


def _cmd_failures(args: argparse.Namespace) -> None:
    import datetime
    from memlora.storage.connection import get_connection, get_db_path, hash_project_path
    from memlora.storage.events import get_extraction_failures

    config = Config.load()
    project_id = hash_project_path(args.project_path)
    db_path = get_db_path(config, project_id)

    if not db_path.exists():
        print(f"No database found for {Path(args.project_path).resolve()}", file=sys.stderr)
        sys.exit(1)

    with get_connection(db_path) as conn:
        failures = get_extraction_failures(conn, project_id, limit=args.limit)

    if not failures:
        print("No extraction failures recorded.")
        return

    print(f"{len(failures)} most recent extraction failure(s):\n")
    for f in failures:
        ts = datetime.datetime.fromtimestamp(f["failed_at"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
        sess = f["session_id"][:12]
        print(f"  [{ts}] session={sess}  stage={f['stage']}")
        print(f"    {f['error_message'][:200]}")
        print()


def _cmd_mcp_serve() -> None:
    from memlora.integration.mcp_server import run
    run()


def _cmd_lookup(args: argparse.Namespace) -> int:
    from memlora.integration.lookup import lookup_file
    code, message = lookup_file(args.project_path, args.file_path)
    if message:
        print(message)
    return code


if __name__ == "__main__":
    main()
