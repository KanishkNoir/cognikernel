"""Minimal CLI entry point for MemLoRA Edge — drives E2E testing and project management."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from memlora.config import Config

# `memlora.integration.session` (and its extraction / symbol / tree-sitter stack)
# is imported lazily inside the handlers that need it, NOT at module top — so the
# `python -m memlora hook-*` hot path (hook-pretool fires on every Read) never pays
# for the heavy stack it doesn't use. See main()'s hook fast-path dispatch (CK-6a).


def _ensure_utf8_output() -> None:
    """Make CLI/hook output encoding-safe on non-UTF-8 consoles (e.g. Windows
    cp1252). The rendered block uses non-ASCII (the skeleton's '→'), which raises
    UnicodeEncodeError when printed to a cp1252 stdout. Best-effort: reconfigure
    stdout/stderr to UTF-8 with replacement; leave streams that can't reconfigure.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


# Hook subcommand → integration.hooks entrypoint. Dispatched before argparse and
# before any heavy import so the per-Read hook stays light (CK-6a).
_HOOK_ENTRYPOINTS = {
    "hook-session-start": "session_start_main",
    "hook-stop": "stop_main",
    "hook-pretool": "pretool_main",
    "hook-posttool": "posttool_main",
    "hook-posttool-read": "posttool_read_main",
    # CK-1: UserPromptSubmit query-time injection (flag: query_time_injection)
    "hook-user-prompt": "user_prompt_submit_main",
    # CK-4: SubagentStop transcript extraction (flag: capture_subagents)
    "hook-subagent-stop": "subagent_stop_main",
    # CK-3a: PostToolUse:Grep cache storage (gate: grep_cache_enabled)
    "hook-posttool-grep": "posttool_grep_main",
}


def main() -> None:
    _ensure_utf8_output()
    argv = sys.argv[1:]
    if argv and argv[0] in _HOOK_ENTRYPOINTS:
        from memlora.integration import hooks
        getattr(hooks, _HOOK_ENTRYPOINTS[argv[0]])()
        return
    parser = argparse.ArgumentParser(
        prog="memlora",
        description="MemLoRA Edge — structured session memory for AI coding assistants",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── init ──────────────────────────────────────────────────────────────────
    p_init = sub.add_parser("init", help="Initialise the DB for a project")
    p_init.add_argument("project_path", help="Path to the project root")
    p_init.add_argument(
        "--no-warm",
        action="store_true",
        help="Skip the one-time embedding-model download (recall stays lexical until warmed)",
    )

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

    # ── capture ───────────────────────────────────────────────────────────────
    p_capture = sub.add_parser(
        "capture",
        help="Store evidence + enqueue job, then spawn background worker (I4 fast path)",
    )
    p_capture.add_argument("project_path", help="Path to the project root")
    p_capture.add_argument("transcript_file", help="Path to the JSONL session file")
    p_capture.add_argument("--auto-session-id", action="store_true",
                           help="Derive session ID from the JSONL filename stem")
    p_capture.add_argument("--session-id", default=None, metavar="ID")
    p_capture.add_argument("--jsonl", action="store_true",
                           help="Transcript is a Claude Code JSONL file (default: yes for capture)")
    p_capture.add_argument("--git-diff", metavar="FILE",
                           help="Optional path to a git-diff file")
    p_capture.add_argument("--no-spawn", action="store_true",
                           help="Store evidence + enqueue only; do not spawn the worker subprocess")

    # ── process-jobs ──────────────────────────────────────────────────────────
    p_pjobs = sub.add_parser(
        "process-jobs",
        help="Claim and process queued extraction jobs (background worker for I4)",
    )
    p_pjobs.add_argument("project_path", help="Path to the project root")
    p_pjobs.add_argument("--max-jobs", type=int, default=50, metavar="N",
                         help="Max jobs to process in one run (default 50)")
    p_pjobs.add_argument("--time-budget", type=float, default=None, metavar="S",
                         dest="time_budget",
                         help="Stop claiming new jobs after S seconds (hook-safe drains)")

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
    p_failures.add_argument(
        "--replay",
        type=int,
        metavar="JOB_ID",
        help="Re-run extraction for a dead-lettered job using its original raw evidence",
    )

    # ── rebuild ───────────────────────────────────────────────────────────────
    p_rebuild = sub.add_parser(
        "rebuild",
        help="Regenerate derived tables from raw_evidence into a sidecar DB",
    )
    p_rebuild.add_argument("project_path", help="Path to the project root")
    p_rebuild.add_argument(
        "--from-raw",
        action="store_true",
        required=True,
        help="Replay all raw_evidence to regenerate events and projections",
    )
    p_rebuild.add_argument(
        "--sidecar",
        action="store_true",
        required=True,
        help=(
            "Write output to <project>.db.rebuild (required in V1 — "
            "the source DB is never modified)"
        ),
    )
    p_rebuild.add_argument(
        "--since",
        type=int,
        default=0,
        metavar="EVIDENCE_ID",
        help="Only replay evidence rows with id > EVIDENCE_ID (default: 0 = all)",
    )
    p_rebuild.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be processed without writing the sidecar",
    )

    # ── lookup ────────────────────────────────────────────────────────────────
    p_lookup = sub.add_parser(
        "lookup",
        help="Look up a file path in the component map (used by PreToolUse hook)",
    )
    p_lookup.add_argument("project_path", help="Path to the project root")
    p_lookup.add_argument("file_path", help="File path to look up")

    # ── warm ──────────────────────────────────────────────────────────────────
    sub.add_parser(
        "warm",
        help=(
            "Download + load the embedding model once into the persistent cache. "
            "Run before benchmarking so no session pays the cold-start download."
        ),
    )

    # ── install-heads ───────────────────────────────────────────────────────────
    p_install = sub.add_parser(
        "install-heads",
        help=(
            "Install the v2 (SetFit) encoder ONNX body + tokenizer into the canonical "
            "~/.memlora/models/salience_v2/ so v2/v2-broad extraction works outside a "
            "repo checkout. Run once before benchmarking the encoder backend."
        ),
    )
    p_install.add_argument(
        "--source",
        help="Dir containing body.onnx + tokenizer.json (default: the repo export output "
             "at models/salience_setfit/onnx). Produce it with scripts/export_setfit_onnx.py.",
    )
    p_install.add_argument(
        "--force", action="store_true", help="Overwrite artifacts already installed",
    )

    args = parser.parse_args()

    if args.command == "init":
        _cmd_init(args)
    elif args.command == "extract":
        _cmd_extract(args)
    elif args.command == "capture":
        _cmd_capture(args)
    elif args.command == "process-jobs":
        _cmd_process_jobs(args)
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
    elif args.command == "rebuild":
        _cmd_rebuild(args)
    elif args.command == "lookup":
        sys.exit(_cmd_lookup(args))
    elif args.command == "warm":
        _cmd_warm()
    elif args.command == "install-heads":
        _cmd_install_heads(args)


# ── subcommand handlers ───────────────────────────────────────────────────────


def _warm_embedding_model() -> bool:
    """Best-effort: download + load the embedding model into the persistent cache.

    The first fastembed fetch is ~130MB and can take minutes; doing it once, in the
    foreground (init or `memlora warm`), means no hook or MCP `recall` ever pays
    that cost mid-session. Prints status; returns True if the model is ready.
    Never raises — recall degrades to deterministic lexical matching without it.
    """
    import importlib.util
    import time

    if importlib.util.find_spec("fastembed") is None:
        print("embedding extra not installed — recall/find_related will use lexical matching.")
        return False

    from memlora.embedding.model import EMBEDDING_MODEL_VERSION, ensure_ready

    print(f"warming embedding model ({EMBEDDING_MODEL_VERSION}); first run downloads ~130MB, please wait…")
    t0 = time.monotonic()
    try:
        ok = ensure_ready(timeout=None)  # block fully: the one place a long wait is OK
    except Exception:
        ok = False
    dt = time.monotonic() - t0
    if ok:
        print(f"embedding model ready in {dt:.1f}s — cached under MEMLORA_DIR/models (default ~/.memlora/models)")
    else:
        print("embedding model download failed — recall/find_related will use lexical matching.")
    return ok


def _cmd_warm() -> None:
    if not _warm_embedding_model():
        sys.exit(1)

# ── In-session slash commands / skills (so operators never drop out of the
# assistant to run the `memlora` CLI) ─────────────────────────────────────────
# Two client surfaces, written by `memlora init` into every project:
#   - Claude Code: `.claude/commands/<name>.md` — a slash command whose body
#     `!`-executes the CLI and asks the model to summarise the output.
#   - Codex:       `.agents/skills/<name>/SKILL.md` — a repo-level skill that
#     instructs Codex to run the CLI itself. (Codex only discovers project skills
#     from `.agents/skills`; `.codex/prompts` is user-home-only and deprecated.)
# These files are CK-managed and rewritten on every init so template fixes
# propagate. `recall` / `find_related` are MCP tools, so those wrappers steer the
# model to the tool rather than shelling out (there is no `memlora recall` CLI).
#
# Spec tuple: (name, kind, target, takes_arg, blurb)
#   kind="cli" → runs `memlora <target>`; kind="mcp" → steers to the <target> MCP tool.
_AGENT_COMMANDS: list[tuple[str, str, str, bool, str]] = [
    ("ck-doctor", "cli", "doctor", False,
     "CogniKernel DB health: schema/projection version, job counts, telemetry, dead-letters."),
    ("ck-show", "cli", "show", False,
     "Render the current CogniKernel session-context block on demand."),
    ("ck-failures", "cli", "failures", False,
     "List CogniKernel dead-lettered extraction jobs for triage."),
    ("ck-lookup", "cli", "lookup", True,
     "Debug CogniKernel's strict Read-gate decision for a file path."),
    ("ck-recall", "mcp", "recall", True,
     "Recall prior CogniKernel decisions/constraints relevant to a query."),
    ("ck-related", "mcp", "find_related", True,
     "Find decisions and code areas related to a query via semantics and the import graph."),
]


def _write_claude_command(
    commands_dir: Path, name: str, kind: str, target: str, takes_arg: bool, blurb: str
) -> None:
    """Write one `.claude/commands/<name>.md` slash command."""
    front = ["---", f"description: {blurb}"]
    if takes_arg:
        front.append(f"argument-hint: {'<file-path>' if target == 'lookup' else '<query>'}")
    if kind == "cli":
        # Prefix-scoped permission so the `!` execution doesn't prompt every time.
        front.append(f"allowed-tools: Bash(python -m memlora {target}:*)")
    front.append("---")

    # The `!`-executed command MUST NOT contain a shell expansion: Claude Code
    # static-checks it against `allowed-tools` before running and rejects any
    # `$VAR` / `$(...)` ("Contains simple_expansion") because the expansion can't
    # be verified — so the command silently fails to run. We therefore pass the
    # project root as a literal `.` (slash-command `!`-bash runs from the project
    # root), NOT `$CLAUDE_PROJECT_DIR`. `$ARGUMENTS` is safe — Claude Code
    # substitutes it into the command string *before* the permission check.
    if kind == "mcp":
        body = (
            f'Use the cognikernel `{target}` MCP tool to answer: "$ARGUMENTS". Pass '
            "project_path as the absolute path of this project's root directory. "
            "Summarise the returned items; do not re-read files for facts already covered.\n"
        )
    elif takes_arg:
        body = (
            f"Explain CogniKernel's `{target}` result for `$ARGUMENTS`, based on the "
            "output below. Do not modify any files.\n\n"
            f'!`python -m memlora {target} . "$ARGUMENTS"`\n'
        )
    else:
        body = (
            f"Run CogniKernel's `{target}` for this project and give a short, "
            "actionable summary of the output below. Do not modify any files.\n\n"
            f"!`python -m memlora {target} .`\n"
        )
    (commands_dir / f"{name}.md").write_text("\n".join(front) + "\n\n" + body, encoding="utf-8")


def _write_codex_skill(
    skills_dir: Path, name: str, kind: str, target: str, takes_arg: bool, blurb: str
) -> None:
    """Write one `.agents/skills/<name>/SKILL.md` repo-level Codex skill."""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    desc = f"{blurb} Use when the user asks for this CogniKernel/memlora operation."
    front = f"---\nname: {name}\ndescription: {desc}\n---\n\n"

    if kind == "mcp":
        body = (
            f"Use the cognikernel `{target}` MCP tool with the user's query and "
            "project_path set to the repository root, then summarise the returned "
            "decisions/constraints concisely.\n"
        )
    elif takes_arg:
        body = (
            f'From the repository root, run `python -m memlora {target} . "<PATH>"`, '
            "substituting the path the user supplied, then explain the result. Do not "
            "modify any files.\n"
        )
    else:
        body = (
            f"From the repository root, run `python -m memlora {target} .` and give a "
            "short, actionable summary of the output. Do not modify any files.\n"
        )
    (skill_dir / "SKILL.md").write_text(front + body, encoding="utf-8")


def _write_agent_commands(project_path: Path) -> None:
    """Scaffold the Claude Code slash commands and Codex skills for a project."""
    commands_dir = project_path / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)
    skills_dir = project_path / ".agents" / "skills"
    skills_dir.mkdir(parents=True, exist_ok=True)
    for spec in _AGENT_COMMANDS:
        _write_claude_command(commands_dir, *spec)
        _write_codex_skill(skills_dir, *spec)


def _cmd_install_heads(args: argparse.Namespace) -> None:
    """Copy the trained encoder ONNX bodies + tokenizers into the canonical install paths.

    The 133 MB body.onnx files are gitignored (regenerable), so they are not shipped. This
    places them at <MEMLORA_DIR>/models/{salience_v2, supersession_xenc}/ — the locations
    extraction.salience_v2 and delta.supersede_xenc resolve — so v2 extraction AND the
    cross-encoder supersession axis work outside a repo checkout (else they fail open to
    legacy / lexical). Each head is independent: a missing source is skipped, not fatal.
    """
    import os
    import shutil

    repo_models = Path(__file__).resolve().parents[3] / "models"
    models_root = Path(os.environ.get("MEMLORA_DIR") or (Path.home() / ".memlora")) / "models"
    heads = [
        ("v2 salience head", repo_models / "salience_setfit" / "onnx", "salience_v2"),
        ("supersession cross-encoder", repo_models / "supersession_xenc" / "onnx", "supersession_xenc"),
    ]
    if args.source:  # explicit override installs just the salience head from that dir
        heads = [("v2 salience head", Path(args.source).resolve(), "salience_v2")]

    needed = ["body.onnx", "tokenizer.json"]
    installed_any = False
    for label, src, dest_name in heads:
        if not all((src / f).exists() for f in needed):
            print(f"  skip [{label}]: source missing in {src} "
                  "(export via scripts/export_setfit_onnx.py / export_xenc_onnx.py)")
            continue
        dest = models_root / dest_name
        dest.mkdir(parents=True, exist_ok=True)
        for f in needed:
            target = dest / f
            if target.exists() and not args.force:
                print(f"  exists (skip): {target}  — use --force to overwrite")
                continue
            shutil.copy2(src / f, target)
            print(f"  installed [{label}]: {target}  ({(src / f).stat().st_size / 1e6:.1f} MB)")
        installed_any = True

    if not installed_any:
        print("no head artifacts found to install", file=sys.stderr)
        sys.exit(1)
    print(f"\nheads installed under {models_root}")
    print('Enable in .memlora/config.toml:  extractor = "v2-broad"  and '
          "(optional) cross_encoder_supersession = true.")


def _cmd_init(args: argparse.Namespace) -> None:
    import shutil

    from memlora.integration.session import init_project
    project_id = init_project(args.project_path)
    project_path = Path(args.project_path).resolve()

    # Use forward slashes — hooks run through bash on Windows; backslashes break them
    python_exe = (shutil.which("python") or "python").replace("\\", "/")

    def _hook_cmd(subcommand: str) -> str:
        # Path-portable: invoke the installed package (`python -m memlora <sub>`)
        # rather than an absolute script path, so moving the repo or reinstalling
        # never breaks the registered hooks (CK-6a). Requires `memlora` importable.
        return f"{python_exe} -m memlora {subcommand}"

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
                    {"type": "command", "command": _hook_cmd("hook-session-start")}
                ]
            }
        ],
        "Stop": [
            {
                "hooks": [
                    {"type": "command", "command": _hook_cmd("hook-stop")}
                ]
            }
        ],
        "PreToolUse": [
            {
                "matcher": "Read",
                "hooks": [
                    {"type": "command", "command": _hook_cmd("hook-pretool")}
                ],
            }
        ],
        "PostToolUse": [
            {
                "matcher": "Write",
                "hooks": [
                    {"type": "command", "command": _hook_cmd("hook-posttool")}
                ],
            },
            {
                "matcher": "Edit",
                "hooks": [
                    {"type": "command", "command": _hook_cmd("hook-posttool")}
                ],
            },
            {
                "matcher": "Read",
                "hooks": [
                    {"type": "command", "command": _hook_cmd("hook-posttool-read")}
                ],
            },
            {
                # CK-3a: cache grep results; gated by grep_cache_enabled in config.
                "matcher": "Grep",
                "hooks": [
                    {"type": "command", "command": _hook_cmd("hook-posttool-grep")}
                ],
            },
        ],
        "SubagentStop": [
            {
                # CK-4: extract decisions from subagent transcripts; gated by
                # capture_subagents in config (default True).
                "hooks": [
                    {"type": "command", "command": _hook_cmd("hook-subagent-stop")}
                ]
            }
        ],
        # UserPromptSubmit (CK-1) is intentionally NOT registered by default —
        # it fires on every prompt and ships behind the query_time_injection flag.
        # Users opt in by adding it to settings.json after measuring injection rate.
    }
    settings_path.write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    # ── .memlora/config.toml — per-project overrides ──────────────────────────
    # New projects ship with hook_policy='strict' so the C1 strict gate is active
    # immediately. Users can edit this file to fall back to advisory mode without
    # touching the global ~/.memlora/config.toml.
    memlora_dir = project_path / ".memlora"
    memlora_dir.mkdir(exist_ok=True)
    project_cfg_path = memlora_dir / "config.toml"
    if not project_cfg_path.exists():
        project_cfg_path.write_text(
            '# CogniKernel per-project config. Overrides ~/.memlora/config.toml.\n'
            'hook_policy = "strict"\n'
            '\n'
            '# Stage-2 extraction backend: legacy | v1 | v1-broad | v2 | v2-broad.\n'
            '#   legacy   = deterministic keyword/Aho-Corasick pipeline (default).\n'
            '#   v2-broad = SetFit fine-tuned encoder head (best quality). Requires the\n'
            '#              ONNX body installed once: `python -m memlora install-heads`.\n'
            '# The MEMLORA_EXTRACTOR env var overrides this. Heads fail open to legacy\n'
            '# when artifacts are absent, so a non-legacy value is always safe.\n'
            '# New projects default to the best mode (v2-broad). Install the ONNX body\n'
            '# once with `python -m memlora install-heads`, or set this to "legacy".\n'
            'extractor = "v2-broad"\n'
            '\n'
            '# Cross-encoder supersession axis (R5): catches a few paraphrased\n'
            '# corrections lexical misses, precision-safe (additive above the\n'
            '# temporal/authority/provenance gates; fail-open to lexical if the\n'
            '# cross-encoder body is not installed). Needs `install-heads` + warm\n'
            '# (embeddings) and adds some Stop-hook cost. Set false to use lexical only.\n'
            'cross_encoder_supersession = true\n',
            encoding="utf-8",
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

This project uses **CogniKernel** for cross-session memory. At the start of every
session a `## Session context` block is injected automatically — it is the
**canonical** source of truth for decisions, constraints, rejected approaches,
open work, and codebase structure, and it **supersedes this file and your own
recollection**. You never maintain memory by hand: the Stop hook extracts and
persists decisions for you.

**The injected block covers:**

1. **Decisions & constraints** — `### Hard constraints` and `### Key decisions`.
2. **Rejected approaches** — `### Do not retry`; never re-propose these.
3. **Codebase structure** — `### Codebase skeleton` is an AST-derived **symbol
   graph**: per-file classes / functions / methods and their import edges, ranked
   by architectural centrality (PageRank over the import graph) so the most
   connected files surface first. Treat it as a trustworthy map of the code.
   **Do not Read/Glob a file whose path appears in the skeleton unless you need the
   function body** (e.g. to replace an implementation). Under strict mode (the
   default), the PreToolUse hook denies such Reads; if you genuinely need the body,
   retry the same Read within 60 seconds and the retry is allowed.
4. **Open work** — `### Active thread` tracks the current focus.

**When something seems missing or you're unsure of a past decision, use
CogniKernel's tools BEFORE re-reading files, Globbing, or asking the user to
rediscover it:**

| MCP tool | Use it to |
| --- | --- |
| `recall(query)` | Retrieve prior decisions/constraints relevant to a question — your FIRST move when a fact isn't in the block. No file reads. |
| `find_related(query)` | Before changing a subsystem: surfaces related decisions AND code, fusing semantic similarity with the import graph (files that import / are imported by the target) — impact you'd otherwise miss. |
| `get_session_state()` | Re-fetch the full block if it is absent from context. |

Structured memory is also exposed as MCP **resources** (any client):
`cognikernel://projects` and
`cognikernel://project/{id}/{constraints,decisions,graveyard,skeleton,threads}`.

**Slash commands** available to the user (Claude Code; Codex exposes the same as
`$ck-*` skills): `/ck-recall <query>`, `/ck-related <query>`, `/ck-show`,
`/ck-doctor`, `/ck-failures`, `/ck-lookup <file>`.

**Do not write decisions, constraints, or architecture notes to this file** — the
Stop hook persists them via extraction. Hand-written notes here are fine only as
supplementary documentation the injection cannot replace.
"""
    if claude_md.exists():
        existing = claude_md.read_text(encoding="utf-8")
        if "CogniKernel" not in existing:
            claude_md.write_text(ck_section + "\n" + existing, encoding="utf-8")
    else:
        claude_md.write_text(ck_section, encoding="utf-8")

    # ── In-session slash commands (Claude Code) + skills (Codex) ──────────────
    _write_agent_commands(project_path)

    # ── Clean up any stale /memlora-extract slash command (enrichment removed) ──
    stale_slash = Path(project_path) / ".claude" / "commands" / "memlora-extract.md"
    if stale_slash.exists():
        stale_slash.unlink()

    n_cmds = len(_AGENT_COMMANDS)
    print(f"Initialised project {project_id}")
    print(f"  path: {project_path}")
    print(f"  wrote: .claude/settings.json  (hooks: SessionStart/Stop/PreTool/PostTool [Write/Edit/Read])")
    print(f"  wrote: .mcp.json              (cognikernel MCP server)")
    print(f"  wrote: CLAUDE.md              (CogniKernel trust section)")
    print(f"  wrote: .memlora/config.toml   (hook_policy=strict)")
    print(f"  wrote: .claude/commands/ck-*.md       ({n_cmds} Claude Code slash commands)")
    print(f"  wrote: .agents/skills/ck-*/SKILL.md   ({n_cmds} Codex skills)")

    # Bundle the one-time embedding-model download into init so the very first
    # session never pays the multi-minute cold start (the bug behind the hung
    # `recall`). Foreground + best-effort; only the first init per machine actually
    # downloads — later inits load from the persistent cache. Opt out with
    # `--no-warm`; MEMLORA_DISABLE_AUTO_WARM=1 disables it for tests/CI (set in
    # tests/conftest.py so the suite never downloads 130MB per test).
    if not getattr(args, "no_warm", False) and not os.environ.get("MEMLORA_DISABLE_AUTO_WARM"):
        _warm_embedding_model()


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
        raw_input = sys.stdin.read()
    else:
        raw_input = Path(args.transcript_file).read_text(encoding="utf-8")

    transcript = raw_input
    evidence_source_type = "transcript"
    if getattr(args, "jsonl", False):
        from memlora.extraction.jsonl_converter import jsonl_to_transcript
        transcript = jsonl_to_transcript(raw_input)
        evidence_source_type = "jsonl_transcript"

    git_diff: str | None = None
    if getattr(args, "git_diff", None):
        git_diff = Path(args.git_diff).read_text(encoding="utf-8")

    from memlora.integration.session import session_end
    stats = session_end(
        args.project_path,
        session_id,
        transcript,
        git_diff=git_diff,
        evidence_content=raw_input,
        evidence_source_type=evidence_source_type,
        evidence_source_path="" if args.transcript_file == "-" else str(Path(args.transcript_file).resolve()),
    )
    print(json.dumps(stats, indent=2))


def _cmd_capture(args: argparse.Namespace) -> None:
    """Store evidence + enqueue, then spawn detached worker (I4 fast path)."""
    session_id: str | None = getattr(args, "session_id", None)
    auto = getattr(args, "auto_session_id", False)
    if auto:
        session_id = Path(args.transcript_file).stem
    if not session_id:
        print("ERROR: provide --session-id or --auto-session-id.", file=sys.stderr)
        sys.exit(1)

    raw_jsonl = Path(args.transcript_file).read_text(encoding="utf-8")

    git_diff: str | None = None
    if getattr(args, "git_diff", None):
        git_diff = Path(args.git_diff).read_text(encoding="utf-8")

    from memlora.integration.session import session_capture
    result = session_capture(
        args.project_path,
        session_id,
        raw_jsonl,
        git_diff=git_diff,
        evidence_source_type="jsonl_transcript",
        evidence_source_path=str(Path(args.transcript_file).resolve()),
    )
    print(json.dumps(result))

    # I7c: no spawn — a worker detached from a Claude Code hook is killed with
    # the hook's Job Object at hook exit (and a doomed worker can orphan the
    # single-flight lock). The Stop hook runs a sync time-budgeted drain after
    # capture instead; --no-spawn is kept for CLI back-compat and is a no-op.


def _cmd_process_jobs(args: argparse.Namespace) -> None:
    """Claim and process queued extraction jobs (worker / hook-drain entry point)."""
    from memlora.integration.session import process_jobs
    summary = process_jobs(
        args.project_path,
        max_jobs=args.max_jobs,
        time_budget_s=getattr(args, "time_budget", None),
    )
    print(json.dumps(summary))


def _cmd_show(args: argparse.Namespace) -> None:
    from memlora.integration.session import get_projection, render_state
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
    from memlora.storage.migrations import run_migrations
    from memlora.storage.evidence import get_evidence_summary
    from memlora.storage.jobs import list_jobs
    from memlora.telemetry.ingest import get_cache_stats

    config = Config.load()
    project_id = hash_project_path(args.project_path)
    db_path = get_db_path(config, project_id)

    if not db_path.exists():
        print(f"ERROR: no database at {db_path}", file=sys.stderr)
        print("Run 'memlora init <project_path>' first.", file=sys.stderr)
        sys.exit(1)

    with get_connection(db_path) as conn:
        run_migrations(conn)
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
            """
            SELECT COUNT(DISTINCT session_id)
            FROM (
                SELECT session_id FROM extraction_jobs WHERE project_id = ?
                UNION
                SELECT session_id FROM events WHERE project_id = ?
            )
            """,
            (project_id, project_id),
        ).fetchone()[0]
        cache_stats = get_cache_stats(conn, project_id)
        evidence_summary = get_evidence_summary(conn, project_id)
        import time as _time
        dead_jobs = list_jobs(conn, project_id, state="dead_lettered", limit=1000)
        queued_jobs = list_jobs(conn, project_id, state="queued", limit=1000)
        claimed_jobs = list_jobs(conn, project_id, state="claimed", limit=1000)
        retryable_jobs = list_jobs(conn, project_id, state="retryable_failure", limit=1000)
        now_ms = int(_time.time() * 1000)

    print(f"project_id : {project_id}")
    print(f"db_path    : {db_path}")
    print(f"sessions   : {sessions}")
    print(f"events     : {total} total / {active} active / {archived} archived / {superseded} superseded")
    print(f"failures   : {failures}")
    print(
        "evidence   : "
        f"{evidence_summary['count']} rows / "
        f"{evidence_summary['average_compression_ratio']:.2f}x avg compression"
    )
    print(
        f"jobs       : {len(queued_jobs)} queued / "
        f"{len(claimed_jobs)} claimed / "
        f"{len(retryable_jobs)} retryable / "
        f"{len(dead_jobs)} dead-lettered"
    )
    stale_claimed = [
        j for j in claimed_jobs
        if j.claimed_at is not None and now_ms - j.claimed_at > j.hard_timeout_ms
    ]
    if stale_claimed:
        print(
            f"  WARNING: {len(stale_claimed)} claimed job(s) exceeded hard_timeout — "
            "likely orphaned by a crashed process; run 'memlora doctor' again after "
            "the next session to confirm they clear."
        )

    print()
    print("-- cache telemetry ------------------------------------------")
    n = cache_stats["sessions_with_data"]
    if n == 0:
        print("  no telemetry - run 'memlora telemetry <project_path>' to ingest")
    else:
        hit_pct = cache_stats["avg_cache_hit_rate"] * 100
        read = cache_stats["total_cache_read_tokens"]
        saved = cache_stats["effective_tokens_saved"]
        print(f"  sessions with data : {n}")
        print(f"  avg cache hit rate : {hit_pct:.1f}%")
        print(f"  cache reads served : {read:,} tok")
        print(f"  effective saved    : {saved:,} tok (read billed ~0.1x)")
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
    from memlora.storage.migrations import run_migrations

    config = Config.load()
    project_id = hash_project_path(args.project_path)
    db_path = get_db_path(config, project_id)

    with get_connection(db_path) as conn:
        run_migrations(conn)
        conn.execute("DELETE FROM extraction_job_acks")
        conn.execute("DELETE FROM extraction_jobs")
        conn.execute("DELETE FROM event_provenance")
        conn.execute("DELETE FROM events")
        conn.execute("DELETE FROM state_projections")
        conn.execute("DELETE FROM extraction_failures")
        conn.execute("DELETE FROM raw_evidence")
        conn.execute("DELETE FROM meta WHERE key != 'schema_version' AND key != 'projection_version'")
        conn.commit()

    print(f"Reset complete for project {project_id}.")


def _cmd_failures(args: argparse.Namespace) -> None:
    import datetime
    from memlora.integration.session import replay_job
    from memlora.storage.connection import get_connection, get_db_path, hash_project_path
    from memlora.storage.events import get_extraction_failures
    from memlora.storage.jobs import list_jobs
    from memlora.storage.migrations import run_migrations

    config = Config.load()
    project_id = hash_project_path(args.project_path)
    db_path = get_db_path(config, project_id)

    if not db_path.exists():
        print(f"No database found for {Path(args.project_path).resolve()}", file=sys.stderr)
        sys.exit(1)

    if getattr(args, "replay", None) is not None:
        try:
            stats = replay_job(args.project_path, args.replay, config=config)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        except Exception as exc:
            print(
                f"ERROR: replay of job {args.replay} failed during re-execution: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Replayed job {args.replay} — re-ran extraction.")
        print(json.dumps(stats, indent=2))
        return

    with get_connection(db_path) as conn:
        run_migrations(conn)
        failures = get_extraction_failures(conn, project_id, limit=args.limit)
        dead_jobs = list_jobs(conn, project_id, state="dead_lettered", limit=args.limit)

    if not failures and not dead_jobs:
        print("No extraction failures recorded.")
        return

    if dead_jobs:
        print(f"{len(dead_jobs)} dead-lettered extraction job(s):\n")
        for job in dead_jobs:
            ts = datetime.datetime.fromtimestamp(job.updated_at / 1000).strftime("%Y-%m-%d %H:%M:%S")
            sess = job.session_id[:12]
            err = (job.last_error or "")[:200]
            print(
                f"  job={job.id} [{ts}] session={sess} "
                f"stage={job.stage} class={job.failure_class}"
            )
            print(f"    {err}")
            print(f'    replay: memlora failures "{args.project_path}" --replay {job.id}')
            print()

    if not failures:
        return

    print(f"{len(failures)} legacy extraction failure(s):\n")
    for f in failures:
        ts = datetime.datetime.fromtimestamp(f["failed_at"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
        sess = f["session_id"][:12]
        print(f"  [{ts}] session={sess}  stage={f['stage']}")
        print(f"    {f['error_message'][:200]}")
        print()


def _cmd_rebuild(args: argparse.Namespace) -> None:
    from memlora.integration.session import rebuild_from_raw
    config = Config.load()
    stats = rebuild_from_raw(
        project_path=args.project_path,
        since_evidence_id=args.since,
        dry_run=args.dry_run,
        config=config,
    )
    print(json.dumps(stats, indent=2))


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
