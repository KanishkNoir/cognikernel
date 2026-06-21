"""In-process hook entrypoints (CK-6a).

Single home for the logic behind CogniKernel's Claude Code hooks, so each can be
invoked two ways with *identical* behavior:
  - ``python -m memlora hook-<event>`` — path-portable; what ``memlora init`` writes.
  - the thin ``scripts/memlora_*_hook.py`` shims — back-compat for older settings.json.

Every entrypoint is FAIL-OPEN: read stdin, do the work, print the hook's stdout
protocol if any, swallow all exceptions so a hook never blocks Claude. Heavy
imports stay lazy (inside the functions) so the hot path — ``hook-pretool`` on
every Read — pays only for what it uses, never the extraction/symbol stack.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

_REINJECT_SOURCES = frozenset({"startup", "resume", "compact", "clear"})

# Fail-open is the contract (a hook must never block Claude), but a swallowed
# exception that leaves no trace is indistinguishable from "nothing happened" —
# the silence-reads-as-success failure mode (audit P3). Log the swallow at WARNING
# with the traceback so a degraded hook is greppable; never re-raise.
_log = logging.getLogger("memlora.hooks")


def _log_swallowed(hook: str, exc: BaseException) -> None:
    """Record a fail-open swallow without ever raising from the logger itself."""
    try:
        _log.warning("hook.%s_swallowed_error: %s", hook, exc, exc_info=True)
    except Exception:
        pass


# ── shared helpers ────────────────────────────────────────────────────────────


def _read_payload(strip_bom: bool = False) -> dict:
    try:
        if strip_bom:
            raw = sys.stdin.buffer.read().decode("utf-8-sig", errors="replace")
        else:
            raw = sys.stdin.read()
        return json.loads(raw) if raw.strip() else {}
    except Exception:
        return {}


def _find_project_root(file_path: Path) -> Path | None:
    """Walk up to 12 parents for a dir containing .claude/settings.json."""
    current = file_path.resolve().parent if file_path.is_absolute() else file_path.parent
    for _ in range(12):
        if (current / ".claude" / "settings.json").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return None


def _pretool(decision: str, *, reason: str | None = None, context: str | None = None) -> None:
    out: dict = {"hookEventName": "PreToolUse", "permissionDecision": decision}
    if reason is not None:
        out["permissionDecisionReason"] = reason
    if context is not None:
        out["additionalContext"] = context
    print(json.dumps({"hookSpecificOutput": out}))


_HOOK_TIMEOUT_S = 3.0  # per-prompt budget; fail-open on breach


# ── UserPromptSubmit — query-time injection (CK-1) ────────────────────────────


def user_prompt_submit_main() -> None:
    """Inject a short memory snippet alongside the user's prompt (CK-1).

    Silence is the default: if nothing clears the relevance gate, print nothing
    and exit 0 (Claude Code treats no-stdout-on-exit-0 as a pass-through). This
    runs on every prompt; it must be fast — the embedding model is warmed at
    SessionStart to amortise cold-load. Flag-gated: exits immediately when
    query_time_injection is disabled.
    """
    import threading

    try:
        payload = _read_payload()
        cwd = payload.get("cwd", "")
        prompt = payload.get("prompt", "")
        if not cwd or not prompt:
            return

        from memlora.config import Config
        config = Config.load(project_path=cwd)
        if not config.query_time_injection:
            return

        from memlora.integration.query import recall_for_prompt
        session_id = payload.get("session_id") or None
        # Hard budget (audit P1). A `with ThreadPoolExecutor()` block joins the
        # worker on exit (shutdown(wait=True)), AND the executor registers an
        # atexit handler that joins all worker threads at interpreter shutdown —
        # so even shutdown(wait=False) would not let the hook *process* exit while
        # a recall is stalled on a write lock (up to busy_timeout=30s). The 3s
        # result-timeout was therefore fiction. Use a daemon thread instead: it is
        # abandoned on timeout and never joined at exit, so the hook always returns
        # within budget. recall is read-only; the orphaned reader dies with the
        # process and any uncommitted CK-1 ledger write rolls back harmlessly.
        result: dict[str, str] = {"snippet": ""}

        def _recall() -> None:
            try:
                result["snippet"] = recall_for_prompt(
                    cwd, prompt, config=config, session_id=session_id
                ) or ""
            except Exception:
                result["snippet"] = ""

        worker = threading.Thread(target=_recall, daemon=True)
        worker.start()
        worker.join(timeout=_HOOK_TIMEOUT_S)
        snippet = result["snippet"]  # "" if the worker is still running (timed out)

        if not snippet:
            return  # silence — print nothing, exit 0

        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": snippet,
            }
        }))
    except Exception as exc:
        _log_swallowed("user_prompt_submit", exc)  # never block the user


# ── SubagentStop — subagent memory capture (CK-4) ─────────────────────────────


def subagent_stop_main() -> None:
    """Capture a subagent transcript into the parent project DB (CK-4, async since I4).

    SubagentStop stdin provides transcript_path (direct JSONL path) and cwd
    (parent project dir) — no search required. Subagent events land with a
    capped authority so they cannot override the main agent or the user.

    Gamma post-mortem hardening:
    - If transcript_path is the MAIN session's JSONL (stem == session_id), skip:
      the Stop hook owns that transcript. Processing it here under the agent id
      double-extracts the whole session under a wrong session id and (pre-I4)
      held the write lock for the entire merge, starving every other writer.
    - Use the async capture+queue path (session_capture + detached worker), never
      the synchronous session_end — a hook must not run a multi-minute merge.
    - source_type must be one of migration 005's CHECK values. The old value
      'subagent_transcript' violated the CHECK and was silently dropped by
      INSERT OR IGNORE, breaking CK-4 storage from day one.
    """
    try:
        payload = _read_payload()
        transcript_path = payload.get("transcript_path", "")
        cwd = payload.get("cwd", "")
        session_id = payload.get("session_id", "")
        agent_id = payload.get("agent_id", "")

        if not transcript_path or not cwd:
            return

        from memlora.config import Config
        config = Config.load(project_path=cwd)
        if not config.capture_subagents:
            return

        jsonl = Path(transcript_path)
        if not jsonl.exists():
            return
        if session_id and jsonl.stem == session_id:
            return  # main transcript — the Stop hook owns it

        raw_jsonl = jsonl.read_text(encoding="utf-8", errors="replace")
        sub_session_id = agent_id or f"subagent_{session_id}"

        from memlora.integration.session import session_capture
        session_capture(
            cwd,
            sub_session_id,
            raw_jsonl,
            config=config,
            evidence_source_type="jsonl_transcript",
            evidence_source_path=transcript_path,
        )
        # No spawn: detached workers die with the hook's Job Object (I7c).
        # The job drains at the next Stop firing's sync drain or via the MCP
        # server's background drainer.
    except Exception as exc:
        _log_swallowed("subagent_stop", exc)  # never block subagent teardown


# ── PostToolUse:Grep — cache grep results (CK-3a) ─────────────────────────────


def posttool_grep_main() -> None:
    """Store Grep results in grep_cache (CK-3a).

    PostToolUse:Grep fires after the tool completes; we cache the result so the
    PreToolUse:Grep path can serve repeat identical searches from the DB. Enabled
    only when grep_cache_enabled = True. Never raises — fail-open.
    """
    payload = _read_payload(strip_bom=True)
    if payload.get("tool_name") != "Grep":
        return
    tool_input = payload.get("tool_input", {})
    pattern = tool_input.get("pattern", "")
    cwd = payload.get("cwd", "")
    if not pattern or not cwd:
        return
    path_filter = tool_input.get("path", "") or ""
    glob_filter = tool_input.get("glob", "") or ""
    result_text = payload.get("tool_response", "") or payload.get("tool_result", "") or ""
    if not isinstance(result_text, str):
        result_text = json.dumps(result_text)
    try:
        from memlora.config import Config
        from memlora.storage.connection import get_connection, get_db_path, hash_project_path
        from memlora.storage.grep_cache import store_grep_result
        from memlora.storage.migrations import run_migrations

        cfg = Config.load()
        if not cfg.grep_cache_enabled:
            return
        project_id = hash_project_path(cwd)
        db_path = get_db_path(cfg, project_id)
        if not db_path.exists():
            return
        with get_connection(db_path) as conn:
            run_migrations(conn)
            store_grep_result(conn, project_id, pattern, path_filter, glob_filter, result_text)
    except Exception:
        pass


# ── SessionStart ──────────────────────────────────────────────────────────────


def session_start_main() -> None:
    try:
        payload = _read_payload()
        if payload.get("source", "") not in _REINJECT_SOURCES:
            return
        cwd = payload.get("cwd", "")
        if not cwd:
            return
        from memlora.integration.session_start import handle_session_start
        context = handle_session_start(cwd, session_id=payload.get("session_id") or None)
        if not context:
            return
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": context,
            }
        }))

        # Kick the embedding model load in the background — NEVER block session
        # start on the multi-minute cold download. Backfill any vector-less events
        # only when the model is already resident (is_ready) so this stays O(ms);
        # on a cold start the long-lived MCP server completes the download and a
        # later `memlora warm` / extraction backfills. recall / per-prompt injection
        # stay correct throughout via the deterministic lexical fallback.
        try:
            from memlora.embedding.model import is_ready, warm
            warm()
            if cwd and is_ready():
                from memlora.config import Config as _Cfg
                from memlora.embedding.backfill import backfill_embeddings
                from memlora.storage.connection import (
                    get_connection as _gc,
                    get_db_path as _gd,
                    hash_project_path as _hp,
                )
                from memlora.storage.migrations import run_migrations as _rm
                _cfg = _Cfg.load(project_path=cwd)
                _pid = _hp(cwd)
                _db = _gd(_cfg, _pid)
                if _db.exists():
                    with _gc(_db) as _conn:
                        _rm(_conn)
                        backfill_embeddings(_conn, _pid)
        except Exception:
            pass
    except Exception as exc:
        _log_swallowed("pretool", exc)  # never block Claude


# ── PreToolUse (Read gate + optional Grep cache) ──────────────────────────────


def pretool_main() -> None:
    try:
        payload = _read_payload(strip_bom=True)
    except Exception:
        _pretool("allow")
        return
    tool_name = payload.get("tool_name", "")
    if tool_name == "Read":
        _pretool_read(payload)
    elif tool_name == "Grep":
        _pretool_grep(payload)
    elif tool_name in ("Write", "Edit", "MultiEdit"):
        _pretool_edit(payload)
    else:
        _pretool("allow")


def _pretool_read(payload: dict) -> None:
    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd", "")
    if not file_path:
        _pretool("allow")
        return
    try:
        from memlora.config import Config
        from memlora.integration.lookup import decide_pretool_read
        from memlora.storage.connection import get_connection, get_db_path, hash_project_path
        from memlora.storage.migrations import run_migrations

        project_root = _find_project_root(Path(file_path))
        if project_root is None:
            project_root = Path(cwd) if cwd else Path(file_path).parent

        config = Config.load(project_path=project_root)
        project_id = hash_project_path(str(project_root))
        db_path = get_db_path(config, project_id)
        if not db_path.exists():
            _pretool("allow")
            return

        retry_window_ms = config.deny_retry_window_seconds * 1000
        with get_connection(db_path) as conn:
            run_migrations(conn)
            decision = decide_pretool_read(
                conn,
                project_id=project_id,
                session_id=session_id or "__unknown__",
                file_path=file_path,
                project_path=str(project_root),
                policy=config.hook_policy,
                retry_window_ms=retry_window_ms,
            )

        if decision.is_deny:
            _pretool("deny", reason=decision.message)
        elif decision.outcome_hint == "body_needed_retry":
            _pretool("allow", context=(
                "[CogniKernel] body-needed retry granted — record this read in your "
                "context; the next attempt to re-read this file will be denied."
            ))
        else:
            _pretool("allow")
    except Exception:
        traceback.print_exc(file=sys.stderr)
        _pretool("allow")


def _edit_diff_text(tool_name: str, tool_input: dict) -> str:
    """The new code a Write/Edit/MultiEdit is about to write — the surface a
    prohibition would be violated on. Empty when nothing new is being written."""
    if tool_name == "Write":
        return tool_input.get("content", "") or ""
    if tool_name == "Edit":
        return tool_input.get("new_string", "") or ""
    if tool_name == "MultiEdit":
        edits = tool_input.get("edits", []) or []
        return "\n".join(
            e.get("new_string", "") for e in edits if isinstance(e, dict)
        )
    return ""


def _pretool_edit(payload: dict) -> None:
    """K2 — JIT bind at the action point. ALWAYS allows; on a prohibition match
    it attaches an advisory `additionalContext` reminding the agent of the prior
    decision. Fail-open: any error → plain allow, never a deny."""
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {}) or {}
    file_path = tool_input.get("file_path", "")
    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd", "")
    diff_text = _edit_diff_text(tool_name, tool_input)
    if not diff_text.strip():
        _pretool("allow")
        return
    try:
        from memlora.config import Config
        from memlora.integration.query import surface_prohibitions_for_edit

        project_root = _find_project_root(Path(file_path)) if file_path else None
        if project_root is None:
            project_root = Path(cwd) if cwd else (
                Path(file_path).parent if file_path else None
            )
        if project_root is None:
            _pretool("allow")
            return
        config = Config.load(project_path=project_root)
        context = surface_prohibitions_for_edit(
            str(project_root),
            diff_text,
            file_path=file_path,
            config=config,
            session_id=session_id or "__unknown__",
        )
        if context:
            _pretool("allow", context=context)
        else:
            _pretool("allow")
    except Exception:
        traceback.print_exc(file=sys.stderr)
        _pretool("allow")


def _pretool_grep(payload: dict) -> None:
    tool_input = payload.get("tool_input", {})
    cwd = payload.get("cwd", "")
    pattern = tool_input.get("pattern", "")
    if not pattern or not cwd:
        _pretool("allow")
        return
    path_filter = tool_input.get("path", "") or ""
    glob_filter = tool_input.get("glob", "") or ""
    try:
        from memlora.config import Config
        from memlora.storage.connection import get_connection, get_db_path, hash_project_path
        from memlora.storage.grep_cache import lookup_grep_result
        from memlora.storage.migrations import run_migrations

        cfg = Config.load()
        if not cfg.grep_cache_enabled:
            _pretool("allow")
            return
        project_id = hash_project_path(cwd)
        db_path = get_db_path(cfg, project_id)
        if not db_path.exists():
            _pretool("allow")
            return
        with get_connection(db_path) as conn:
            run_migrations(conn)
            cached = lookup_grep_result(conn, project_id, pattern, path_filter, glob_filter)
        if cached is not None:
            _pretool("deny", reason=(
                f"[CogniKernel grep-cache] Pattern `{pattern}` matched "
                f"{path_filter or '(all)'} — cached result:\n\n{cached}"
            ))
        else:
            _pretool("allow")
    except Exception:
        traceback.print_exc(file=sys.stderr)
        _pretool("allow")


# ── PostToolUse (Write/Edit → symbol graph) ───────────────────────────────────


def posttool_main() -> None:
    payload = _read_payload(strip_bom=True)
    tool_name = payload.get("tool_name", "")
    if tool_name not in ("Write", "Edit"):
        return
    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    session_id = payload.get("session_id", "")
    if not file_path:
        return
    abs_path = Path(file_path).resolve()
    if not abs_path.exists():
        return
    project_path = _find_project_root(abs_path)
    if project_path is None:
        return
    try:
        from memlora.config import Config
        from memlora.extraction.git_augment import FileChange
        from memlora.storage.connection import get_connection, get_db_path, hash_project_path
        from memlora.storage.migrations import run_migrations
        from memlora.symbols.extractor import build_symbol_update
        from memlora.symbols.store import apply_symbol_update

        config = Config.load(project_path=project_path)
        project_id = hash_project_path(project_path)
        db_path = get_db_path(config, project_id)
        if not db_path.exists():
            return
        rel_path = str(abs_path.relative_to(Path(project_path).resolve())).replace("\\", "/")
        changed_files = [FileChange(path=rel_path, change_type="modified", lines_changed=0)]
        update = build_symbol_update(project_id, str(project_path), changed_files)
        with get_connection(db_path) as conn:
            run_migrations(conn)
            apply_symbol_update(
                conn, update,
                project_path=str(project_path),
                session_id=session_id,
                last_action=tool_name,
            )
            if config.grep_cache_enabled:
                from memlora.storage.grep_cache import invalidate_project_cache
                invalidate_project_cache(conn, project_id, changed_path=rel_path)
    except Exception as exc:
        _log_swallowed("posttool", exc)  # posttool hook must never block Claude


# ── PostToolUse (Read → read_session_cache) ───────────────────────────────────


def posttool_read_main() -> None:
    payload = _read_payload(strip_bom=True)
    if payload.get("tool_name") != "Read":
        return
    tool_input = payload.get("tool_input", {})
    file_path = tool_input.get("file_path", "")
    session_id = payload.get("session_id", "")
    cwd = payload.get("cwd", "")
    if not file_path or not session_id:
        return
    try:
        from memlora.config import Config
        from memlora.integration.lookup import resolve_post_read_outcome
        from memlora.storage import read_cache as rc
        from memlora.storage.connection import get_connection, get_db_path, hash_project_path
        from memlora.storage.migrations import run_migrations
        from memlora.utils.paths import canonicalize_path

        project_root = _find_project_root(Path(file_path))
        if project_root is None:
            project_root = Path(cwd) if cwd else Path(file_path).parent
        config = Config.load(project_path=project_root)
        project_id = hash_project_path(str(project_root))
        db_path = get_db_path(config, project_id)
        if not db_path.exists():
            return
        canonical = canonicalize_path(file_path, str(project_root))
        if not canonical:
            return
        retry_window_ms = config.deny_retry_window_seconds * 1000
        with get_connection(db_path) as conn:
            run_migrations(conn)
            outcome = resolve_post_read_outcome(
                conn,
                project_id=project_id,
                session_id=session_id,
                canonical_path=canonical,
                retry_window_ms=retry_window_ms,
            )
            rc.record_read(conn, project_id, session_id, canonical, outcome=outcome)
    except Exception:
        traceback.print_exc(file=sys.stderr)


# ── Stop (session-end extraction) ─────────────────────────────────────────────


def _warn(msg: str) -> None:
    print(msg, file=sys.stderr)


def stop_main() -> None:
    payload = _read_payload()
    session_id = payload.get("session_id", "")
    project_dir = payload.get("cwd", payload.get("project_dir", ""))
    if not session_id:
        _warn("memlora hook-stop: no session_id in payload — skipping extraction")
        return
    if not project_dir:
        _warn("memlora hook-stop: no cwd/project_dir in payload — skipping extraction")
        return

    claude_projects = Path.home() / ".claude" / "projects"
    project_path = Path(project_dir).resolve()
    jsonl_path: Path | None = None
    try:
        for candidate_dir in claude_projects.iterdir():
            candidate = candidate_dir / f"{session_id}.jsonl"
            if candidate.exists():
                jsonl_path = candidate
                break
    except Exception:
        pass
    if jsonl_path is None:
        _warn(f"memlora hook-stop: JSONL not found for session {session_id} — skipping")
        return

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

    # I4/I7c: capture (fast, evidence+enqueue) then a SYNCHRONOUS time-budgeted
    # drain. Detached workers are useless under Claude Code on Windows — the
    # hook's Job Object kills the whole tree at hook exit (after any liveness
    # probe passes), so the only drain that reliably happens during a session
    # is the one this hook runs itself. The worker single-flight lock makes the
    # sync drain a no-op when the MCP server's background drainer is already
    # working the queue. Per-turn cost with a warm cursor: one delta job,
    # ~2-15s; budget-capped so the hook always fits its 60s ceiling.
    cmd = [
        sys.executable, "-m", "memlora", "capture",
        str(project_path), str(jsonl_path), "--auto-session-id", "--no-spawn",
    ]
    git_diff_file = None
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            _warn(f"memlora hook-stop: capture failed (rc={result.returncode}): {result.stderr[:300]}")
        else:
            _warn(f"memlora hook-stop: captured session {session_id} → {result.stdout.strip()[:200]}")
    except subprocess.TimeoutExpired:
        _warn("memlora hook-stop: capture timed out after 15s — jobs stay queued for the next drain")
    except Exception as exc:
        _warn(f"memlora hook-stop: unexpected error: {exc}")
    finally:
        if git_diff_file is not None:
            try:
                Path(git_diff_file.name).unlink(missing_ok=True)
            except Exception:
                pass

    # Drain the queue inside the hook's own lifetime (survives the Job Object).
    # 150s budget / 180s kill: each drain subprocess pays a cold model load
    # (~20s) before its first job; the old 40s ceiling killed nearly every
    # drain mid-job (GAMMA_CK_TEST: 30+ min queue lag). Requires the Stop hook
    # timeout in settings.json to be >= 200s (init now writes 300).
    try:
        drain = subprocess.run(
            [sys.executable, "-m", "memlora", "process-jobs", str(project_path),
             "--time-budget", "150"],
            capture_output=True, text=True, timeout=180,
        )
        _warn(f"memlora hook-stop: drain → {drain.stdout.strip()[:160]}")
    except subprocess.TimeoutExpired:
        _warn("memlora hook-stop: drain hit the 180s ceiling — remaining jobs stay queued")
    except Exception as exc:
        _warn(f"memlora hook-stop: drain error: {exc}")
