"""Pull Codex CLI session memory into the shared store (Sprint L / L2).

Codex has no Stop-hook equivalent, but it records every session as a rollout JSONL
under ``~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl`` whose ``session_meta`` header
carries the working directory. This module scans those rollouts, keeps the ones
whose ``cwd`` maps to a given project, and feeds each through the *same*
``session_capture`` path Claude Code uses — so the entire delta/dedup/idempotency
machinery applies unchanged and re-scanning is a cheap no-op.

Driven at the handoff boundary (Claude's SessionStart, or ``cognikernel codex-sync``),
this is what closes the Codex -> Claude half of the cross-platform loop. It never
raises: a missing dir, an unreadable file, or a malformed rollout degrades to
"captured nothing", logged at WARNING.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from cognikernel.config import Config
from cognikernel.storage.connection import project_paths_equivalent, resolve_project_id

_log = logging.getLogger("cognikernel.codex_sync")

# Bytes-ish prefix read per rollout for the cwd check. The session_meta header is
# normally line 1; 64K covers ordering tolerance without paying a full read of
# every large rollout on every SessionStart just to discover it's another
# project's session.
_HEADER_PREFIX_CHARS = 64 * 1024


def codex_sessions_root(config: Config | None = None) -> Path:
    """Resolve the Codex sessions directory: config override, $CODEX_HOME, else ~/.codex."""
    if config is not None and config.codex_home is not None:
        home = Path(config.codex_home)
    else:
        env = os.environ.get("CODEX_HOME")
        home = Path(env) if env else Path.home() / ".codex"
    return home / "sessions"


def _read_session_meta(text: str) -> dict[str, Any] | None:
    """Return the session_meta payload (cwd, id, ...) from the first lines, or None.

    The header is normally line 1; scan a small prefix to be tolerant of ordering.
    """
    for raw_line in text.splitlines()[:5]:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if obj.get("type") == "session_meta" and isinstance(obj.get("payload"), dict):
            return obj["payload"]
    return None


def sync_codex_rollouts(
    project_path: str | Path,
    config: Config | None = None,
    *,
    scan_window_days: int | None = None,
    spawn_worker: bool = False,
) -> dict[str, Any]:
    """Capture Codex rollouts belonging to *project_path* into its store.

    Returns {"scanned", "matched", "captured", "jobs", "enabled"}. Fail-open:
    any error is swallowed (logged) and reflected as a partial/zero result.
    """
    stats: dict[str, Any] = {"scanned": 0, "matched": 0, "captured": 0, "jobs": 0, "enabled": True}
    try:
        config = config or Config.load(project_path=project_path)
        if not config.codex_sync_enabled:
            stats["enabled"] = False
            return stats

        root = codex_sessions_root(config)
        if not root.is_dir():
            return stats

        window_days = scan_window_days if scan_window_days is not None else config.codex_scan_window_days
        cutoff = time.time() - window_days * 86400 if window_days and window_days > 0 else 0.0
        target_id = resolve_project_id(project_path, config)

        # Imported lazily: session_capture lives in the integration layer alongside
        # us, but keeping the import local avoids a heavy import at module load.
        from cognikernel.integration.session import session_capture

        for rollout in sorted(root.rglob("rollout-*.jsonl")):
            try:
                if cutoff and rollout.stat().st_mtime < cutoff:
                    continue
                stats["scanned"] += 1
                # Header check on a bounded prefix; the full (possibly large)
                # rollout is read only once the cwd actually matches.
                with open(rollout, encoding="utf-8", errors="replace") as f:
                    head = f.read(_HEADER_PREFIX_CHARS)
                meta = _read_session_meta(head)
                if not meta:
                    continue
                cwd = meta.get("cwd")
                if not cwd or not _cwd_matches_project(cwd, project_path, config, target_id):
                    continue
                stats["matched"] += 1
                text = rollout.read_text(encoding="utf-8", errors="replace")
                session_id = str(meta.get("id") or rollout.stem)
                result = session_capture(
                    project_path,
                    session_id=session_id,
                    raw_jsonl=text,
                    config=config,
                    evidence_source_type="codex_rollout",
                    evidence_source_path=str(rollout),
                )
                if result.get("job_id") is not None:
                    stats["captured"] += 1
                    stats["jobs"] += 1
            except Exception as exc:  # one bad rollout must not abort the scan
                _log.warning("codex_sync.rollout_failed", extra={"file": str(rollout), "error": repr(exc)})
                continue

        if spawn_worker and stats["jobs"] > 0:
            _spawn_worker(project_path)

        _log.info("codex_sync.done", extra=stats)
        return stats
    except Exception as exc:
        _log.warning("codex_sync.failed", extra={"error": repr(exc)})
        return stats


def _spawn_worker(project_path: str | Path) -> None:
    """Best-effort detached worker drain (mirror of the capture command's spawn)."""
    try:
        import subprocess
        import sys

        subprocess.Popen(
            [sys.executable, "-m", "cognikernel", "process-jobs", str(project_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except Exception as exc:
        _log.warning("codex_sync.spawn_failed", extra={"error": repr(exc)})


def _cwd_matches_project(
    cwd: str,
    project_path: str | Path,
    config: Config,
    target_id: str,
) -> bool:
    if project_paths_equivalent(cwd, project_path):
        return True
    if config.project_identity:
        try:
            cwd_config = Config.load(project_path=cwd)
            return (
                cwd_config.project_identity == config.project_identity
                and resolve_project_id(cwd, cwd_config) == target_id
            )
        except Exception:
            return False
    return False
