"""PreToolUse:Read decision logic — Stage C1.

The decision tree (per v2 plan §2):

  STEP 1 — RE-READ CHECK (always runs):
    If (project_id, session_id, file_path) is in read_session_cache → DENY.

  STEP 2 — SKELETON-BASED GATING (only under strict policy):
    Lookup canonical_path in symbol_files:
      Case A  freshness='fresh' AND scan_status='scanned' AND symbol_count > 0:
              Check denied_reads for the same (project, session, path):
                If within retry window  → ALLOW as 'body_needed_retry'
                Otherwise               → DENY (record in denied_reads)
      Case B  freshness='stale'         → ALLOW (skeleton may be out of date)
      Case C  scan_status in {parse_error, ignored} → ALLOW (no signatures to offer)
      Case D  symbol_count = 0          → ALLOW (no public surface to defer to)
      Case E  no symbol_files row at all → ALLOW (genuinely new file)

Under `advisory` policy, STEP 2 is skipped entirely — every non-re-read goes
through (the v1 behaviour, kept available for one minor version per the plan).

This module is the single source of truth for the policy. The hook script is a
thin wrapper that translates the Decision dataclass to Claude Code's JSON
permission-decision protocol.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from memlora.config import Config
from memlora.storage import denied_reads as dr
from memlora.storage import read_cache as rc
from memlora.storage import symbol_files as sf
from memlora.utils.paths import canonicalize_path


# ── public types ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Decision:
    """Outcome of the PreToolUse:Read decision tree.

    `allow`           — Claude Code lets the Read proceed.
    `deny`            — Claude Code blocks the Read; `message` is shown to Claude.
    `outcome_hint`    — for ALLOW only; tells PostToolUse:Read what to record
                        in read_session_cache ('ok' or 'body_needed_retry').
                        None means PostToolUse decides on its own (legacy).
    `reason`          — debug label for logs/tests; not shown to Claude.
    """
    action: str                       # 'allow' | 'deny'
    message: str = ""
    outcome_hint: str | None = None   # 'ok' | 'body_needed_retry' | None
    reason: str = ""

    @property
    def is_deny(self) -> bool:
        return self.action == "deny"


# ── primary entrypoint ───────────────────────────────────────────────────────


def decide_pretool_read(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    file_path: str,
    project_path: str,
    *,
    policy: str = "strict",
    retry_window_ms: int = 60_000,
    now_ms: int | None = None,
) -> Decision:
    """Run the C1 decision tree for a PreToolUse:Read.

    Inputs are pre-validated; caller (the hook script) is responsible for
    extracting them from the tool payload.

    `file_path` may be absolute or relative — internally normalized to a
    canonical relative-from-project path via memlora.utils.paths. When the
    path cannot be canonicalized (outside project root, escape attempt, etc.),
    we allow the read so the hook never blocks paths it doesn't own.
    """
    canonical = canonicalize_path(file_path, project_path)
    if not canonical:
        return Decision(action="allow", reason="path_outside_project")

    # ── STEP 1 — re-read check (universal) ───────────────────────────────────
    was_read, last_outcome = rc.was_read_in_session(conn, project_id, session_id, canonical)
    if was_read:
        if last_outcome == "body_needed_retry":
            return Decision(
                action="deny",
                message=(
                    f"[CogniKernel] {canonical} body was already provided in a previous "
                    f"read this session — cite the existing content instead of re-reading."
                ),
                reason="rereread_after_body_retry",
            )
        # last_outcome == 'ok'
        return Decision(
            action="deny",
            message=(
                f"[CogniKernel] {canonical} was already read in this session — "
                f"its content is in your context. Cite it directly rather than re-reading."
            ),
            reason="reread_same_session",
        )

    # ── advisory mode skips STEP 2 ───────────────────────────────────────────
    if policy != "strict":
        return Decision(action="allow", reason="advisory_policy")

    # ── STEP 2 — skeleton-based gating (strict only) ─────────────────────────
    file_row = sf.get(conn, project_id, canonical)

    if file_row is None:
        return Decision(action="allow", reason="not_in_symbol_files")

    if file_row.freshness == "stale":
        return Decision(action="allow", reason="symbol_files_stale")

    if file_row.scan_status in ("parse_error", "ignored"):
        return Decision(action="allow", reason=f"scan_status_{file_row.scan_status}")

    if file_row.scan_status == "pending":
        # Edge: file row exists but symbols haven't been scanned yet. Be permissive.
        return Decision(action="allow", reason="scan_status_pending")

    # scan_status == "scanned" past this point
    if file_row.symbol_count == 0:
        return Decision(action="allow", reason="no_public_symbols")

    # Case A — fresh, scanned, has symbols. Apply 60s retry escape hatch.
    if dr.was_denied_within(
        conn, project_id, session_id, canonical,
        window_ms=retry_window_ms,
        now_ms=now_ms,
    ):
        # Second-attempt allowance. Clear the denial so a future Edit-cycle
        # starts a clean denial timer.
        dr.clear(conn, project_id, session_id, canonical)
        return Decision(
            action="allow",
            outcome_hint="body_needed_retry",
            reason="body_needed_retry_within_window",
        )

    # First denial — record it and tell Claude what's available.
    dr.record(conn, project_id, session_id, canonical, reason="skeleton_fresh", now_ms=now_ms)
    return Decision(
        action="deny",
        message=(
            f"[CogniKernel] {canonical} signatures are listed in the §Codebase skeleton "
            f"section of your session context. Use them. If you genuinely need the function "
            f"body (e.g., to replace an implementation), retry this Read once within 60 seconds "
            f"and it will be allowed."
        ),
        reason="skeleton_fresh_first_denial",
    )


# ── post-tool outcome resolution ─────────────────────────────────────────────


def resolve_post_read_outcome(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    canonical_path: str,
    *,
    retry_window_ms: int = 60_000,
    now_ms: int | None = None,
) -> str:
    """Determine what outcome to record in read_session_cache after a successful Read.

    PostToolUse:Read fires only on success (per Anthropic docs verified during C0).
    Because PreToolUse may have allowed the read as a body_needed_retry, we need
    to detect that situation here. We DO NOT clear denied_reads here — that's
    PreToolUse's responsibility when it consumes the retry allowance.

    Logic:
      If the read just succeeded but `denied_reads` still has a recent row,
      that means PreToolUse denied the FIRST attempt but allowed the SECOND;
      record as 'body_needed_retry'. Otherwise record as 'ok'.

    In the current PreToolUse implementation, the row is cleared inside
    decide_pretool_read() the moment the retry is granted, so by the time
    PostToolUse runs, the row is already gone. We keep this query as
    defense-in-depth in case the clear ever races.
    """
    if dr.was_denied_within(
        conn, project_id, session_id, canonical_path,
        window_ms=retry_window_ms,
        now_ms=now_ms,
    ):
        return "body_needed_retry"
    return "ok"


# ── legacy CLI shim ──────────────────────────────────────────────────────────


_LEGACY_ALLOW_STATUSES = frozenset({"modified", "in_flux", "added", "deleted"})


def lookup_file(
    project_path: str,
    file_path: str,
    config: Config | None = None,
) -> tuple[int, str]:
    """Legacy CLI subcommand entrypoint — kept for `memlora lookup` debugging.

    The PreToolUse hook no longer routes through this function (it imports
    decide_pretool_read directly to avoid subprocess overhead). This shim
    exists so `python -m memlora lookup <project> <file>` continues to work
    as an admin/debugging utility. It runs the C1 strict-mode tree with a
    synthetic session_id ("__cli__") so re-read protection never fires.
    """
    from memlora.storage.connection import get_connection, get_db_path, hash_project_path

    config = config or Config.load()
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)

    if not db_path.exists():
        return 1, ""

    with get_connection(db_path) as conn:
        decision = decide_pretool_read(
            conn,
            project_id=project_id,
            session_id="__cli__",
            file_path=file_path,
            project_path=project_path,
            policy=config.hook_policy,
        )

    if decision.is_deny:
        return 0, decision.message
    return 1, ""
