"""JSONL-based telemetry ingestion — reads Claude Code session files for cache stats.

Claude Code stores sessions at ~/.claude/projects/<project_hash>/<session_id>.jsonl.
Assistant responses carry a `message.usage` dict with input/cache/output token counts.
This module parses those files and stores per-session aggregates in api_telemetry.

ONE RESPONSE, MANY LINES. Claude Code writes one JSONL line per content block (text,
thinking, tool_use), and every line of a response repeats that response's full
`usage`. Usage is therefore summed once per `message.id`, never once per line.
Summing per line counted a response once per block — 2-3x on real sessions, and by
a different factor per session depending on how many blocks its responses had, so
it did not merely scale the numbers but distorted comparisons between sessions.
Verified on real transcripts: 1,077 of 1,077 multi-line responses carried identical
usage on every line.

ROUND-TRIPS COGNIKERNEL ADDS. Each session row also counts the responses that exist
only because of CogniKernel's own tool surface: responses whose every tool call was a
CogniKernel memory tool, responses containing a read the PreToolUse gate denied, and
responses re-issuing a call that was denied. An agent with no memory makes none of
them. Measured on the four-project benchmark they accounted for more than all of
Relay's cost over the no-memory arm, which makes them the instrument G1 is judged on.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

# A tool whose name contains this is a CogniKernel memory tool (MCP names look like
# `mcp__cognikernel__recall`).
_MEMORY_TOOL_MARKER = "cognikernel"
# Every PreToolUse denial message begins with this (integration/lookup.py). Matching
# the marker, not the word "cognikernel", keeps a failed memory-tool call from being
# miscounted as a denied read.
_DENIAL_MARKER = "[CogniKernel]"
# Rows ingested before usage was counted per response (see migration 022).
_LEGACY_BASIS = "per_line_legacy"
_ROUND_TRIP_KEYS = ("responses", "memory_tool_responses", "denied_responses", "retried_denials")


def _tool_result_text(content: Any) -> str:
    """Flatten a tool_result's content — a string, or a list of text blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return ""


def ingest_session_jsonl(
    jsonl_path: Path,
    session_id: str,
    project_id: str,
) -> dict[str, Any]:
    """Parse a Claude Code JSONL session file and return aggregated usage stats.

    Only assistant messages carry usage data; user/meta/tool lines are skipped.
    Usage is counted once per API response (grouped by `message.id`, see the module
    docstring); a line with no id has nothing to group on and counts as its own
    response. `responses` is the number of API responses that carried usage.

    Each counted response also lands in at most one round-trip class, checked in
    this order: `denied_responses` (it holds a tool call whose result carries the
    PreToolUse denial marker), `retried_denials` (it re-issues the name and input of
    an earlier denied call), `memory_tool_responses` (every tool call it made was a
    CogniKernel memory tool). `usage_basis` is always 'per_response'.
    """
    entries: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    denied_tool_ids: set[str] = set()
    anonymous = 0

    try:
        text = jsonl_path.read_text(encoding="utf-8")
    except OSError:
        text = ""

    for raw_line in text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        line_type = obj.get("type")
        message = obj.get("message") or {}
        content = message.get("content")

        if line_type == "user":
            if isinstance(content, list):
                for block in content:
                    if (
                        isinstance(block, dict)
                        and block.get("type") == "tool_result"
                        and block.get("is_error")
                        and _DENIAL_MARKER in _tool_result_text(block.get("content"))
                    ):
                        denied_tool_ids.add(block.get("tool_use_id"))
            continue
        if line_type != "assistant":
            continue

        message_id = message.get("id")
        if not message_id:
            anonymous += 1
            message_id = f"__line_{anonymous}"
        entry = entries.get(message_id)
        if entry is None:
            entry = {"usage": {}, "tools": []}
            entries[message_id] = entry
            order.append(message_id)
        usage = message.get("usage") or {}
        if usage:
            # Every line of a response repeats the same usage; keeping the last
            # one seen is equivalent and tolerates a truncated earlier line.
            entry["usage"] = usage
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    entry["tools"].append((
                        block.get("id"),
                        block.get("name") or "",
                        json.dumps(block.get("input") or {}, sort_keys=True),
                    ))

    counted = [entries[key] for key in order if entries[key]["usage"]]

    memory_tool_responses = denied_responses = retried_denials = 0
    denied_calls: set[tuple[str, str]] = set()
    for entry in counted:
        tools = entry["tools"]
        calls = {(name, call_input) for _, name, call_input in tools}
        if any(tool_id in denied_tool_ids for tool_id, _, _ in tools):
            denied_responses += 1
        elif calls & denied_calls:
            retried_denials += 1
        elif tools and all(_MEMORY_TOOL_MARKER in name.lower() for _, name, _ in tools):
            memory_tool_responses += 1
        for tool_id, name, call_input in tools:
            if tool_id in denied_tool_ids:
                denied_calls.add((name, call_input))

    usages = [entry["usage"] for entry in counted]
    return {
        "project_id": project_id,
        "session_id": session_id,
        "input_tokens": sum(u.get("input_tokens", 0) or 0 for u in usages),
        "cache_creation_tokens": sum(u.get("cache_creation_input_tokens", 0) or 0 for u in usages),
        "cache_read_tokens": sum(u.get("cache_read_input_tokens", 0) or 0 for u in usages),
        "output_tokens": sum(u.get("output_tokens", 0) or 0 for u in usages),
        "responses": len(counted),
        "memory_tool_responses": memory_tool_responses,
        "denied_responses": denied_responses,
        "retried_denials": retried_denials,
        "usage_basis": "per_response",
    }


def store_telemetry(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """Upsert a telemetry row into api_telemetry. Replaces on (project_id, session_id).

    Round-trip counters default to 0 and `usage_basis` to 'per_response' when a
    caller omits them: the only producer of rows is `ingest_session_jsonl`, which
    counts usage per response. The column's own DEFAULT is 'per_line_legacy' on
    purpose (migration 022), so the basis is always written explicitly here.
    """
    now = int(time.time() * 1000)
    conn.execute(
        """
        INSERT INTO api_telemetry
            (project_id, session_id, input_tokens, cache_creation_tokens,
             cache_read_tokens, output_tokens, responses, memory_tool_responses,
             denied_responses, retried_denials, usage_basis, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (project_id, session_id) DO UPDATE SET
            input_tokens          = excluded.input_tokens,
            cache_creation_tokens = excluded.cache_creation_tokens,
            cache_read_tokens     = excluded.cache_read_tokens,
            output_tokens         = excluded.output_tokens,
            responses             = excluded.responses,
            memory_tool_responses = excluded.memory_tool_responses,
            denied_responses      = excluded.denied_responses,
            retried_denials       = excluded.retried_denials,
            usage_basis           = excluded.usage_basis,
            ingested_at           = excluded.ingested_at
        """,
        (
            row["project_id"],
            row["session_id"],
            row["input_tokens"],
            row["cache_creation_tokens"],
            row["cache_read_tokens"],
            row["output_tokens"],
            row.get("responses", 0),
            row.get("memory_tool_responses", 0),
            row.get("denied_responses", 0),
            row.get("retried_denials", 0),
            row.get("usage_basis", "per_response"),
            now,
        ),
    )
    conn.commit()


# cache_read tokens are billed at ~0.1x base input on Anthropic pricing, so the
# saving vs re-sending the same tokens uncached is ~90%, not 100%.
_CACHE_READ_DISCOUNT: float = 0.9


def get_cache_stats(conn: sqlite3.Connection, project_id: str) -> dict[str, Any]:
    """Return cache effectiveness summary for a project.

    Keys:
      sessions_with_data       — number of sessions with telemetry
      avg_cache_hit_rate       — mean(read / (input + cache_creation + read)) across sessions.
                                 cache_creation is in the denominator because those tokens were
                                 written to the cache (~1.25x), not served from it.
      total_cache_read_tokens  — Σ cache_read_tokens (tokens served from cache this period)
      effective_tokens_saved   — ≈ 0.9 × total_cache_read_tokens; the honest saving vs uncached
      recent_sessions          — list of last 10 rows, newest first
      legacy_sessions          — rows ingested while usage was summed per transcript line
                                 (usage_basis='per_line_legacy'). They are inflated 2-3x, so
                                 they are counted here and kept out of every figure above;
                                 re-ingesting a session rewrites its row per response.
      round_trips              — Σ responses / memory_tool_responses / denied_responses /
                                 retried_denials over per-response rows
      induced_share            — (memory-tool-only + denied + retried) / responses: the share
                                 of round-trips CogniKernel's own tools added
    """
    all_rows = conn.execute(
        """
        SELECT session_id, input_tokens, cache_creation_tokens,
               cache_read_tokens, output_tokens, ingested_at,
               responses, memory_tool_responses, denied_responses,
               retried_denials, usage_basis
        FROM api_telemetry
        WHERE project_id = ?
        ORDER BY ingested_at DESC
        """,
        (project_id,),
    ).fetchall()

    legacy_sessions = sum(1 for r in all_rows if r["usage_basis"] == _LEGACY_BASIS)
    rows = [r for r in all_rows if r["usage_basis"] != _LEGACY_BASIS]
    round_trips = {key: sum(r[key] for r in rows) for key in _ROUND_TRIP_KEYS}
    induced = (
        round_trips["memory_tool_responses"]
        + round_trips["denied_responses"]
        + round_trips["retried_denials"]
    )
    induced_share = induced / round_trips["responses"] if round_trips["responses"] else 0.0

    if not rows:
        return {
            "sessions_with_data": 0,
            "avg_cache_hit_rate": 0.0,
            "total_cache_read_tokens": 0,
            "effective_tokens_saved": 0,
            "recent_sessions": [],
            "legacy_sessions": legacy_sessions,
            "round_trips": round_trips,
            "induced_share": 0.0,
        }

    hit_rates: list[float] = []
    total_read = 0
    for row in rows:
        inp = row["input_tokens"]
        create = row["cache_creation_tokens"]
        read = row["cache_read_tokens"]
        denom = inp + create + read
        if denom > 0:
            hit_rates.append(read / denom)
        total_read += read

    avg_hit_rate = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0

    recent = [
        {
            "session_id": r["session_id"],
            "input_tokens": r["input_tokens"],
            "cache_read_tokens": r["cache_read_tokens"],
            "output_tokens": r["output_tokens"],
            "ingested_at": r["ingested_at"],
        }
        for r in rows[:10]
    ]

    return {
        "sessions_with_data": len(rows),
        "avg_cache_hit_rate": avg_hit_rate,
        "total_cache_read_tokens": total_read,
        "effective_tokens_saved": int(round(_CACHE_READ_DISCOUNT * total_read)),
        "recent_sessions": recent,
        "legacy_sessions": legacy_sessions,
        "round_trips": round_trips,
        "induced_share": induced_share,
    }


def whole_session_rollup(conn: sqlite3.Connection, project_id: str) -> dict[str, Any]:
    """Aggregate per-session totals into one whole-project token report.

    This is the top-line meter for between-mode/before-after comparison: the
    sum a session actually costs is input + cache_creation + cache_read (read
    weighted at ~0.1x gives the billed-equivalent). Returns raw sums plus a
    billed-equivalent input figure and per-session rows for drill-down.
    """
    rows = conn.execute(
        """
        SELECT session_id, input_tokens, cache_creation_tokens,
               cache_read_tokens, output_tokens
        FROM api_telemetry
        WHERE project_id = ?
        ORDER BY ingested_at ASC
        """,
        (project_id,),
    ).fetchall()

    totals = {"input": 0, "cache_creation": 0, "cache_read": 0, "output": 0}
    sessions: list[dict[str, int]] = []
    for r in rows:
        totals["input"] += r["input_tokens"]
        totals["cache_creation"] += r["cache_creation_tokens"]
        totals["cache_read"] += r["cache_read_tokens"]
        totals["output"] += r["output_tokens"]
        sessions.append({
            "session_id": r["session_id"],
            "input": r["input_tokens"],
            "cache_creation": r["cache_creation_tokens"],
            "cache_read": r["cache_read_tokens"],
            "output": r["output_tokens"],
        })

    # Billed-equivalent input tokens: cache_creation ~1.25x, cache_read ~0.1x.
    billed_equiv_input = (
        totals["input"]
        + int(round(1.25 * totals["cache_creation"]))
        + int(round(0.1 * totals["cache_read"]))
    )
    return {
        "sessions_with_data": len(rows),
        "totals": totals,
        "billed_equivalent_input_tokens": billed_equiv_input,
        "sessions": sessions,
    }


def find_and_ingest_telemetry(
    project_path: str | Path,
    config=None,
    claude_projects_dir: Path | None = None,
) -> dict[str, Any]:
    """Scan Claude Code's JSONL session files for known project sessions and ingest usage stats.

    Returns a summary: {"ingested": N, "skipped": M, "total_sessions_known": K}
    """
    from cognikernel.config import Config
    from cognikernel.storage.connection import get_connection, get_db_path, resolve_project_id

    # Project-aware load (H2): must resolve the same DB the hooks write to.
    config = config or Config.load(project_path=project_path)
    project_id = resolve_project_id(project_path, config)
    db_path = get_db_path(config, project_id)

    if not db_path.exists():
        return {"ingested": 0, "skipped": 0, "total_sessions_known": 0}

    with get_connection(db_path) as conn:
        known_sessions = [
            r[0] for r in conn.execute(
                "SELECT DISTINCT session_id FROM events WHERE project_id = ?",
                (project_id,),
            ).fetchall()
        ]

    if not known_sessions:
        return {"ingested": 0, "skipped": 0, "total_sessions_known": 0}

    claude_projects = claude_projects_dir if claude_projects_dir is not None else Path.home() / ".claude" / "projects"
    session_set = set(known_sessions)

    ingested = 0
    skipped = 0

    jsonl_by_session: dict[str, Path] = {}
    if claude_projects.is_dir():
        for project_dir in claude_projects.iterdir():
            if not project_dir.is_dir():
                continue
            for jsonl in project_dir.glob("*.jsonl"):
                stem = jsonl.stem
                if stem in session_set:
                    jsonl_by_session[stem] = jsonl

    with get_connection(db_path) as conn:
        for session_id in known_sessions:
            if session_id not in jsonl_by_session:
                skipped += 1
                continue
            stats = ingest_session_jsonl(jsonl_by_session[session_id], session_id, project_id)
            store_telemetry(conn, stats)
            ingested += 1

    return {
        "ingested": ingested,
        "skipped": skipped,
        "total_sessions_known": len(known_sessions),
    }
