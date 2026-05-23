"""JSONL-based telemetry ingestion — reads Claude Code session files for cache stats.

Claude Code stores sessions at ~/.claude/projects/<project_hash>/<session_id>.jsonl.
Each assistant message has a `message.usage` dict with input/cache/output token counts.
This module parses those files and stores per-session aggregates in api_telemetry.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def ingest_session_jsonl(
    jsonl_path: Path,
    session_id: str,
    project_id: str,
) -> dict[str, Any]:
    """Parse a Claude Code JSONL session file and return aggregated usage stats.

    Only assistant messages carry usage data; user/meta/tool lines are skipped.
    """
    input_tokens = 0
    cache_creation_tokens = 0
    cache_read_tokens = 0
    output_tokens = 0

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
        if obj.get("type") != "assistant":
            continue
        usage = obj.get("message", {}).get("usage", {})
        if not usage:
            continue
        input_tokens += usage.get("input_tokens", 0)
        cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
        cache_read_tokens += usage.get("cache_read_input_tokens", 0)
        output_tokens += usage.get("output_tokens", 0)

    return {
        "project_id": project_id,
        "session_id": session_id,
        "input_tokens": input_tokens,
        "cache_creation_tokens": cache_creation_tokens,
        "cache_read_tokens": cache_read_tokens,
        "output_tokens": output_tokens,
    }


def store_telemetry(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    """Upsert a telemetry row into api_telemetry. Replaces on (project_id, session_id)."""
    now = int(time.time() * 1000)
    conn.execute(
        """
        INSERT INTO api_telemetry
            (project_id, session_id, input_tokens, cache_creation_tokens,
             cache_read_tokens, output_tokens, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (project_id, session_id) DO UPDATE SET
            input_tokens          = excluded.input_tokens,
            cache_creation_tokens = excluded.cache_creation_tokens,
            cache_read_tokens     = excluded.cache_read_tokens,
            output_tokens         = excluded.output_tokens,
            ingested_at           = excluded.ingested_at
        """,
        (
            row["project_id"],
            row["session_id"],
            row["input_tokens"],
            row["cache_creation_tokens"],
            row["cache_read_tokens"],
            row["output_tokens"],
            now,
        ),
    )
    conn.commit()


def get_cache_stats(conn: sqlite3.Connection, project_id: str) -> dict[str, Any]:
    """Return cache effectiveness summary for a project.

    Keys:
      sessions_with_data  — number of sessions with telemetry
      avg_cache_hit_rate  — mean(cache_read / (input + cache_read)) across sessions (0.0–1.0)
      total_tokens_saved  — sum of cache_read_tokens (tokens served from cache)
      recent_sessions     — list of last 10 rows, newest first
    """
    rows = conn.execute(
        """
        SELECT session_id, input_tokens, cache_creation_tokens,
               cache_read_tokens, output_tokens, ingested_at
        FROM api_telemetry
        WHERE project_id = ?
        ORDER BY ingested_at DESC
        """,
        (project_id,),
    ).fetchall()

    if not rows:
        return {
            "sessions_with_data": 0,
            "avg_cache_hit_rate": 0.0,
            "total_tokens_saved": 0,
            "recent_sessions": [],
        }

    hit_rates: list[float] = []
    total_saved = 0
    for row in rows:
        inp = row["input_tokens"]
        read = row["cache_read_tokens"]
        total = inp + read
        if total > 0:
            hit_rates.append(read / total)
        total_saved += read

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
        "total_tokens_saved": total_saved,
        "recent_sessions": recent,
    }


def find_and_ingest_telemetry(
    project_path: str | Path,
    config=None,
    claude_projects_dir: Path | None = None,
) -> dict[str, Any]:
    """Scan Claude Code's JSONL session files for known project sessions and ingest usage stats.

    Returns a summary: {"ingested": N, "skipped": M, "total_sessions_known": K}
    """
    from memlora.config import Config
    from memlora.storage.connection import get_connection, get_db_path, hash_project_path

    config = config or Config.load()
    project_id = hash_project_path(project_path)
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
