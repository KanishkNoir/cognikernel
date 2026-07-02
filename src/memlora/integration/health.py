"""Subsystem health checks — the diagnostic spine for fail-open (audit P3 / #66).

Fail-open keeps a session alive when a subsystem degrades, but without a health
surface a degraded subsystem looks identical to a healthy one — which is how
silent degradation (a missing FTS5 build, an embedding model that never loaded, a
queue full of dead-letters) read as "working" for a long time. These checks make
degradation legible: ``memlora doctor`` prints them and ``memlora doctor
--strict`` exits nonzero if any subsystem is unhealthy, so a pre-flight or CI can
catch what fail-open would otherwise hide.

Each check is itself fail-open (a probe that raises is reported as unhealthy, it
never crashes doctor).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from memlora.config import EXPECTED_SCHEMA_VERSION, Config


@dataclass
class HealthCheck:
    name: str
    ok: bool
    detail: str


def check_config(project_path: str | None) -> HealthCheck:
    """Config files parse cleanly (global + project overlay).

    Config.load is fail-open per key — an invalid value degrades to the default
    with a WARNING that hooks swallow. Without this check, a typo'd
    `.memlora/config.toml` silently downgrades behavior (e.g. hook_policy back
    to advisory) with no visible signal anywhere.
    """
    from memlora.config import Config
    try:
        _, issues = Config.load_with_issues(project_path=project_path)
    except Exception as exc:
        return HealthCheck("config", False, f"config load failed ({exc})")
    if issues:
        return HealthCheck("config", False, "; ".join(issues))
    return HealthCheck("config", True, "global + project config parse clean")


def check_schema_version(conn: sqlite3.Connection) -> HealthCheck:
    try:
        v = int(conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()[0])
    except Exception as exc:
        return HealthCheck("schema", False, f"schema_version unreadable ({exc})")
    if v == EXPECTED_SCHEMA_VERSION:
        return HealthCheck("schema", True, f"v{v} (current)")
    return HealthCheck(
        "schema", False,
        f"v{v} != expected v{EXPECTED_SCHEMA_VERSION} — migrations not applied",
    )


def check_fts(conn: sqlite3.Connection) -> HealthCheck:
    try:
        from memlora.storage.fts import fts_available
        avail = fts_available(conn)
    except Exception as exc:
        return HealthCheck("fts5", False, f"probe failed ({exc})")
    if avail:
        return HealthCheck("fts5", True, "available (lexical retrieval active)")
    return HealthCheck(
        "fts5", False,
        "unavailable in this SQLite build — lexical retrieval axis is disabled",
    )


def check_embedding(config: Config) -> HealthCheck:
    if not config.embedding_enabled:
        return HealthCheck("embedding", True, "disabled by config (lexical-only)")
    try:
        from memlora.embedding import model
        if model.is_ready() or model.is_available():
            return HealthCheck("embedding", True, "model loaded")
        return HealthCheck(
            "embedding", False,
            "model failed to load — semantic recall falls back to lexical",
        )
    except Exception as exc:
        return HealthCheck(
            "embedding", False,
            f"model error ({exc}) — semantic recall falls back to lexical",
        )


def check_symbol_extraction() -> HealthCheck:
    try:
        from memlora.symbols.extractor import typescript_support_status
        ts_ok, ts_detail = typescript_support_status()
    except Exception as exc:
        return HealthCheck("symbols", False, f"probe failed ({exc})")
    if ts_ok:
        return HealthCheck("symbols", True, "python ast + typescript OK")
    return HealthCheck(
        "symbols", False,
        f"python ast OK; typescript unavailable — {ts_detail} "
        "(TS/JS files yield no symbol graph)",
    )


def check_worker_queue(conn: sqlite3.Connection, project_id: str) -> HealthCheck:
    try:
        from memlora.storage.jobs import list_jobs
        dead = len(list_jobs(conn, project_id, state="dead_lettered", limit=1000))
        retry = len(list_jobs(conn, project_id, state="retryable_failure", limit=1000))
    except Exception as exc:
        return HealthCheck("worker_queue", False, f"probe failed ({exc})")
    if dead == 0:
        return HealthCheck("worker_queue", True, f"no dead-letters ({retry} retryable)")
    return HealthCheck(
        "worker_queue", False,
        f"{dead} dead-lettered job(s) — extraction silently dropped; "
        "inspect failure_class and replay",
    )


def check_codex(config: Config) -> HealthCheck:
    """Cross-platform capture wiring (Sprint L). Codex is optional, so its absence
    is HEALTHY (like embedding disabled) — this surfaces whether sync *can* run."""
    if not config.codex_sync_enabled:
        return HealthCheck("codex", True, "sync disabled by config")
    try:
        from memlora.extraction.codex_converter import codex_rollout_to_transcript  # noqa: F401
        from memlora.integration.codex_sync import codex_sessions_root
        root = codex_sessions_root(config)
    except Exception as exc:
        return HealthCheck("codex", False, f"codex_sync probe failed ({exc})")
    if not root.is_dir():
        return HealthCheck("codex", True, f"no Codex sessions at {root} — nothing to sync")
    try:
        n = sum(1 for _ in root.rglob("rollout-*.jsonl"))
    except Exception:
        n = -1
    return HealthCheck("codex", True, f"sessions dir present ({n} rollouts) — cross-platform capture active")


def run_health_checks(
    conn: sqlite3.Connection,
    project_id: str,
    config: Config,
    project_path: str | None = None,
) -> list[HealthCheck]:
    """Run every subsystem probe. Order: foundational first, optional last."""
    return [
        check_config(project_path),
        check_schema_version(conn),
        check_fts(conn),
        check_embedding(config),
        check_symbol_extraction(),
        check_worker_queue(conn, project_id),
        check_codex(config),
    ]
