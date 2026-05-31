"""Agent-facing memory queries (CK-2) — backs the `recall` / `find_related` MCP tools.

The PULL surface: the agent asks memory a question on demand instead of relying only
on the session-start block. Returns compact, token-bounded text.

Invariants honored:
  - Determinism degrades, never crashes: semantic when embeddings are available,
    else a deterministic token-overlap (lexical) scan.
  - Never raises into the MCP layer: internal errors return a safe message string.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from memlora.config import Config
from memlora.storage.connection import get_connection, get_db_path, hash_project_path
from memlora.storage.migrations import run_migrations

_MAX_DESC = 140
# Event types always in the static session-start block — skip them in per-prompt
# injection to avoid re-injecting what the agent already has in context.
_ALWAYS_INJECTED: frozenset[str] = frozenset({
    "CONSTRAINT_HARD", "APPROACH_ABANDONED_DO_NOT_RETRY",
})


def _resolve(project_path: str, config: Config | None) -> tuple[str, Path]:
    config = config or Config.load(project_path=project_path)
    project_id = hash_project_path(project_path)
    return project_id, get_db_path(config, project_id)


def _lexical_recall(
    conn: sqlite3.Connection, project_id: str, query: str, limit: int
) -> list[dict]:
    """Deterministic token-overlap fallback used when embeddings are unavailable.

    Jaccard over normalized description tokens — same normalization as supersession,
    so behavior is consistent with the lexical merge path.
    """
    from memlora.delta.supersede import normalize_for_overlap

    q = normalize_for_overlap(query)
    if not q:
        return []
    scored: list[tuple[float, int, str, dict]] = []
    for r in conn.execute(
        "SELECT id, event_type, payload FROM events "
        "WHERE project_id = ? AND archived = 0 AND superseded_by IS NULL",
        (project_id,),
    ).fetchall():
        payload = json.loads(r["payload"])
        toks = normalize_for_overlap(payload.get("description", ""))
        if not toks:
            continue
        score = len(q & toks) / len(q | toks)
        if score > 0:
            scored.append((score, r["id"], r["event_type"], payload))
    scored.sort(key=lambda t: -t[0])
    return [
        {"id": i, "event_type": et, "score": round(s, 4),
         "description": p.get("description", ""), "subject": p.get("subject", "")}
        for s, i, et, p in scored[:limit]
    ]


def _recall_hits(conn: sqlite3.Connection, project_id: str, query: str, k: int) -> list[dict]:
    """Semantic recall when vectors are stored; deterministic lexical otherwise.

    Tries semantic first (embedding model + stored vectors). Falls back to the
    deterministic lexical path when:
      - the model is not installed, OR
      - no events have stored vectors yet (e.g. before the first session_end with
        the model available, or before a backfill has run at SessionStart).
    This makes the hook correct even on a fresh install and progressively stronger
    as the vector store fills in.
    """
    from memlora.embedding.model import is_available

    if is_available():
        from memlora.embedding.retrieval import recall
        hits = recall(conn, project_id, query, k=k)
        if hits:
            return hits
        # No stored vectors yet — fall through to lexical so the hook isn't silent.
    return _lexical_recall(conn, project_id, query, k)


def _fmt_hit(h: dict, extra: str = "") -> str:
    subj = f"[{h['subject']}] " if h.get("subject") else ""
    desc = (h.get("description") or "")[:_MAX_DESC]
    return f"- ({h['event_type']} · {h['score']:.2f}{extra}) {subj}{desc}"


def recall_for_prompt(
    project_path: str,
    prompt_text: str,
    *,
    config: Config | None = None,
) -> str:
    """Per-prompt injection candidate for the UserPromptSubmit hook (CK-1).

    Returns a compact snippet (≤ config.query_injection_max_tokens chars) when a
    high-confidence, non-redundant prior decision is relevant to `prompt_text`.
    Returns '' (silence) when nothing clears the threshold — silence is the
    default, not the fallback. The caller must check the flag and handle timeouts.
    """
    try:
        config = config or Config.load(project_path=project_path)
        project_id, db_path = _resolve(project_path, config)
        if not db_path.exists():
            return ""
        threshold = config.query_injection_threshold
        max_chars = config.query_injection_max_tokens * 4  # tok → approx chars
        with get_connection(db_path) as conn:
            run_migrations(conn)
            hits = _recall_hits(conn, project_id, prompt_text, 5)
        # Filter: skip event types that are always in the static block.
        hits = [h for h in hits if h.get("event_type") not in _ALWAYS_INJECTED]
        # Apply relevance gate — inject nothing below threshold.
        hits = [h for h in hits if h.get("score", 0) >= threshold]
        if not hits:
            return ""
        lines = ["[CogniKernel — relevant prior context]"]
        used = len(lines[0])
        for h in hits:
            line = _fmt_hit(h)
            if used + len(line) + 1 > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception:  # never raise; silence on any error
        return ""


def recall_memory(project_path: str, query: str, limit: int = 8, config: Config | None = None) -> str:
    """Return prior decisions/constraints most relevant to `query`, ranked."""
    try:
        project_id, db_path = _resolve(project_path, config)
        if not db_path.exists():
            return "No CogniKernel memory exists for this project yet."
        with get_connection(db_path) as conn:
            run_migrations(conn)
            hits = _recall_hits(conn, project_id, query, limit)
        if not hits:
            return f"No stored decisions relevant to: {query!r}"
        return "\n".join(
            [f"CogniKernel — decisions relevant to {query!r}:"] + [_fmt_hit(h) for h in hits]
        )
    except Exception as exc:  # never raise into the MCP layer
        return f"recall failed: {exc}"


def find_related_memory(project_path: str, query: str, limit: int = 8, config: Config | None = None) -> str:
    """Find decisions + code areas related to `query` via semantics ∪ the import graph.

    Query-seeded: recall the single most relevant event, then expand from it with
    `find_related` (semantic neighbours ∪ symbol-graph-adjacent events).
    """
    try:
        project_id, db_path = _resolve(project_path, config)
        if not db_path.exists():
            return "No CogniKernel memory exists for this project yet."
        with get_connection(db_path) as conn:
            run_migrations(conn)
            seeds = _recall_hits(conn, project_id, query, 1)
            if not seeds:
                return f"No memory to relate to: {query!r}"
            seed = seeds[0]
            from memlora.embedding.retrieval import find_related
            related = find_related(conn, project_id, seed["id"], k=limit)
            meta: dict[int, tuple[str, str, str]] = {}
            ids = [r["id"] for r in related]
            if ids:
                placeholders = ",".join("?" * len(ids))
                for row in conn.execute(
                    f"SELECT id, event_type, payload FROM events WHERE id IN ({placeholders})", ids
                ).fetchall():
                    p = json.loads(row["payload"])
                    meta[row["id"]] = (row["event_type"], p.get("description", ""), p.get("subject", ""))
        if not related:
            return f"No related decisions found for: {query!r} (seed: {seed.get('description','')[:_MAX_DESC]})"
        lines = [f"CogniKernel — related to {query!r} (seed: {seed.get('description','')[:_MAX_DESC]}):"]
        for r in related:
            et, desc, subj = meta.get(r["id"], ("?", "", ""))
            subj = f"[{subj}] " if subj else ""
            lines.append(f"- ({et} · {r['why']} · {r['score']:.2f}) {subj}{desc[:_MAX_DESC]}")
        return "\n".join(lines)
    except Exception as exc:  # never raise into the MCP layer
        return f"find_related failed: {exc}"
