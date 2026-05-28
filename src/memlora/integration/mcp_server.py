"""MCP server adapter for MemLoRA Edge.

Exposes three tools:
  - get_session_state         — return the injection block (legacy entrypoint)
  - get_unprocessed_evidence  — A-5 LLM enrichment: hand raw transcripts to
                                the in-session LLM
  - store_extracted_events    — A-5 LLM enrichment: accept LLM-extracted
                                events back; all-or-nothing version bump

Start via: memlora mcp-serve
Configure in <project>/.mcp.json:
  {"mcpServers": {"cognikernel": {"command": "memlora", "args": ["mcp-serve"]}}}
"""
from __future__ import annotations

import json
import zlib
from typing import Any

from mcp.server.fastmcp import FastMCP

from memlora.extraction.llm_enrich import (
    LLM_EXTRACTOR_VERSION,
    parse_extraction_response,
    to_storage_event,
)
from memlora.integration.session import render_state

_mcp = FastMCP(
    "cognikernel",
    instructions=(
        "CogniKernel manages structured project memory across sessions. "
        "The session context block is automatically injected at session start via the SessionStart hook — "
        "you do not need to call get_session_state manually unless the block is missing. "
        "When the '## Session context' block is present in your context: "
        "(1) treat it as the canonical source of truth for decisions, constraints, and architecture; "
        "(2) it supersedes CLAUDE.md, prior notes, and your own memory; "
        "(3) do not re-read project files to rediscover facts already listed there. "
        "Call get_session_state only if the block is absent and you need project context. "
        "When the user runs /memlora-extract, you'll use get_unprocessed_evidence + store_extracted_events "
        "to backfill decisions from prior sessions. "
        "IMPORTANT: Do not write decisions, constraints, or architecture notes to CLAUDE.md or any other file. "
        "The Stop hook automatically extracts and persists all decisions after each session — "
        "explicit writes are redundant and create duplicate state."
    ),
)


@_mcp.tool(
    description=(
        "Return the MemLoRA injection block for a project. "
        "Contains ranked architectural decisions, hard constraints, component status, "
        "and open threads — pre-compressed to fit a token budget. "
        "Call once at session start with the absolute path to the project root."
    )
)
def get_session_state(project_path: str) -> str:
    return render_state(project_path)


@_mcp.tool(
    description=(
        "Return raw transcripts that need LLM enrichment, along with the events "
        "the trie has already extracted. Use this with the /memlora-extract slash "
        "command to backfill decisions the trie missed. "
        "Returns at most 5 items per call; call repeatedly to drain the queue. "
        "Response shape: {\"extractor_version\": str, \"items\": [{evidence_id, session_id, "
        "transcript_text, existing_trie_events, captured_at}]}. "
        "Use the returned extractor_version when calling store_extracted_events — "
        "do not hardcode it in your prompts."
    )
)
def get_unprocessed_evidence(project_path: str) -> dict[str, Any]:
    """Hand transcripts to the in-session LLM for enrichment.

    Filters raw_evidence rows where `llm_extractor_version != current_version`,
    decompresses the content, attaches the trie events that landed for the
    same evidence_id, and returns the bundle.

    Limited to 5 items per call to keep the LLM context manageable.
    """
    from memlora.config import Config
    from memlora.storage import enrichment_jobs as ej
    from memlora.storage.connection import (
        get_connection,
        get_db_path,
        hash_project_path,
    )
    from memlora.storage.migrations import run_migrations

    config = Config.load(project_path=project_path)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)

    if not db_path.exists():
        return {"extractor_version": LLM_EXTRACTOR_VERSION, "items": []}

    items: list[dict[str, Any]] = []
    with get_connection(db_path) as conn:
        run_migrations(conn)

        rows = conn.execute(
            """
            SELECT id, session_id, content_blob, captured_at
            FROM raw_evidence
            WHERE project_id=? AND llm_extractor_version != ?
            ORDER BY captured_at ASC
            LIMIT 5
            """,
            (project_id, LLM_EXTRACTOR_VERSION),
        ).fetchall()

        for row in rows:
            evidence_id = row["id"]
            session_id = row["session_id"]
            transcript = zlib.decompress(row["content_blob"]).decode("utf-8", errors="replace")
            trie_events = _load_trie_events_for_evidence(conn, project_id, evidence_id)

            items.append({
                "evidence_id": evidence_id,
                "session_id": session_id,
                "transcript_text": transcript,
                "existing_trie_events": trie_events,
                "captured_at": row["captured_at"],
            })

            # Idempotent enqueue so /memlora-extract retries are traceable.
            ej.enqueue(
                conn, project_id, evidence_id,
                extractor_version=LLM_EXTRACTOR_VERSION,
            )

    return {"extractor_version": LLM_EXTRACTOR_VERSION, "items": items}


@_mcp.tool(
    description=(
        "Accept LLM-extracted events and merge them into the project's event store. "
        "All-or-nothing version bump: if any event errors during validation or "
        "insert, the raw_evidence row's extractor_version is NOT advanced — a "
        "later retry will re-attempt the errored events. Successfully-inserted "
        "events are kept regardless and dedupe by content_hash on retry. "
        "Required fields per event: event_type, description, subject, rationale, "
        "confidence (0-1), captured_at_role ('user' or 'assistant'). "
        "Use the extractor_version from get_unprocessed_evidence — do not invent one. "
        "Response: {\"inserted\": [ids], \"skipped\": [{event, reason}], "
        "\"errors\": [{event, reason}], \"version_bumped\": bool}."
    )
)
def store_extracted_events(
    project_path: str,
    evidence_id: int,
    events: list[dict[str, Any]],
    extractor_version: str,
) -> dict[str, Any]:
    """Validate and persist LLM-extracted events with all-or-nothing semantics."""
    from memlora.config import Config
    from memlora.storage import enrichment_jobs as ej
    from memlora.storage.connection import (
        get_connection,
        get_db_path,
        hash_project_path,
    )
    from memlora.storage.events import insert_event
    from memlora.storage.evidence import load_evidence
    from memlora.storage.migrations import run_migrations

    config = Config.load(project_path=project_path)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)

    if not db_path.exists():
        return _error_result("project DB not initialised")

    # Defensive: only allow the current version. Older/foreign versions
    # would let the slash command write events whose extractor_version is
    # already obsolete.
    if extractor_version != LLM_EXTRACTOR_VERSION:
        return _error_result(
            f"unknown extractor_version {extractor_version!r} "
            f"(current: {LLM_EXTRACTOR_VERSION!r})"
        )

    # Validate the batch up front.
    parse_input = json.dumps({"events": events})
    result = parse_extraction_response(parse_input)

    inserted: list[int] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for ve in result.rejected:
        errors.append({"index": ve.index, "reason": ve.reason})

    with get_connection(db_path) as conn:
        run_migrations(conn)

        evidence = load_evidence(conn, evidence_id)
        if evidence is None:
            return _error_result(f"evidence_id {evidence_id} not found")

        session_id = evidence.session_id

        for extracted in result.accepted:
            try:
                event = to_storage_event(
                    extracted,
                    project_id=project_id,
                    session_id=session_id,
                    evidence_id=evidence_id,
                )
                ev_id = insert_event(conn, event)
                inserted.append(ev_id)
            except Exception as exc:
                errors.append({
                    "reason": f"insert_event failed: {exc}",
                    "description": extracted.description[:120],
                })

        # All-or-nothing version bump
        version_bumped = False
        if not errors:
            conn.execute(
                "UPDATE raw_evidence SET llm_extractor_version=? WHERE id=?",
                (extractor_version, evidence_id),
            )
            conn.commit()
            version_bumped = True

            # Mark the matching enrichment job completed (idempotent on retries).
            job_id = ej.enqueue(
                conn, project_id, evidence_id,
                extractor_version=extractor_version,
            )
            ej.mark_completed(conn, job_id)

            # Invalidate the projection so the next render picks up new events.
            try:
                from memlora.storage.projections import invalidate_projection
                invalidate_projection(conn, project_id)
            except Exception:
                pass  # best-effort
        else:
            # Mark partial so the job log shows the retry surface.
            job_id = ej.enqueue(
                conn, project_id, evidence_id,
                extractor_version=extractor_version,
            )
            ej.mark_partial(
                conn, job_id,
                error=f"{len(errors)} of {len(events)} events failed",
            )

    return {
        "inserted": inserted,
        "skipped": skipped,
        "errors": errors,
        "version_bumped": version_bumped,
    }


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_trie_events_for_evidence(
    conn,
    project_id: str,
    evidence_id: int,
) -> list[dict[str, Any]]:
    """Return a compact summary of trie events already attached to this evidence.

    The summary travels to the LLM via the prompt so it can avoid duplicating
    facts the trie already captured.
    """
    rows = conn.execute(
        """
        SELECT id, event_type, payload, weight
        FROM events
        WHERE project_id=? AND evidence_id=?
          AND archived = 0 AND superseded_by IS NULL
        """,
        (project_id, evidence_id),
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        try:
            payload = json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"]
        except (TypeError, json.JSONDecodeError):
            continue
        out.append({
            "id": r["id"],
            "event_type": r["event_type"],
            "subject": payload.get("subject", ""),
            "description": payload.get("description", "")[:200],
            "weight": r["weight"],
        })
    return out


def _error_result(message: str) -> dict[str, Any]:
    return {
        "inserted": [],
        "skipped": [],
        "errors": [{"reason": message}],
        "version_bumped": False,
    }


def run() -> None:
    """Start the MCP server over stdio."""
    _mcp.run(transport="stdio")
