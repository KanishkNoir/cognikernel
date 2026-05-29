"""Integration layer — high-level orchestration used by the CLI and external callers."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from memlora.config import Config
from memlora.storage.connection import get_connection, get_db_path, hash_project_path
from memlora.storage.migrations import run_migrations
from memlora.storage.projections import Projection, load_or_rebuild

_log = logging.getLogger("memlora.integration")


def init_project(
    project_path: str | Path,
    config: Config | None = None,
) -> str:
    """Create and migrate the DB for a project. Idempotent. Returns project_id."""
    config = config or Config.load(project_path=project_path)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)
    with get_connection(db_path) as conn:
        run_migrations(conn)
    _log.info("init_project.done", extra={"project_id": project_id, "db": str(db_path)})
    return project_id


def session_end(
    project_path: str | Path,
    session_id: str,
    transcript: str,
    config: Config | None = None,
    git_diff: str | None = None,
    evidence_content: str | bytes | None = None,
    evidence_source_type: str = "transcript",
    evidence_source_path: str = "",
) -> dict[str, Any]:
    """Extract events from *transcript* and merge them into the project DB.

    Returns a stats dict with keys: extracted, inserted, updated,
    superseded, cascaded, archived.
    """
    from memlora.extraction.pipeline import SessionMetadata, extract_session
    from memlora.delta.merge import execute_merge
    from memlora.storage.evidence import store_evidence
    from memlora.storage.jobs import ack_stage, enqueue_extraction, fail_job

    config = config or Config.load(project_path=project_path)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)

    with get_connection(db_path) as conn:
        run_migrations(conn)
        evidence_id = store_evidence(
            conn,
            project_id=project_id,
            session_id=session_id,
            source_type=evidence_source_type,
            content=evidence_content if evidence_content is not None else transcript,
            source_path=evidence_source_path,
            metadata={"git_diff": bool(git_diff)},
        )
        job_id = enqueue_extraction(
            conn,
            project_id=project_id,
            session_id=session_id,
            evidence_id=evidence_id,
            job_category="extract.transcript",
        )
        ack_stage(conn, job_id, "OBSERVED", output_ref=f"raw_evidence:{evidence_id}")

    now = int(time.time() * 1000)
    session_meta = SessionMetadata(
        project_id=project_id,
        session_id=session_id,
        started_at=now,
        ended_at=now,
    )
    try:
        candidates = extract_session(transcript, session_meta, git_diff=git_diff)
        for event in candidates:
            event.evidence_id = evidence_id

        with get_connection(db_path) as conn:
            ack_stage(conn, job_id, "PARSED", output_ref=f"events:{len(candidates)}")
            ack_stage(conn, job_id, "CLASSIFIED", output_ref=f"events:{len(candidates)}")
            stats = execute_merge(
                conn, session_id, candidates, embed_events=config.embedding_enabled
            )
            ack_stage(
                conn,
                job_id,
                "MERGED",
                output_ref=json_like_stats(stats),
            )
            _update_symbol_graph(conn, project_id, str(project_path), git_diff, session_id=session_id)
            ack_stage(conn, job_id, "PROJECTED", output_ref="projection:invalidated")
            ack_stage(conn, job_id, "COMPLETED", output_ref="session_end")
    except Exception as exc:
        with get_connection(db_path) as conn:
            try:
                fail_job(conn, job_id, "EXTRACTOR_BUG", str(exc))
            except Exception:
                pass
        raise

    stats["extracted"] = len(candidates)
    stats["evidence_id"] = evidence_id
    stats["job_id"] = job_id
    _log.info("session_end.done", extra={"session_id": session_id, **stats})
    return stats


def json_like_stats(stats: dict[str, Any]) -> str:
    import json

    return json.dumps(stats, sort_keys=True, separators=(",", ":"))


def replay_job(
    project_path: str | Path,
    job_id: int,
    config: Config | None = None,
) -> dict[str, Any]:
    """Re-run a dead-lettered extraction job using its original raw_evidence.

    Flips the job state via replay_dead_letter, decompresses the original
    content from raw_evidence, then re-invokes session_end with the same
    session_id. The existing job row is reused (INSERT OR IGNORE on
    enqueue_extraction) so it advances queued -> completed in place, or
    back to dead_lettered if the underlying problem persists.
    """
    from memlora.storage.evidence import load_evidence
    from memlora.storage.jobs import get_job, replay_dead_letter

    config = config or Config.load(project_path=project_path)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)

    with get_connection(db_path) as conn:
        run_migrations(conn)
        job = get_job(conn, job_id)
        if job.state != "dead_lettered":
            raise ValueError(
                f"job {job_id} is not dead-lettered (state={job.state!r}); "
                f"only dead-lettered jobs can be replayed"
            )
        evidence = load_evidence(conn, job.evidence_id)
        if evidence is None:
            raise ValueError(
                f"evidence_id={job.evidence_id} referenced by job {job_id} "
                f"is missing from raw_evidence — cannot replay"
            )
        replay_dead_letter(conn, job_id)

    raw = evidence.content.decode("utf-8")
    transcript = raw
    if evidence.source_type == "jsonl_transcript":
        from memlora.extraction.jsonl_converter import jsonl_to_transcript
        transcript = jsonl_to_transcript(raw)

    return session_end(
        project_path=project_path,
        session_id=job.session_id,
        transcript=transcript,
        config=config,
        evidence_content=raw,
        evidence_source_type=evidence.source_type,
        evidence_source_path=evidence.source_path,
    )


def get_projection(
    project_path: str | Path,
    config: Config | None = None,
) -> Projection:
    """Return the current (possibly rebuilt) Projection for *project_path*."""
    config = config or Config.load(project_path=project_path)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)
    with get_connection(db_path) as conn:
        run_migrations(conn)
        return load_or_rebuild(conn, project_id)


def render_state(
    project_path: str | Path,
    config: Config | None = None,
) -> str:
    """Return the rendered injection block — what would be prepended to the LLM system prompt."""
    from memlora.storage import symbol_files as sf
    from memlora.storage.projections import load_or_rebuild, projection_to_events
    from memlora.compression.greedy import greedy_fill
    from memlora.injection.ordering import make_injection_context
    from memlora.injection.template import render_with_budget_enforcement
    from memlora.symbols.store import load_symbol_nodes, load_symbol_edges
    from memlora.symbols.projection import compress_to_skeleton

    config = config or Config.load(project_path=project_path)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)

    with get_connection(db_path) as conn:
        run_migrations(conn)
        # Source events from the projection (single partition + component-collapse
        # site, behind the high-water cache) rather than re-querying + re-routing
        # all events on every render.
        events = projection_to_events(load_or_rebuild(conn, project_id))
        session_count: int = conn.execute(
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
        nodes = load_symbol_nodes(conn, project_id)
        edges = load_symbol_edges(conn, project_id)
        # Phase B-2: read symbol_files for truthful skeleton header.
        coverage = sf.coverage_stats(conn, project_id)
        refresh = sf.most_recent_refresh(conn, project_id)

    project_name = Path(project_path).resolve().name
    hot_files = _compute_hot_files(events)
    selected = greedy_fill(events, config.token_budget)
    hot_paths = frozenset(hf[0] for hf in hot_files)
    skeleton = compress_to_skeleton(
        nodes, edges,
        budget_tokens=config.skeleton_budget,
        hot_paths=hot_paths,
    )

    ctx = make_injection_context(
        events=selected,
        project_name=project_name,
        session_number=session_count,
        total_sessions=session_count,
        state_version=1,
        token_budget=config.token_budget,
    )
    ctx.hot_files = hot_files
    ctx.skeleton = skeleton
    ctx.ckl_mode = config.ckl_mode
    ctx.ckl_v2 = config.ckl_v2
    ctx.section_budgets = config.section_budgets
    # Phase B trust signals — only carried through to the renderer.
    ctx.hook_policy = config.hook_policy
    ctx.retry_window_seconds = config.deny_retry_window_seconds
    ctx.skeleton_coverage = coverage
    ctx.skeleton_refresh = refresh
    return render_with_budget_enforcement(ctx)


def rebuild_from_raw(
    project_path: str | Path,
    since_evidence_id: int = 0,
    dry_run: bool = False,
    config: Config | None = None,
) -> dict[str, Any]:
    """Regenerate derived tables in a sidecar DB from raw_evidence.

    Writes to <project_id>.db.rebuild — the source DB is never touched.
    The sidecar is always recreated from scratch to guarantee determinism.

    The audit invariant: given the same raw_evidence + extractor version,
    the (event_type, content_hash, payload) set in the sidecar is identical
    to what the original run produced. Lifecycle metadata (weight,
    mention_count, archived, superseded_by) is recomputed from the replay
    order and may differ from live values — see design decision §6.5(c).
    """
    import zlib

    from memlora.extraction.pipeline import SessionMetadata, extract_session
    from memlora.delta.merge import execute_merge

    config = config or Config.load(project_path=project_path)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)

    with get_connection(db_path) as source_conn:
        run_migrations(source_conn)
        evidence_rows = source_conn.execute(
            """
            SELECT id, session_id, source_type, source_path, captured_at,
                   content_sha256, content_encoding, content_blob,
                   original_size_bytes, stored_size_bytes, metadata
            FROM raw_evidence
            WHERE project_id = ? AND id > ?
            ORDER BY id ASC
            """,
            (project_id, since_evidence_id),
        ).fetchall()

    evidence_count = len(evidence_rows)
    sidecar_path = db_path.parent / (db_path.name + ".rebuild")

    if dry_run:
        return {
            "dry_run": True,
            "evidence_count": evidence_count,
            "since_evidence_id": since_evidence_id,
            "sidecar_path": str(sidecar_path),
        }

    # Always start from a clean sidecar for determinism.
    if sidecar_path.exists():
        sidecar_path.unlink()
    for _ext in ("-wal", "-shm"):
        stale = Path(str(sidecar_path) + _ext)
        if stale.exists():
            stale.unlink()

    total_extracted = 0
    total_inserted = 0
    total_updated = 0
    sessions_seen: set[str] = set()
    errors = 0

    # One sidecar connection for the whole run avoids per-row WAL churn.
    with get_connection(sidecar_path) as sidecar_conn:
        run_migrations(sidecar_conn)
        sidecar_conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('rebuild_source', ?)",
            (str(db_path),),
        )
        sidecar_conn.commit()

        for row in evidence_rows:
            evidence_id = row["id"]
            session_id = row["session_id"]
            source_type = row["source_type"]

            # Copy evidence row into sidecar (preserving original id for FK integrity).
            sidecar_conn.execute(
                """
                INSERT OR IGNORE INTO raw_evidence
                    (id, project_id, session_id, source_type, source_path, captured_at,
                     content_sha256, content_encoding, content_blob,
                     original_size_bytes, stored_size_bytes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence_id,
                    project_id,
                    session_id,
                    source_type,
                    row["source_path"],
                    row["captured_at"],
                    row["content_sha256"],
                    row["content_encoding"],
                    row["content_blob"],
                    row["original_size_bytes"],
                    row["stored_size_bytes"],
                    row["metadata"],
                ),
            )
            sidecar_conn.commit()

            raw = zlib.decompress(row["content_blob"]).decode("utf-8")
            transcript = raw
            if source_type == "jsonl_transcript":
                from memlora.extraction.jsonl_converter import jsonl_to_transcript
                transcript = jsonl_to_transcript(raw)

            now = int(time.time() * 1000)
            session_meta = SessionMetadata(
                project_id=project_id,
                session_id=session_id,
                started_at=now,
                ended_at=now,
            )

            try:
                candidates = extract_session(transcript, session_meta)
                for event in candidates:
                    event.evidence_id = evidence_id
                stats = execute_merge(
                    sidecar_conn, session_id, candidates,
                    embed_events=config.embedding_enabled,
                )
                _update_symbol_graph(sidecar_conn, project_id, str(project_path), git_diff=None, session_id=session_id)
                total_extracted += len(candidates)
                total_inserted += stats.get("inserted", 0)
                total_updated += stats.get("updated", 0)
                sessions_seen.add(session_id)
            except Exception as exc:
                _log.warning(
                    "rebuild.evidence_failed",
                    extra={"evidence_id": evidence_id, "error": str(exc)},
                )
                errors += 1

    _log.info(
        "rebuild.done",
        extra={
            "project_id": project_id,
            "evidence_count": evidence_count,
            "sessions_processed": len(sessions_seen),
            "errors": errors,
        },
    )
    return {
        "sidecar_path": str(sidecar_path),
        "evidence_count": evidence_count,
        "sessions_processed": len(sessions_seen),
        "total_extracted": total_extracted,
        "total_inserted": total_inserted,
        "total_updated": total_updated,
        "errors": errors,
        "since_evidence_id": since_evidence_id,
    }


def _update_symbol_graph(
    conn,
    project_id: str,
    project_path: str,
    git_diff: str | None,
    session_id: str = "",
) -> None:
    """Parse changed files and upsert symbol graph + symbol_files (C1).

    Errors are logged, never raised. Passes `project_path` so apply_symbol_update
    populates `symbol_files` rows — necessary for the first-session walk where
    no PostToolUse hook fired but the next session's strict-mode gate needs
    file-level authority for every scanned file.
    """
    try:
        from memlora.symbols.extractor import build_symbol_update
        from memlora.symbols.store import apply_symbol_update
        from memlora.extraction.git_augment import parse_diff

        changed_files = parse_diff(git_diff) if git_diff else []
        update = build_symbol_update(project_id, project_path, changed_files)
        apply_symbol_update(
            conn,
            update,
            project_path=project_path,
            session_id=session_id,
            last_action="scan",
        )
    except Exception as exc:
        _log.warning("symbol_graph.update_failed", extra={"error": str(exc)})


def _compute_hot_files(
    events: list,
    min_mentions: int = 2,
) -> list[tuple[str, int, str]]:
    """Aggregate COMPONENT_STATUS events by path, return paths with total_mentions >= min_mentions.

    Defensively drops bare-basename paths (e.g. ``env.py``) — these are
    extractor noise that conflicts with the prefixed canonical form
    (``alembic/env.py``). New extractions filter at insertion time
    (see ``extraction/file_mentions.py``); this filter protects projects
    whose DB already contains pre-fix bare-basename rows.
    """
    from collections import defaultdict
    from memlora.utils.paths import is_bare_basename
    files: dict = defaultdict(lambda: {"mentions": 0, "intent": ""})
    for e in events:
        if e.event_type != "COMPONENT_STATUS":
            continue
        path = e.payload.get("path", "")
        if not path or is_bare_basename(path):
            continue
        files[path]["mentions"] += e.mention_count or 1
        if not files[path]["intent"]:
            intent = e.payload.get("intent", "")
            if intent and intent != path:
                files[path]["intent"] = intent
    return sorted(
        [(p, d["mentions"], d["intent"]) for p, d in files.items()
         if d["mentions"] >= min_mentions],
        key=lambda x: -x[1],
    )
