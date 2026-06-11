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
        # Persist the resolved path so resource discovery can reverse-map
        # project_id → path (cognikernel://projects MCP resource, CK-5).
        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('project_path', ?)",
            (str(Path(project_path).resolve()),),
        )
        conn.commit()
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
    from memlora.storage.cursors import (
        get_cursor, save_cursor,
        slice_jsonl_for_extraction, slice_storage_delta,
    )
    from memlora.storage.evidence import store_evidence
    from memlora.storage.jobs import ack_stage, enqueue_extraction, fail_job, recover_stuck_running_jobs

    config = config or Config.load(project_path=project_path)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)

    # I2: delta extraction — process only new JSONL lines since last firing.
    # I3: delta storage — store only new lines, chain-linked to previous chunk.
    # Fail-open: cursor miss or compaction detected -> full extraction + full store.
    raw_jsonl = evidence_content if isinstance(evidence_content, str) else transcript
    with get_connection(db_path) as conn:
        run_migrations(conn)
        recover_stuck_running_jobs(conn)
        cursor = get_cursor(conn, project_id, session_id)

    extraction_slice, new_line_count, new_anchor = slice_jsonl_for_extraction(
        raw_jsonl, cursor
    )
    storage_bytes, is_delta_store, has_new = slice_storage_delta(raw_jsonl, cursor)
    if not has_new:
        # Nothing new since the cursor — storing again would collide on
        # content_sha256 and hand back a terminal job (ack would raise).
        _log.info("session_end.no_new_content", extra={"session_id": session_id})
        return {
            "extracted": 0, "inserted": 0, "updated": 0,
            "superseded": 0, "cascaded": 0, "archived": 0,
            "evidence_id": None, "job_id": None, "delta_mode": False,
        }
    delta_mode = cursor is not None and extraction_slice != raw_jsonl

    # Use the chain's previous evidence id when storing a delta chunk.
    prev_ev_id = cursor.last_evidence_id if (is_delta_store and cursor) else None

    with get_connection(db_path) as conn:
        evidence_id = store_evidence(
            conn,
            project_id=project_id,
            session_id=session_id,
            source_type=evidence_source_type,
            content=storage_bytes if is_delta_store else (
                evidence_content if evidence_content is not None else transcript
            ),
            source_path=evidence_source_path,
            metadata={"git_diff": bool(git_diff), "delta_mode": delta_mode, "chain_delta": is_delta_store},
            prev_evidence_id=prev_ev_id,
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
        # Extract from the delta slice (or full transcript on first/fallback run).
        candidates = extract_session(
            extraction_slice, session_meta, git_diff=git_diff, extractor=config.extractor
        )
        for event in candidates:
            event.evidence_id = evidence_id

        with get_connection(db_path) as conn:
            ack_stage(conn, job_id, "PARSED", output_ref=f"events:{len(candidates)}")
            ack_stage(conn, job_id, "CLASSIFIED", output_ref=f"events:{len(candidates)}")
            stats = execute_merge(
                conn, session_id, candidates,
                embed_events=config.embedding_enabled,
                use_cross_encoder=config.cross_encoder_supersession,
            )
            ack_stage(
                conn,
                job_id,
                "MERGED",
                output_ref=json_like_stats(stats),
            )
            # Advance cursor only after a successful merge — never on exception.
            save_cursor(conn, project_id, session_id, new_line_count, new_anchor,
                        last_evidence_id=evidence_id)
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
    stats["delta_mode"] = delta_mode
    _log.info("session_end.done", extra={"session_id": session_id, **stats})
    return stats


def json_like_stats(stats: dict[str, Any]) -> str:
    import json

    return json.dumps(stats, sort_keys=True, separators=(",", ":"))


def session_capture(
    project_path: str | Path,
    session_id: str,
    raw_jsonl: str,
    config: Config | None = None,
    git_diff: str | None = None,
    evidence_source_type: str = "jsonl_transcript",
    evidence_source_path: str = "",
) -> dict[str, Any]:
    """Store evidence + enqueue extraction job. Returns immediately without extracting.

    This is the capture half of the I4 decoupled pipeline. It performs the fast,
    I/O-only work: delta evidence storage (I3 chain) + job enqueue. The slow work
    (extract_session + execute_merge + symbol graph) is handled by process_jobs(),
    which runs as a detached worker subprocess so the Stop hook returns in <500ms.

    When the transcript has no new lines since the cursor, returns
    {"job_id": None, ...} and stores nothing — re-storing identical content
    would collide on content_sha256 and return a possibly-terminal job.

    A git diff, when provided, is persisted as its own raw_evidence row
    (source_type='git_diff') and linked via the transcript evidence's metadata
    so the worker can pass it to extraction + the symbol graph. (Gamma
    post-mortem F5: the diff content used to be silently dropped.)

    Returns {"job_id": int|None, "evidence_id": int|None, "delta_mode": bool}.
    """
    from memlora.storage.cursors import get_cursor, slice_storage_delta
    from memlora.storage.evidence import store_evidence
    from memlora.storage.jobs import enqueue_extraction, recover_stuck_running_jobs

    config = config or Config.load(project_path=project_path)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)

    with get_connection(db_path) as conn:
        run_migrations(conn)
        # 30-min grace: must exceed any plausible live merge so capture never
        # dead-letters a job the worker is actively processing between acks.
        recover_stuck_running_jobs(conn, stale_after_ms=30 * 60 * 1000)
        cursor = get_cursor(conn, project_id, session_id)

    storage_bytes, is_delta_store, has_new = slice_storage_delta(raw_jsonl, cursor)
    if not has_new:
        _log.info("session_capture.no_new_content", extra={"session_id": session_id})
        return {"job_id": None, "evidence_id": None, "delta_mode": False}

    prev_ev_id = cursor.last_evidence_id if (is_delta_store and cursor) else None

    with get_connection(db_path) as conn:
        metadata: dict[str, Any] = {"chain_delta": is_delta_store}
        if git_diff:
            git_ev_id = store_evidence(
                conn,
                project_id=project_id,
                session_id=session_id,
                source_type="git_diff",
                content=git_diff,
                metadata={"for_session": session_id},
            )
            metadata["git_diff_evidence_id"] = git_ev_id

        evidence_id = store_evidence(
            conn,
            project_id=project_id,
            session_id=session_id,
            source_type=evidence_source_type,
            content=storage_bytes if is_delta_store else raw_jsonl,
            source_path=evidence_source_path,
            metadata=metadata,
            prev_evidence_id=prev_ev_id,
        )
        job_id = enqueue_extraction(
            conn,
            project_id=project_id,
            session_id=session_id,
            evidence_id=evidence_id,
            job_category="extract.transcript",
        )
    _log.info("session_capture.done", extra={
        "session_id": session_id, "job_id": job_id,
        "evidence_id": evidence_id, "is_delta": is_delta_store,
    })
    return {"job_id": job_id, "evidence_id": evidence_id, "delta_mode": is_delta_store}


def _worker_log(project_id: str, msg: str) -> None:
    """Append a line to the worker log. Workers run detached with stdout/stderr
    on DEVNULL — without this, every failure is invisible (gamma post-mortem F2)."""
    try:
        log_dir = Path.home() / ".memlora" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(log_dir / f"worker-{project_id}.log", "a", encoding="utf-8") as f:
            f.write(f"{stamp} {msg}\n")
    except Exception:
        pass


def _acquire_worker_lock(project_id: str, stale_after_s: int = 15 * 60) -> Path | None:
    """Single-flight guard: one worker per project (gamma post-mortem F2).

    Every Stop firing spawns a worker; without this, N workers pile onto the
    same SQLite file and starve each other. Returns the lock path on success,
    None if another live worker holds it. A lock older than `stale_after_s`
    is treated as crashed and taken over.
    """
    import os
    lock_dir = Path.home() / ".memlora" / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"worker-{project_id}.lock"
    try:
        with open(lock_path, "x", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return lock_path
    except FileExistsError:
        try:
            age = time.time() - lock_path.stat().st_mtime
            if age > stale_after_s:
                lock_path.write_text(str(os.getpid()), encoding="utf-8")  # takeover
                return lock_path
        except Exception:
            pass
        return None


def process_jobs(
    project_path: str | Path,
    config: Config | None = None,
    max_jobs: int = 50,
    time_budget_s: float | None = None,
) -> dict[str, Any]:
    """Claim and process all queued extraction jobs for a project.

    This is the processing half of the I4 decoupled pipeline. Runs as a detached
    subprocess spawned by `memlora capture`. Processes jobs in oldest-first order,
    advances the ingest cursor after each successful merge, then exits.

    Hardened after the gamma post-mortem:
    - single-flight lock (one worker per project; extra spawns exit immediately)
    - startup retried on a locked DB instead of crashing silently
    - claim loop survives a race-lost claim (re-checks the queue, doesn't exit)
    - only TIMEOUT dead-letters are auto-replayed (poison jobs stay dead)
    - git diffs persisted by capture are loaded and passed through
    - the cursor never moves backwards

    Returns summary stats: {"processed": int, "failed": int, "replayed": int,
    "skipped": bool} — skipped=True means another worker held the lock.
    """
    import json as _json
    import os
    from memlora.delta.merge import execute_merge
    from memlora.extraction.pipeline import SessionMetadata, extract_session
    from memlora.extraction.jsonl_converter import jsonl_to_transcript
    from memlora.storage.cursors import get_cursor, save_cursor, slice_jsonl_for_extraction
    from memlora.storage.evidence import load_evidence, load_full_transcript
    from memlora.storage.jobs import (
        ack_stage, claim_next_job, fail_job,
        list_jobs, reclaim_stale_jobs, recover_stuck_running_jobs, replay_dead_letter,
    )

    config = config or Config.load(project_path=project_path)
    project_id = hash_project_path(project_path)
    db_path = get_db_path(config, project_id)
    claimant = f"worker-{os.getpid()}"
    processed = failed = replayed = 0

    lock_path = _acquire_worker_lock(project_id)
    if lock_path is None:
        _worker_log(project_id, f"{claimant} exit: another worker holds the lock")
        return {"processed": 0, "failed": 0, "replayed": 0, "skipped": True}

    try:
        # Startup: a long-running merge elsewhere can hold the write lock briefly.
        # Retry instead of crashing — a crashed worker leaves jobs queued forever.
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with get_connection(db_path) as conn:
                    run_migrations(conn)
                    reclaim_stale_jobs(conn, stale_after_ms=10 * 60 * 1000)
                    recover_stuck_running_jobs(conn, stale_after_ms=10 * 60 * 1000)
                    # Auto-replay ONLY process-killed jobs (failure_class TIMEOUT).
                    # POISON_INPUT / EXTRACTOR_BUG dead-letters need a human or a
                    # code fix — resurrecting them forever masks real bugs.
                    dead_jobs = list_jobs(conn, project_id, state="dead_lettered", limit=max_jobs)
                    for dj in dead_jobs:
                        if dj.failure_class != "TIMEOUT":
                            continue
                        try:
                            replay_dead_letter(conn, dj.id)
                            replayed += 1
                        except Exception:
                            pass
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                _worker_log(project_id, f"{claimant} startup attempt {attempt + 1} failed: {exc}")
                time.sleep(2.0 * (attempt + 1))
        if last_exc is not None:
            _worker_log(project_id, f"{claimant} giving up after 3 startup attempts")
            return {"processed": 0, "failed": 0, "replayed": replayed, "skipped": False}

        _worker_log(project_id, f"{claimant} started (replayed {replayed} TIMEOUT dead-letters)")

        started_at = time.monotonic()
        budget = max_jobs
        while budget > 0:
            budget -= 1
            if time_budget_s is not None and (time.monotonic() - started_at) >= time_budget_s:
                _worker_log(project_id, f"{claimant} time budget {time_budget_s}s exhausted — exiting "
                                        f"(remaining jobs stay queued for the next drain)")
                break
            try:
                with get_connection(db_path) as conn:
                    job = claim_next_job(conn, "extract.transcript", claimant)
            except Exception as exc:
                _worker_log(project_id, f"{claimant} claim failed: {exc}")
                time.sleep(1.0)
                continue
            if job is None:
                # Empty queue OR race-lost claim. Check which before exiting.
                try:
                    with get_connection(db_path) as conn:
                        remaining = conn.execute(
                            "SELECT COUNT(*) FROM extraction_jobs "
                            "WHERE state IN ('queued','retryable_failure') "
                            "AND job_category='extract.transcript'"
                        ).fetchone()[0]
                    if remaining == 0:
                        break
                    time.sleep(0.3)
                    continue
                except Exception:
                    break

            try:
                with get_connection(db_path) as conn:
                    full_bytes = load_full_transcript(conn, job.evidence_id)
                    cursor = get_cursor(conn, project_id, job.session_id)
                    evidence = load_evidence(conn, job.evidence_id)
                    git_diff: str | None = None
                    if evidence is not None:
                        git_eid = (evidence.metadata or {}).get("git_diff_evidence_id")
                        if git_eid:
                            git_ev = load_evidence(conn, int(git_eid))
                            if git_ev is not None:
                                git_diff = git_ev.content.decode("utf-8", errors="replace")

                raw = full_bytes.decode("utf-8", errors="replace")
                extraction_slice, new_line_count, new_anchor = slice_jsonl_for_extraction(
                    raw, cursor
                )
                if evidence is not None and evidence.source_type == "jsonl_transcript":
                    transcript = jsonl_to_transcript(extraction_slice)
                else:
                    transcript = extraction_slice  # plain-text transcript path

                now = int(time.time() * 1000)
                session_meta = SessionMetadata(
                    project_id=project_id,
                    session_id=job.session_id,
                    started_at=now,
                    ended_at=now,
                )
                candidates = extract_session(
                    transcript, session_meta, git_diff=git_diff, extractor=config.extractor
                )
                for event in candidates:
                    event.evidence_id = job.evidence_id

                with get_connection(db_path) as conn:
                    ack_stage(conn, job.id, "PARSED", output_ref=f"events:{len(candidates)}")
                    ack_stage(conn, job.id, "CLASSIFIED", output_ref=f"events:{len(candidates)}")
                    stats = execute_merge(
                        conn, job.session_id, candidates,
                        embed_events=config.embedding_enabled,
                        use_cross_encoder=config.cross_encoder_supersession,
                    )
                    ack_stage(conn, job.id, "MERGED", output_ref=json_like_stats(stats))
                    # Monotonic guard: a retried old job must never rewind the
                    # cursor below what a newer job already committed (F7).
                    latest = get_cursor(conn, project_id, job.session_id)
                    if latest is None or new_line_count >= latest.last_line_count:
                        save_cursor(conn, project_id, job.session_id, new_line_count,
                                    new_anchor, last_evidence_id=job.evidence_id)
                    _update_symbol_graph(conn, project_id, str(project_path),
                                         git_diff=git_diff, session_id=job.session_id)
                    ack_stage(conn, job.id, "PROJECTED", output_ref="projection:invalidated")
                    ack_stage(conn, job.id, "COMPLETED", output_ref="process_jobs")
                processed += 1
                _worker_log(project_id, f"{claimant} job={job.id} done: {json_like_stats(stats)}")

            except Exception as exc:
                with get_connection(db_path) as conn:
                    try:
                        fail_job(conn, job.id, "EXTRACTOR_BUG", str(exc))
                    except Exception:
                        pass
                failed += 1
                _worker_log(project_id, f"{claimant} job={job.id} FAILED: {exc}")

            # Heartbeat: refresh lock mtime so a long queue isn't stolen mid-run.
            try:
                os.utime(lock_path)
            except Exception:
                pass

    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except Exception:
            pass

    summary = {"processed": processed, "failed": failed, "replayed": replayed, "skipped": False}
    _worker_log(project_id, f"{claimant} exit: {_json.dumps(summary)}")
    _log.info("process_jobs.done", extra=summary)
    return summary


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
    from memlora.storage.evidence import load_evidence, load_full_transcript
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
        # I3: reconstruct the full transcript by following the evidence chain.
        # A delta-stored chunk must be concatenated with its ancestors to
        # produce the complete JSONL that session_end expects.
        full_bytes = load_full_transcript(conn, job.evidence_id)
        replay_dead_letter(conn, job_id)

    raw = full_bytes.decode("utf-8", errors="replace")
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
                candidates = extract_session(
                    transcript, session_meta, extractor=config.extractor
                )
                for event in candidates:
                    event.evidence_id = evidence_id
                stats = execute_merge(
                    sidecar_conn, session_id, candidates,
                    embed_events=config.embedding_enabled,
                    use_cross_encoder=config.cross_encoder_supersession,
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
