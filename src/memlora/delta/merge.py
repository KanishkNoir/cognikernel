"""Delta merge orchestrator — the session-end bookkeeping step.

Six idempotent operations in one transaction:
  1. Hash-based deduplication (insert-or-update)
  2. Constraint supersession (overlap detection → mark superseded_by)
  3. Component dependency cascade (blocked/abandoned → needs_review)
  4. Weight decay (0.92× for non-current-session events)
  5. Archive (events below 0.05 threshold, excluding protected types)
  6. Projection invalidation (force lazy rebuild on next read)

The whole merge runs inside a single `with conn:` transaction block.
Python's sqlite3 context manager issues BEGIN on entry and COMMIT on success
(ROLLBACK on exception), giving us atomicity without manual BEGIN/COMMIT calls.

Internal helpers (_insert_or_update, _apply_decay_inner) do NOT call
conn.commit() — they are designed to run inside the wrapping transaction.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import TYPE_CHECKING

from memlora.delta.cascade import cascade_component_status
from memlora.delta.decay import (
    ARCHIVE_THRESHOLD,
    DECAY_FACTOR,
    _META_KEY_PREFIX,
    _PROTECTED_FROM_ARCHIVE,
)
from memlora.delta.supersede import (
    apply_supersession,
    find_superseded,
    jaccard_similarity,
    supersedes,
)
from memlora.storage.events import (
    MAX_EVENT_WEIGHT,
    WEIGHT_INCREMENT_ON_DEDUP,
    insert_extraction_failure,
)

if TYPE_CHECKING:
    from memlora.storage.events import Event

_log = logging.getLogger("memlora.delta")

# Cross-type dedup: when the same concept appears under different event types,
# keep only the highest-priority one.
_DEDUP_GROUP: frozenset[str] = frozenset({
    "APPROACH_ABANDONED_DO_NOT_RETRY",
    "CONSTRAINT_HARD",
    "APPROACH_ABANDONED",
})
_DEDUP_PRIORITY: dict[str, int] = {
    "APPROACH_ABANDONED_DO_NOT_RETRY": 1,  # highest
    "CONSTRAINT_HARD": 2,
    "APPROACH_ABANDONED": 3,               # lowest
}

# R3 — echo fold. A NEAR-IDENTICAL same-type restatement (a recitation of an existing
# fact, common in recall-heavy sessions) is folded into its canonical: bump
# mention_count instead of minting a near-dup OR — worse — letting the newer recitation
# supersede the original. Threshold is deliberately HIGH (well above supersession's 0.6)
# so genuine refinements (e.g. "120 lines" -> "300 lines", Jaccard ~0.6) are NOT folded;
# only verbatim-ish echoes. Lossless: raw_evidence retains the echo and its provenance is
# linked to the canonical, exactly like exact-hash dedup.
_ECHO_JACCARD: float = 0.85


def merge_event(conn: sqlite3.Connection, event: Event) -> tuple[str, int]:
    """Insert an event or increment its mention_count on hash collision.

    Returns ("inserted", row_id) or ("updated", row_id).
    Commits after the operation — use _insert_or_update() inside transactions.
    """
    outcome, row_id = _insert_or_update(conn, event)
    conn.commit()
    return outcome, row_id


def execute_merge(
    conn: sqlite3.Connection,
    session_id: str,
    candidates: list[Event],
    embed_events: bool = False,
    use_cross_encoder: bool = False,
) -> dict:
    """Run the full six-step merge inside a single transaction.

    Returns a stats dict: {inserted, updated, superseded, cascaded, archived}.
    On failure, the transaction is rolled back and the error written to the
    dead-letter queue (extraction_failures).

    Supersession always runs through `find_superseded`, so the temporal,
    authority, and provenance gates are the baseline regardless of embeddings —
    a lower-trust, newer, or same-transcript event never supersedes. The
    `embed_events` flag (config.embedding_enabled) only toggles the *semantic*
    candidate axis on top of that gated lexical floor: when True, each event also
    gets a local embedding stored and cosine retrieval contributes extra
    candidates; when False, no embedding model is touched and matching is
    gated-lexical only.
    """
    if not candidates:
        return {"inserted": 0, "updated": 0, "superseded": 0, "cascaded": 0, "archived": 0}

    project_id = candidates[0].project_id
    stats = {"inserted": 0, "updated": 0, "superseded": 0, "cascaded": 0, "archived": 0}

    # Idempotency guard (audit P1). A worker can be killed (the Job Object tears
    # down hook-spawned drains at hook exit) AFTER this merge commits but BEFORE
    # the ingest cursor advances; the job is then recovered and the SAME evidence
    # slice is replayed. _insert_or_update is hash-idempotent, but the echo
    # mention_count bump and the ×0.92 decay tick are NOT — a replay double-applies
    # both. event_provenance rows for this evidence are written inside this very
    # transaction (link_event_provenance in _insert_or_update / _bump_event), so
    # their presence is a durable "already merged" marker with no crash window:
    # provenance exists iff the full merge committed. Skip the replay; the caller
    # still advances the cursor and completes the job.
    evidence_id = candidates[0].evidence_id
    if evidence_id is not None and _evidence_already_merged(conn, evidence_id):
        _log.info(
            "merge.skip_replayed_evidence",
            extra={"session_id": session_id, "evidence_id": evidence_id},
        )
        return stats

    try:
        with conn:
            for event in candidates:
                # J2: derive the decision key at the single mint choke point so
                # every extraction path (broad, patterns, co-capture) gets one.
                if event.decision_key is None:
                    from memlora.extraction.decision_key import derive_decision_key
                    event.decision_key = derive_decision_key(event.payload, event.event_type)
                # R3: fold a near-identical restatement (echo) into its canonical
                # BEFORE supersession can fire — bump mention_count, never mint a
                # near-dup or let a recitation supersede the original. Lossless.
                echo_id = _find_echo(conn, event)
                if echo_id is not None:
                    _bump_event(conn, echo_id, event)
                    event.id = echo_id
                    stats["updated"] += 1
                    continue
                outcome, row_id = _insert_or_update(conn, event)
                stats[outcome] += 1
                event.id = row_id  # needed by cascade_component_status

                # Always store embeddings when the model is available — recall
                # (rank-and-return, agent judges) benefits from semantic retrieval
                # regardless of whether semantic *supersession* is enabled.
                # The precision failure was about auto-supersession, not retrieval.
                # _store_event_embedding is best-effort and never breaks the merge.
                _store_event_embedding(conn, row_id, event)
                # Gated supersession is the baseline (temporal + authority +
                # provenance). `embed_events` (config.embedding_enabled) still
                # controls whether the semantic axis fires for auto-supersession.
                sup_ids = find_superseded(
                    conn, event, use_embeddings=embed_events, use_cross_encoder=use_cross_encoder
                )
                stats["superseded"] += apply_supersession(conn, row_id, sup_ids)
                stats["superseded"] += _cross_type_dedup(conn, row_id, event)

                if event.event_type == "COMPONENT_STATUS":
                    stats["cascaded"] += cascade_component_status(conn, event)

            stats["archived"] += _apply_decay_inner(conn, project_id, session_id)
            _invalidate_projection_inner(conn, project_id)

    except Exception as exc:
        _log.error(
            "merge.transaction_failed",
            extra={"session_id": session_id, "error": str(exc)},
        )
        try:
            insert_extraction_failure(
                conn,
                project_id=project_id,
                session_id=session_id,
                stage="delta.merge",
                error_message=str(exc),
                raw_input_path="",
            )
        except Exception:
            pass
        raise

    _log.info("merge.complete", extra={"session_id": session_id, **stats})
    return stats


# ── transaction-internal helpers (no conn.commit()) ───────────────────────────

def _evidence_already_merged(conn: sqlite3.Connection, evidence_id: int) -> bool:
    """True if a prior merge of this raw evidence already committed (audit P1).

    event_provenance links every minted/bumped event to its raw evidence inside
    the merge transaction, so a row here is a durable, crash-safe witness that the
    merge ran to completion. Indexed by raw_evidence_id — a single sub-ms lookup.
    """
    row = conn.execute(
        "SELECT 1 FROM event_provenance WHERE raw_evidence_id = ? LIMIT 1",
        (evidence_id,),
    ).fetchone()
    return row is not None


def _store_event_embedding(conn: sqlite3.Connection, event_id: int, event: Event) -> None:
    """Compute + store the event's embedding from its composed input (E1).
    Best-effort, no-op if the model is unavailable (degrades to lexical)."""
    try:
        from memlora.embedding.input import embedding_input
        from memlora.embedding.model import EMBEDDING_MODEL_VERSION, embed_text
        from memlora.embedding.store import upsert_embedding

        vec = embed_text(embedding_input(event.payload, event.event_type))
        if vec is not None:
            upsert_embedding(conn, event_id, vec, EMBEDDING_MODEL_VERSION)
    except Exception as exc:  # never let embedding failures break the merge
        _log.warning("merge.embedding_failed", extra={"event_id": event_id, "error": str(exc)})


def _insert_or_update(
    conn: sqlite3.Connection,
    event: Event,
) -> tuple[str, int]:
    """INSERT or UPDATE without committing — for use inside execute_merge."""
    try:
        cursor = conn.execute(
            """
            INSERT INTO events
                (project_id, session_id, created_at, event_type,
                 payload, content_hash, weight, mention_count, evidence_id,
                 decision_key)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.project_id,
                event.session_id,
                event.created_at,
                event.event_type,
                json.dumps(event.payload, sort_keys=True, separators=(",", ":")),
                event.content_hash,
                event.weight,
                event.mention_count,
                event.evidence_id,
                event.decision_key,
            ),
        )
        row_id = cursor.lastrowid  # type: ignore[assignment]
    except sqlite3.IntegrityError:
        row = conn.execute(
            "SELECT id, session_id FROM events WHERE project_id = ? AND content_hash = ?",
            (event.project_id, event.content_hash),
        ).fetchone()
        row_id = row["id"]
        # F9: mention/weight bumps credit CROSS-SESSION restatements only.
        # Under delta ingest (I2), an in-session duplicate is overwhelmingly the
        # overlap window re-extracting the same lines on the next firing — not a
        # genuine restatement. Bumping those systematically inflated events that
        # straddle firing boundaries, letting boundary junk outrank real
        # decisions inside the block's section budgets (GAMMA_CK_TEST S2: two
        # mc=4 question-fragments displaced the default-alias decision).
        cross_session = row["session_id"] != event.session_id
        bump = WEIGHT_INCREMENT_ON_DEDUP if cross_session else 0.0
        mention = 1 if cross_session else 0
        # archived=0 always: a fresh mention RESURRECTS an archived event even
        # in-session — re-statement is direct evidence of renewed relevance.
        conn.execute(
            """
            UPDATE events
            SET mention_count = mention_count + ?,
                weight        = MIN(weight + ?, ?),
                archived      = 0,
                evidence_id   = COALESCE(evidence_id, ?)
            WHERE id = ?
            """,
            (
                mention,
                bump,
                MAX_EVENT_WEIGHT,
                event.evidence_id,
                row_id,
            ),
        )
        outcome = "updated"
    else:
        outcome = "inserted"

    if event.evidence_id is not None:
        from memlora.storage.evidence import link_event_provenance

        link_event_provenance(
            conn,
            event_id=row_id,
            evidence_id=event.evidence_id,
            extractor_version="memlora.v2",
        )
    return outcome, row_id


def _find_echo(conn: sqlite3.Connection, event: Event) -> int | None:
    """Return the id of a near-identical active same-type event (a restatement), else None.

    Only very-high lexical overlap (>= _ECHO_JACCARD) qualifies, so refinements and
    distinct facts are never folded. Exact-hash dups are excluded here (handled by
    _insert_or_update's UNIQUE-constraint path)."""
    new_desc = event.payload.get("description", "")
    if not new_desc.strip():
        return None
    rows = conn.execute(
        """
        SELECT id, payload FROM events
        WHERE project_id    = ?
          AND event_type    = ?
          AND archived      = 0
          AND superseded_by IS NULL
          AND content_hash != ?
        """,
        (event.project_id, event.event_type, event.content_hash),
    ).fetchall()
    best_id, best_j = None, _ECHO_JACCARD
    for row in rows:
        cand = json.loads(row["payload"]).get("description", "")
        j = jaccard_similarity(new_desc, cand)
        if j >= best_j:
            best_j, best_id = j, row["id"]
    return best_id


def _bump_event(conn: sqlite3.Connection, event_id: int, event: Event) -> None:
    """Fold an echo into its canonical: bump mention_count + weight, link provenance.

    Mirrors the exact-hash dedup path in _insert_or_update — the recitation strengthens
    the canonical instead of creating a near-duplicate row. F9: the bump credits
    cross-session echoes only; an in-session near-duplicate under delta ingest is
    overwhelmingly the overlap window re-extracting boundary lines."""
    row = conn.execute("SELECT session_id FROM events WHERE id = ?", (event_id,)).fetchone()
    cross_session = row is not None and row["session_id"] != event.session_id
    conn.execute(
        """
        UPDATE events
        SET mention_count = mention_count + ?,
            weight        = MIN(weight + ?, ?),
            evidence_id   = COALESCE(evidence_id, ?)
        WHERE id = ?
        """,
        (
            1 if cross_session else 0,
            WEIGHT_INCREMENT_ON_DEDUP if cross_session else 0.0,
            MAX_EVENT_WEIGHT,
            event.evidence_id,
            event_id,
        ),
    )
    if event.evidence_id is not None:
        from memlora.storage.evidence import link_event_provenance

        link_event_provenance(
            conn,
            event_id=event_id,
            evidence_id=event.evidence_id,
            extractor_version="memlora.v2",
        )


def _apply_decay_inner(
    conn: sqlite3.Connection,
    project_id: str,
    current_session_id: str,
) -> int:
    """Decay and archive without committing — for use inside execute_merge."""
    meta_key = f"{_META_KEY_PREFIX}{project_id}"
    last_row = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (meta_key,)
    ).fetchone()
    if last_row and last_row["value"] == current_session_id:
        return 0

    conn.execute(
        """
        UPDATE events
        SET weight = MAX(0.0, weight * ?)
        WHERE project_id = ?
          AND session_id != ?
          AND archived   = 0
        """,
        (DECAY_FACTOR, project_id, current_session_id),
    )

    protected_placeholders = ",".join("?" * len(_PROTECTED_FROM_ARCHIVE))
    result = conn.execute(
        f"""
        UPDATE events
        SET archived = 1
        WHERE project_id = ?
          AND archived   = 0
          AND weight     < ?
          AND event_type NOT IN ({protected_placeholders})
        """,
        (project_id, ARCHIVE_THRESHOLD, *_PROTECTED_FROM_ARCHIVE),
    )
    archived_count = result.rowcount

    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        (meta_key, current_session_id),
    )
    return archived_count


def _cross_type_dedup(
    conn: sqlite3.Connection,
    new_event_id: int,
    event: "Event",
) -> int:
    """Supersede cross-type duplicates within _DEDUP_GROUP.

    When the same concept is captured under e.g. both CONSTRAINT_HARD and
    APPROACH_ABANDONED_DO_NOT_RETRY, keep only the highest-priority type.
    Returns the count of events superseded.
    """
    if event.event_type not in _DEDUP_GROUP:
        return 0

    new_priority = _DEDUP_PRIORITY[event.event_type]
    new_desc = event.payload.get("description", "")
    superseded = 0

    peer_types = [t for t in _DEDUP_GROUP if t != event.event_type]
    placeholders = ",".join("?" * len(peer_types))
    rows = conn.execute(
        f"""
        SELECT id, event_type, payload FROM events
        WHERE project_id    = ?
          AND event_type    IN ({placeholders})
          AND archived      = 0
          AND superseded_by IS NULL
        """,
        (event.project_id, *peer_types),
    ).fetchall()

    for row in rows:
        import json as _json
        peer_desc = _json.loads(row["payload"]).get("description", "")
        if not supersedes(new_desc, peer_desc):
            continue

        peer_priority = _DEDUP_PRIORITY[row["event_type"]]
        if new_priority < peer_priority:
            # New event wins — supersede the peer
            conn.execute(
                "UPDATE events SET superseded_by = ? WHERE id = ?",
                (new_event_id, row["id"]),
            )
            superseded += 1
        else:
            # Peer wins — mark new event as superseded by peer
            conn.execute(
                "UPDATE events SET superseded_by = ? WHERE id = ?",
                (row["id"], new_event_id),
            )

    return superseded


def _invalidate_projection_inner(conn: sqlite3.Connection, project_id: str) -> None:
    """Set high_water to -1 (sentinel) to force full projection rebuild."""
    conn.execute(
        "UPDATE state_projections SET event_id_high_water = -1 WHERE project_id = ?",
        (project_id,),
    )
