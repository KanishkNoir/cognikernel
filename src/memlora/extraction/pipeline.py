"""Extraction pipeline orchestrator — Stage 2.

extract_session() is a pure transformation: no database I/O.
persist_events() writes the result to the storage layer.

Backpressure thresholds (from ARCHITECTURE.md §6):
  ≤ 500 KB  — process in foreground
  > 500 KB  — process last 500 KB in foreground; older content deferred
  > 5 MB    — hard cap: extract only the last 5 MB tail
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass

from memlora.storage.events import Event, insert_event, insert_extraction_failure

_log = logging.getLogger("memlora.extraction")

_FOREGROUND_BYTES = 500 * 1_024          # 500 KB
_HARD_CAP_BYTES   = 5 * 1_024 * 1_024   # 5 MB


@dataclass
class SessionMetadata:
    project_id: str
    session_id: str
    started_at: int   # Unix milliseconds
    ended_at: int     # Unix milliseconds


def extract_session(
    transcript: str,
    session_meta: SessionMetadata,
    git_diff: str | None = None,
) -> list[Event]:
    """Extract structured events from a transcript and optional git diff.

    Pure transformation — call persist_events() to write to the database.
    """
    # Lazy imports avoid circular-import issues at module load time.
    from memlora.extraction.tokenize import tokenize
    from memlora.extraction.trie import get_scanner
    from memlora.extraction.windowing import extract_events_from_matches
    from memlora.extraction.classifier import classify_event
    from memlora.extraction.hashing import compute_content_hash
    from memlora.extraction.git_augment import extract_git_events, cross_reference_signals

    transcript = _apply_size_limits(transcript, session_meta)
    events: list[Event] = []

    # ── Transcript extraction ─────────────────────────────────────────────────
    sentences: list = []
    try:
        from memlora.extraction.file_mentions import extract_file_mention_events
        sentences = tokenize(transcript)
        matches   = get_scanner().scan(sentences, transcript)
        raw       = extract_events_from_matches(
            sentences, matches,
            session_meta.project_id, session_meta.session_id,
        )
        classified = [classify_event(e) for e in raw]
        from memlora.extraction.triple import augment_with_triple
        for e in classified:
            e.content_hash = compute_content_hash(
                e.event_type, e.payload.get("description", "")
            )
            augment_with_triple(e)
        events.extend(classified)

        mention_events = extract_file_mention_events(
            sentences, session_meta.project_id, session_meta.session_id
        )
        events.extend(mention_events)
    except Exception as exc:
        _log.error(
            "transcript extraction failed",
            extra={"session_id": session_meta.session_id, "error": str(exc)},
        )

    # ── Git augmentation ──────────────────────────────────────────────────────
    if git_diff:
        try:
            git_events = extract_git_events(
                git_diff, session_meta.project_id, session_meta.session_id
            )
            events = cross_reference_signals(events, git_events)
            events.extend(git_events)
        except Exception as exc:
            _log.warning(
                "git augmentation failed",
                extra={"session_id": session_meta.session_id, "error": str(exc)},
            )

    return events


def persist_events(
    events: list[Event],
    conn: sqlite3.Connection,
    session_meta: SessionMetadata | None = None,
) -> list[int]:
    """Write extracted events to storage. Returns row IDs of inserted/updated rows."""
    ids: list[int] = []
    for event in events:
        try:
            ids.append(insert_event(conn, event))
        except Exception as exc:
            _log.error(
                "event persist failed",
                extra={"content_hash": event.content_hash, "error": str(exc)},
            )
            if session_meta is not None:
                try:
                    insert_extraction_failure(
                        conn,
                        project_id=event.project_id,
                        session_id=event.session_id,
                        stage="pipeline.persist",
                        error_message=str(exc),
                        raw_input_path="",
                    )
                except Exception:
                    pass
    return ids


# ── internals ────────────────────────────────────────────────────────────────

def _apply_size_limits(transcript: str, meta: SessionMetadata) -> str:
    encoded = transcript.encode("utf-8", errors="replace")
    if len(encoded) > _HARD_CAP_BYTES:
        _log.warning(
            "transcript exceeds 5 MB hard cap — truncating to tail",
            extra={"session_id": meta.session_id, "size_bytes": len(encoded)},
        )
        return encoded[-_HARD_CAP_BYTES:].decode("utf-8", errors="replace")
    return transcript
