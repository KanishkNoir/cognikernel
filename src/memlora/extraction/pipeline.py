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
import os
import re
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
        from memlora.extraction.normalize import normalize_description
        from memlora.extraction.patterns import scan_patterns
        from memlora.extraction.sanitize import sanitize_description
        from memlora.extraction.windowing import extract_co_captures
        sentences = tokenize(transcript)

        # Broad mode: the head classifies EVERY prose sentence — high-recall candidate
        # gen + high-precision learned filter. v1-broad uses the frozen head; v2-broad
        # uses the SetFit fine-tuned head (salience_v2). Gated behind MEMLORA_EXTRACTOR.
        if _extractor_mode() in ("v1-broad", "v2-broad"):
            head = _head_module()
            if head.is_available():
                broad = _extract_via_head(sentences, session_meta, head)
                if broad is not None:
                    broad.extend(extract_file_mention_events(
                        sentences, session_meta.project_id, session_meta.session_id))
                    return broad
                _log.info("salience head unavailable — falling back to legacy")

        matches   = get_scanner().scan(sentences, transcript)
        raw       = extract_events_from_matches(
            sentences, matches,
            session_meta.project_id, session_meta.session_id,
        )
        # A-3: pattern-with-capture events run in parallel with the trie.
        # They use the same sentence list but their own scan algorithm so
        # captured subjects can ride along in the payload.
        pattern_events = scan_patterns(
            sentences, session_meta.project_id, session_meta.session_id,
        )
        # Pattern events skip the trie's structural-label filter (already
        # excluded by shape guards) but DO need sanitization + classification.
        # Drop any whose description sanitizes to empty (a matched token with no
        # recallable context is noise, not a fact).
        _sanitized: list = []
        for pe in pattern_events:
            pe.payload["description"] = sanitize_description(pe.payload["description"])
            if pe.payload["description"].strip():
                _sanitized.append(pe)
        pattern_events = _sanitized

        # A-4: co-capture the assistant's reply when a USER trie match landed.
        # These produce CONSTRAINT_SOFT events tagged
        # `authority=assistant_answer_to_user_question`, which the renderer
        # routes to a Pending Confirmation section.
        cocapture_events = extract_co_captures(
            sentences, matches,
            session_meta.project_id, session_meta.session_id,
        )

        combined = raw + pattern_events + cocapture_events
        # v1 B: the learned salience head filters NOISE out of the candidate set
        # and re-assigns the type. Falls back to the keyword classifier if the
        # head/model is unavailable, so extraction never breaks.
        classified = None
        if _use_head_extractor():
            classified = _filter_and_retype_with_head(combined)
        if classified is None:
            classified = [classify_event(e) for e in combined]
        from memlora.extraction.triple import augment_with_triple
        for e in classified:
            # A-1: strip prompt-verb prefixes BEFORE hashing so equivalent
            # descriptions normalize to the same content_hash, enabling dedup.
            desc = e.payload.get("description", "")
            e.payload["description"] = normalize_description(desc)
            e.content_hash = compute_content_hash(
                e.event_type, e.payload["description"]
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


# ── v1 B: learned salience head path ─────────────────────────────────────────

_THREAD_CLOSE_VERB = re.compile(
    r"\b(done|closed|complete|completed|finished|shipped|merged|resolved)\b",
    re.IGNORECASE,
)


def _extractor_mode() -> str:
    """legacy | v1 | v1-broad | v2 | v2-broad.

    v1* uses the frozen-backbone head (salience); v2* uses the SetFit fine-tuned head
    (salience_v2). Plain modes filter legacy candidates; -broad classifies all sentences.
    """
    return os.environ.get("MEMLORA_EXTRACTOR", "legacy").lower()


def _head_module():
    """The salience head module for the current mode: salience_v2 for v2*, else salience."""
    if _extractor_mode() in ("v2", "v2-broad"):
        from memlora.extraction import salience_v2
        return salience_v2
    from memlora.extraction import salience
    return salience


def _use_head_extractor() -> bool:
    """True for filter mode (v1/v2) when the selected head is available."""
    if _extractor_mode() not in ("v1", "v2"):
        return False
    return _head_module().is_available()


_MIN_CONTENT_WORDS = 4
_CONTENT_WORD_RE = re.compile(r"[a-z0-9]{3,}")


def _extract_via_head(sentences: list, session_meta: SessionMetadata, head=None) -> list[Event] | None:
    """Broad mode: classify every prose sentence; keep non-NOISE as typed events.

    High-recall candidate generation (all prose, both roles) + the head as the
    salience filter and typer. `head` is the salience module (v1 or v2). Returns None
    if the model drops out mid-run.
    """
    if head is None:
        head = _head_module()
    from memlora.extraction.authority import default_authority_for_role
    from memlora.extraction.hashing import compute_content_hash
    from memlora.extraction.normalize import normalize_description
    from memlora.extraction.sanitize import sanitize_description
    from memlora.extraction.triple import augment_with_triple
    from memlora.extraction.windowing import _is_structural_label

    provenance = f"salience_{_extractor_mode()}".replace("-", "_")
    events: list[Event] = []
    seen: set[str] = set()
    for s in sentences:
        if s.is_code_block:
            continue
        raw_text = s.text.strip()
        if not raw_text or _is_structural_label(raw_text):
            continue
        desc = sanitize_description(raw_text)
        if not desc or len(_CONTENT_WORD_RE.findall(desc.lower())) < _MIN_CONTENT_WORDS:
            continue
        scored = head.classify_scored(desc)
        if scored is None:
            return None
        label, conf = scored
        if label == "NOISE":
            continue
        if label == "THREAD":
            label = "THREAD_CLOSE" if _THREAD_CLOSE_VERB.search(desc) else "THREAD_OPEN"
        desc_norm = normalize_description(desc)
        if not desc_norm:
            continue
        chash = compute_content_hash(label, desc_norm)
        if chash in seen:
            continue
        seen.add(chash)
        ev = Event(
            project_id=session_meta.project_id,
            session_id=session_meta.session_id,
            event_type=label,
            payload={
                "description": desc_norm, "rationale": "", "confidence": conf,
                "source_role": s.role, "matched_phrase": "HEAD", "affected_files": [],
                "authority": default_authority_for_role(s.role), "provenance": provenance,
            },
            content_hash=chash, weight=conf,
        )
        augment_with_triple(ev)
        events.append(ev)
    return events


def _filter_and_retype_with_head(events: list[Event], head=None) -> list[Event] | None:
    """Drop NOISE candidates and re-assign the type from the learned head (v1/v2).

    Candidate generation (trie + patterns + co-capture) stays as the high-recall
    front end; the head is the high-precision filter + typer over that curated
    set. This is robust to a modestly-sized head — it never has to judge the full
    sentence stream, only the already-surfaced candidates.

    Returns None if the model drops out mid-run so the caller falls back to the
    keyword classifier rather than silently losing the session.
    """
    if head is None:
        head = _head_module()

    kept: list[Event] = []
    for e in events:
        desc = e.payload.get("description", "")
        scored = head.classify_scored(desc)
        if scored is None:
            return None  # model dropped out — signal legacy fallback
        label, conf = scored
        if label == "NOISE":
            continue
        if label == "THREAD":
            label = "THREAD_CLOSE" if _THREAD_CLOSE_VERB.search(desc) else "THREAD_OPEN"
        e.event_type = label
        e.payload["confidence"] = conf
        e.payload["provenance"] = (e.payload.get("provenance", "") + "+head").lstrip("+")
        kept.append(e)
    return kept


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
