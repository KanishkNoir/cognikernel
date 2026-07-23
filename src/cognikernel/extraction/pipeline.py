"""Extraction pipeline orchestrator — Stage 2.

extract_session() is a pure transformation: no database I/O.
persist_events() writes the result to the storage layer.

Backpressure thresholds (from ARCHITECTURE.md §6):
  ≤ 500 KB  — process in foreground
  > 500 KB  — process last 500 KB in foreground; older content deferred
  > 5 MB    — hard cap: extract only the last 5 MB tail
"""
from __future__ import annotations

import contextvars
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass

from cognikernel.storage.events import Event, insert_event, insert_extraction_failure

_log = logging.getLogger("cognikernel.extraction")

# Per-call extractor selection set by extract_session(). None means "no explicit
# selection" (fall back to the env var, then legacy). The COGNIKERNEL_EXTRACTOR env
# var always wins over this so ops/tests can force a mode regardless of config.
_EXTRACTOR_OVERRIDE: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "cognikernel_extractor_override", default=None
)

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
    extractor: str | None = None,
) -> list[Event]:
    """Extract structured events from a transcript and optional git diff.

    Pure transformation — call persist_events() to write to the database.

    `extractor` selects the Stage-2 backend (legacy | v1 | v1-broad | v2 |
    v2-broad), normally `config.extractor`. The `COGNIKERNEL_EXTRACTOR` env var, when
    set, overrides it (ops/test escape hatch). None means "use the env var, else
    legacy" — the historical behavior.
    """
    # Lazy imports avoid circular-import issues at module load time.
    from cognikernel.extraction.tokenize import tokenize
    from cognikernel.extraction.trie import get_scanner
    from cognikernel.extraction.windowing import extract_events_from_matches
    from cognikernel.extraction.classifier import classify_event
    from cognikernel.extraction.hashing import compute_content_hash
    from cognikernel.extraction.git_augment import extract_git_events, cross_reference_signals

    token = _EXTRACTOR_OVERRIDE.set(extractor)
    try:
        return _extract_session_impl(
            transcript, session_meta, git_diff,
            tokenize, get_scanner, extract_events_from_matches,
            classify_event, compute_content_hash,
            extract_git_events, cross_reference_signals,
        )
    finally:
        _EXTRACTOR_OVERRIDE.reset(token)


def _extract_session_impl(
    transcript: str,
    session_meta: SessionMetadata,
    git_diff,
    tokenize,
    get_scanner,
    extract_events_from_matches,
    classify_event,
    compute_content_hash,
    extract_git_events,
    cross_reference_signals,
) -> list[Event]:
    transcript = _apply_size_limits(transcript, session_meta)
    events: list[Event] = []

    # ── Transcript extraction ─────────────────────────────────────────────────
    sentences: list = []
    try:
        from cognikernel.extraction.file_mentions import extract_file_mention_events
        from cognikernel.extraction.normalize import normalize_description
        from cognikernel.extraction.patterns import scan_patterns
        from cognikernel.extraction.sanitize import sanitize_description
        from cognikernel.extraction.windowing import extract_co_captures
        sentences = tokenize(transcript)

        # #41 — capture schema decisions stated in DDL code blocks (which
        # sanitize strips) before they're lost; keyed canonically by
        # schema_decisions so latest-wins reconciles them. Mode-independent:
        # extended into whichever path returns. Fail-open → [].
        from cognikernel.extraction.schema_decisions import extract_schema_decisions
        schema_events = extract_schema_decisions(sentences, session_meta)

        # Broad mode: the head classifies EVERY prose sentence — high-recall candidate
        # gen + high-precision learned filter. v1-broad uses the frozen head; v2-broad
        # uses the SetFit fine-tuned head (salience_v2). Gated behind COGNIKERNEL_EXTRACTOR.
        if _extractor_mode() in ("v1-broad", "v2-broad"):
            head = _head_module()
            if head.is_available():
                broad = _extract_via_head(sentences, session_meta, head)
                if broad is not None:
                    broad.extend(extract_file_mention_events(
                        sentences, session_meta.project_id, session_meta.session_id))
                    broad.extend(schema_events)
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
        from cognikernel.extraction.triple import augment_with_triple
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
        events.extend(schema_events)
    except Exception as exc:
        # Propagate (M4): swallowing here acked the job through COMPLETED with
        # zero events — a genuinely broken extractor silently lost every
        # session's memory and the EXTRACTOR_BUG dead-letter class could never
        # fire. The evidence is durable; callers fail_job() and the job becomes
        # replayable once the bug is fixed. Git augmentation below stays
        # fail-open — it is auxiliary signal, not the session's memory.
        _log.error(
            "transcript extraction failed",
            extra={"session_id": session_meta.session_id, "error": str(exc)},
        )
        raise

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

    Resolution order: COGNIKERNEL_EXTRACTOR env var (ops/test override) > the per-call
    selection from extract_session(extractor=...) (normally config.extractor) >
    legacy. v1* uses the frozen-backbone head (salience); v2* uses the SetFit
    fine-tuned head (salience_v2). Plain modes filter legacy candidates; -broad
    classifies all sentences.
    """
    env = os.environ.get("COGNIKERNEL_EXTRACTOR")
    if env:
        return env.lower()
    override = _EXTRACTOR_OVERRIDE.get()
    if override:
        return override.lower()
    return "legacy"


def _head_module():
    """The salience head module for the current mode: salience_v2 for v2*, else salience."""
    if _extractor_mode() in ("v2", "v2-broad"):
        from cognikernel.extraction import salience_v2
        return salience_v2
    from cognikernel.extraction import salience
    return salience


def _use_head_extractor() -> bool:
    """True for filter mode (v1/v2) when the selected head is available."""
    if _extractor_mode() not in ("v1", "v2"):
        return False
    return _head_module().is_available()


_MIN_CONTENT_WORDS = 4
_CONTENT_WORD_RE = re.compile(r"[a-z0-9]{3,}")

# R1 — memory-meta self-reference: the assistant narrating CogniKernel's OWN memory
# ("the session context has…", "the recall surfaces…", "graveyard records…") rather
# than stating a project fact. These are extraction echoes from recall-heavy sessions.
# We DEMOTE (not drop): weight collapses so they fall off the budget-ranked block while
# staying in the store — real facts survive via their canonical (non-meta) capture, so
# the retention gate stays green. Terms chosen to NOT match real project facts (e.g.
# 'in-memory'/'in-process' decisions are excluded; 'graveyard' is a CogniKernel-only term).
_MEMORY_META_RE = re.compile(
    r"\b(session[- ]context|injection block|injected (?:session )?context|cognikernel|"
    r"stop hook|graveyard|pending confirmation|memory confirms|recorded in memory|"
    r"from memory|the recall (?:surfaces|surfaced|returns|returned|results|mentions|shows|tool)|"
    # Claude Code compaction-summary instructions leak into transcripts when a
    # session compacts mid-run; they are harness meta, not project facts
    # (GAMMA_CK_TEST: "Resume directly — do not acknowledge the summary"
    # landed in Hard constraints).
    r"resume directly|do not acknowledge the summary|do not recap what was happening|"
    r"continue the conversation from where it left off|"
    # J5: leak shapes collected from the 7 benchmark DBs (scripts/_j5_meta_scan.py).
    # Event-type tokens narrated in prose ("There's an APPROACH_ABANDONED_DO_NOT_RETRY
    # entry recording…") — underscore forms only, so prose like "hard constraint" stays:
    r"approach_abandoned\w*|constraint_hard|constraint_soft|thread_open|component_status|"
    # supersession governance narration (not the superseded fact itself):
    r"(?:now|explicitly) superseded|rejection is superseded|superseded abandoned|"
    # memory-reference framing around a fact whose canonical capture exists separately:
    r"memory (?:shows|says)|recorded decision|decision to record|decision log|"
    r"locked in the project memory|prior decision being overridden|entry recording|"
    # the MCP server instructions themselves leaking into extraction:
    r"call recall\b|missing from the block)\b",
    re.IGNORECASE,
)
_META_DEMOTE = 0.15  # weight multiplier for memory-meta sentences
_FRAG_DEMOTE = 0.4   # weight multiplier for context-dependent fragments (J5.2)

# Deterministic backstop for label-value facts ("Max attempts: 2 (...)",
# "Recovery window: 30 s", "Open threshold: 3 ..."). The salience head was not
# trained on this register and argmaxes it to NOISE, yet these lines carry the
# numerically-precise decisions recall probes target (GAMMA_CK_TEST S2-T3: the
# agent answered "no attempt count decided" because all three such lines were
# dropped). Uses tokenize.is_label_value_line (shared predicate) + a value-ish
# token requirement so bare prose with a leading clause-colon doesn't qualify.
_VALUEISH_RE = re.compile(r"\d|\btrue\b|\bfalse\b|\benabled\b|\bdisabled\b|\bnone\b|\balways\b|\bnever\b", re.I)
_LABEL_FACT_CONF = 0.45  # modest: deterministic floor, below head-confident events


def _extract_via_head(sentences: list, session_meta: SessionMetadata, head=None) -> list[Event] | None:
    """Broad mode: classify every prose sentence; keep non-NOISE as typed events.

    High-recall candidate generation (all prose, both roles) + the head as the
    salience filter and typer. `head` is the salience module (v1 or v2). Returns None
    if the model drops out mid-run.
    """
    if head is None:
        head = _head_module()
    from cognikernel.extraction.authority import default_authority_for_role
    from cognikernel.extraction.hashing import compute_content_hash
    from cognikernel.extraction.head_input import compose_head_input
    from cognikernel.extraction.normalize import normalize_description
    from cognikernel.extraction.sanitize import is_context_dependent_fragment, sanitize_description
    from cognikernel.extraction.triple import augment_with_triple
    from cognikernel.extraction.windowing import _is_structural_label

    # P2: a context-trained head is fed "[role] prev || current"; a bare head gets
    # the sentence alone. The head declares its format so the two can't be mixed.
    use_context = getattr(head, "expects_context", lambda: False)()
    provenance = f"salience_{_extractor_mode()}".replace("-", "_")
    events: list[Event] = []
    seen: set[str] = set()
    prev_desc = ""  # previous prose sentence, for context composition
    for s in sentences:
        if s.is_code_block:
            continue
        raw_text = s.text.strip()
        if not raw_text or _is_structural_label(raw_text):
            continue
        desc = sanitize_description(raw_text)
        if not desc or len(_CONTENT_WORD_RE.findall(desc.lower())) < _MIN_CONTENT_WORDS:
            continue
        # content_hash and the stored description stay BARE (desc); only the
        # CLASSIFICATION input is composed, so event identity is unchanged.
        head_input = compose_head_input(desc, s.role, prev_desc) if use_context else desc
        prev_desc = desc  # for the next sentence
        scored = head.classify_scored(head_input)
        if scored is None:
            return None
        label, conf = scored
        if label == "NOISE":
            # Deterministic label-value backstop (see _VALUEISH_RE above).
            from cognikernel.extraction.tokenize import is_label_value_line
            if is_label_value_line(desc) and _VALUEISH_RE.search(desc):
                label, conf = "DECISION", _LABEL_FACT_CONF
            else:
                continue
        if label == "THREAD":
            label = "THREAD_CLOSE" if _THREAD_CLOSE_VERB.search(desc) else "THREAD_OPEN"
        desc_norm = normalize_description(desc)
        if not desc_norm:
            continue
        # J5.2 — retype BEFORE hashing so event identity reflects the final type.
        # A context-dependent aside must never be budget-exempt mandatory.
        is_frag = is_context_dependent_fragment(desc_norm)
        if is_frag and label == "CONSTRAINT_HARD":
            label = "CONSTRAINT_SOFT"
        chash = compute_content_hash(label, desc_norm)
        if chash in seen:
            continue
        seen.add(chash)
        is_meta = bool(_MEMORY_META_RE.search(desc))
        prov = provenance + ("+meta" if is_meta else "") + ("+frag" if is_frag else "")
        weight = conf * (_META_DEMOTE if is_meta else 1.0) * (_FRAG_DEMOTE if is_frag else 1.0)
        ev = Event(
            project_id=session_meta.project_id,
            session_id=session_meta.session_id,
            event_type=label,
            payload={
                "description": desc_norm, "rationale": "", "confidence": conf,
                "source_role": s.role, "matched_phrase": "HEAD", "affected_files": [],
                "authority": default_authority_for_role(s.role), "provenance": prov,
            },
            content_hash=chash, weight=weight,
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
    from cognikernel.extraction.head_input import compose_head_input
    # Filter mode has no ordered prev sentence (it scores curated candidates), so a
    # context head gets role-only composition — the same "[role] text" shape the
    # frozen eval used, which is consistent with its training.
    use_context = getattr(head, "expects_context", lambda: False)()

    kept: list[Event] = []
    for e in events:
        desc = e.payload.get("description", "")
        head_input = compose_head_input(desc, e.payload.get("source_role", ""), "") if use_context else desc
        scored = head.classify_scored(head_input)
        if scored is None:
            return None  # model dropped out — signal legacy fallback
        label, conf = scored
        if label == "NOISE":
            continue
        if label == "THREAD":
            label = "THREAD_CLOSE" if _THREAD_CLOSE_VERB.search(desc) else "THREAD_OPEN"
        # J5.2 — same fragment contract as the broad path.
        from cognikernel.extraction.sanitize import is_context_dependent_fragment
        is_frag = is_context_dependent_fragment(desc)
        if is_frag and label == "CONSTRAINT_HARD":
            label = "CONSTRAINT_SOFT"
        e.event_type = label
        e.payload["confidence"] = conf
        suffix = "+head"
        if _MEMORY_META_RE.search(desc):
            suffix += "+meta"
            e.weight = (e.weight or conf) * _META_DEMOTE
        if is_frag:
            suffix += "+frag"
            e.weight = (e.weight or conf) * _FRAG_DEMOTE
        e.payload["provenance"] = (e.payload.get("provenance", "") + suffix).lstrip("+")
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
