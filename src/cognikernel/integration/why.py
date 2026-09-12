"""`cognikernel why <subject>` (S5 T-502): one claim, where it came from, and what
it replaced — sessions shown in order, and every field the store did not record
named rather than left out.

Presentation over `storage.provenance`. The one inference is the source
sentence: the v2 extractor records no offsets, so the sentence is located inside
the claim's own evidence by content-word coverage, and shown as located, never
as recorded. Below the coverage floor nothing is shown — a guessed origin would
be worse than an admitted gap.
"""
from __future__ import annotations

import datetime
import logging
import re
import sqlite3
from dataclasses import asdict, dataclass
from typing import Any

from cognikernel.model import Event
from cognikernel.storage.events import get_events_for_projection
from cognikernel.storage.evidence import load_evidence, load_full_transcript
from cognikernel.storage.projections import Projection, build_projection
from cognikernel.storage.provenance import (
    SessionPosition,
    claim_provenance,
    claim_transitions,
    find_claims,
    is_live,
    session_order,
    supersession_chain,
)
from cognikernel.utils.decision_key import derive_decision_key
from cognikernel.utils.paths import canonicalize_path, is_bare_basename

_log = logging.getLogger("cognikernel.why")
_converter_warned = False

_TOKEN = re.compile(r"[a-z0-9_]+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")

# A located sentence must carry at least this share of the claim's content words.
_MIN_COVERAGE = 0.6
_MAX_SOURCE_CHARS = 400
# Bounds the walk back through a session's evidence chunks on a corrupt chain.
_MAX_CHUNKS = 10_000

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it",
    "of", "on", "or", "our", "that", "the", "this", "to", "was", "we", "will", "with",
})

# Gate annotations the quality layer writes into an admitted claim's payload.
_ADMISSION_KEYS = ("quality", "grounding")


@dataclass(frozen=True)
class SourceMatch:
    evidence_id: int   # the chunk the sentence was found in
    role: str          # "user" | "assistant" | "unknown"
    text: str
    coverage: float
    scope: str = "chunk"   # "chunk": inside evidence_id itself; "transcript": only in
                           # the session transcript up to evidence_id (a turn split
                           # across chunks)


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.casefold()) if t not in _STOPWORDS}


def _turns(source_type: str, raw: bytes) -> list[tuple[str, str]]:
    """The evidence's (role, text) turns, roles taken from the source records."""
    try:
        # Imports the extraction package, whose trie needs pyahocorasick. A broken
        # install must not crash `why`; the claim is still explainable without its
        # source sentence, and the warning says why the sentence is missing.
        from cognikernel.extraction.transcript import turns_from_source
    except ImportError as exc:
        global _converter_warned
        if not _converter_warned:  # once per process, not once per claim
            _log.warning("why: transcript converter unavailable, source sentences not located: %s", exc)
            _converter_warned = True
        return []
    try:
        return turns_from_source(source_type, raw.decode("utf-8", errors="replace"))
    except Exception:
        return []


def _best_sentence(turns: list[tuple[str, str]], wanted: set[str]) -> tuple[str, str, float] | None:
    best: tuple[str, str, float] | None = None
    for role, body in turns:
        for sentence in _SENTENCE_BREAK.split(body):
            sentence = sentence.strip()
            if not sentence:
                continue
            coverage = len(wanted & _content_tokens(sentence)) / len(wanted)
            if coverage >= _MIN_COVERAGE and (best is None or coverage > best[2]):
                best = (role, sentence[:_MAX_SOURCE_CHARS], coverage)
    return best


def locate_source(conn: sqlite3.Connection, evidence_id: int, description: str) -> SourceMatch | None:
    """The sentence a claim came from. Never raises.

    Searched chunk by chunk — the claim's own evidence, then each earlier chunk of
    the session, nearest first — so a match names the chunk that holds it. Only if
    no single chunk holds it (a turn split across two captures) is the whole
    transcript searched, and the match is then marked as located in the
    transcript up to this chunk rather than in it.
    """
    wanted = _content_tokens(description)
    if not wanted:
        return None
    try:
        evidence = load_evidence(conn, evidence_id)
    except Exception:
        return None
    if evidence is None:
        return None

    chunk, chunk_id, visited = evidence, evidence_id, set()
    while chunk is not None and chunk_id not in visited and len(visited) < _MAX_CHUNKS:
        visited.add(chunk_id)
        hit = _best_sentence(_turns(chunk.source_type, chunk.content), wanted)
        if hit is not None:
            role, text, coverage = hit
            return SourceMatch(chunk_id, role, text, round(coverage, 2), "chunk")
        if chunk.prev_evidence_id is None:
            break
        chunk_id = chunk.prev_evidence_id
        try:
            chunk = load_evidence(conn, chunk_id)
        except Exception:
            chunk = None

    if evidence.prev_evidence_id is None:
        return None
    try:
        full = load_full_transcript(conn, evidence_id)
    except Exception:
        return None
    hit = _best_sentence(_turns(evidence.source_type, full), wanted)
    if hit is None:
        return None
    role, text, coverage = hit
    return SourceMatch(evidence_id, role, text, round(coverage, 2), "transcript")


# ── structure (shared by text and --json) ─────────────────────────────────────


def _session(order: dict[str, SessionPosition], session_id: str) -> dict[str, Any]:
    pos = order.get(session_id)
    return {
        "id": session_id,
        "position": pos.position if pos else None,
        "total": pos.total if pos else None,
        "first_seen": pos.first_seen if pos else None,
    }


def _status(event: Event, transition: dict[str, Any]) -> str:
    if event.superseded_by is not None:
        return "superseded"
    if event.archived:
        return "archived"
    return "active"


# Projection buckets, as the block names them. Decisions keep the projection's own
# order (it is the ranking); the other buckets are ordered by weight here.
_BUCKETS = (
    ("hard_constraints", "hard constraints"),
    ("ranked_decisions", "decisions"),
    ("graveyard", "do-not-retry entries"),
    ("active_threads", "open threads"),
    ("component_map", "components"),
)


def _ranking(conn: sqlite3.Connection, project_id: str) -> Projection | None:
    """The live ranking with each rec's weight factors attached. Writes nothing.

    rebuild_projection backfills missing decision keys in the store before it
    consolidates; they are derived on these in-memory events instead, as
    `show --as-of` does, so the weights match the block without a write.
    """
    try:
        events = get_events_for_projection(conn, project_id)
        for event in events:
            if event.decision_key is None:
                event.decision_key = derive_decision_key(event.payload, event.event_type)
        return build_projection(conn, project_id, events, explain_weights=True)
    except Exception as exc:
        _log.warning("why: ranking weights unavailable: %s", exc)
        return None


def _importance(event: Event, projection: Projection | None) -> dict[str, Any]:
    """§14 "why was this considered important?": the weight the block ranks by, as
    its six factors — or why the claim is not ranked at all."""
    if event.superseded_by is not None:
        return {"ranked": False, "folded_into": None,
                "reason": f"superseded by #{event.superseded_by}; only live claims are ranked"}
    if event.archived:
        return {"ranked": False, "folded_into": None, "reason": "archived; only live claims are ranked"}
    if projection is None:
        return {"ranked": False, "folded_into": None, "reason": "the ranking could not be rebuilt (see the warning log)"}

    for attr, label in _BUCKETS:
        bucket = getattr(projection, attr)
        recs = list(bucket.values()) if isinstance(bucket, dict) else list(bucket)
        if attr != "ranked_decisions":
            recs.sort(key=lambda r: (-r["weight"], r.get("id") or 0))
        for rank, rec in enumerate(recs, 1):
            if rec.get("id") == event.id:
                detail = rec.get("weight_factors") or {}
                return {
                    "ranked": True,
                    "weight": rec["weight"],
                    "factors": detail.get("factors", {}),
                    "quality_demotes": detail.get("quality_demotes", []),
                    "sessions_ago": detail.get("sessions_ago"),
                    "mention_count": detail.get("mention_count"),
                    "affected_files": detail.get("affected_files", []),
                    "bucket": label,
                    "rank": rank,
                    "of": len(recs),
                }

    # Live but absent: consolidation kept one canonical claim for its topic, or the
    # component collapse kept the latest status for its path.
    key = event.decision_key if event.decision_key is not None else derive_decision_key(event.payload, event.event_type)
    if key:
        for rec in projection.hard_constraints + projection.ranked_decisions:
            if rec.get("decision_key") == key:
                return {"ranked": False, "folded_into": rec["id"],
                        "reason": f"folded into #{rec['id']}, the claim that ranks for this topic"}
    if event.event_type == "COMPONENT_STATUS":
        path = canonicalize_path(str(event.payload.get("path") or ""))
        if path and not is_bare_basename(path) and path in projection.component_map:
            rec = projection.component_map[path]
            return {"ranked": False, "folded_into": rec["id"],
                    "reason": f"folded into #{rec['id']}, the latest status for {path}"}
    return {"ranked": False, "folded_into": None, "reason": "live, but not in the ranked block"}


def _explain(conn: sqlite3.Connection, event: Event, order: dict[str, SessionPosition],
             projection: Projection | None) -> dict[str, Any]:
    payload = event.payload
    chain = supersession_chain(conn, event.id)
    transitions = claim_transitions(conn, [e.id for e in chain] + [event.id])
    provenance = claim_provenance(conn, event.id)

    source = None
    for row in provenance:
        source = locate_source(conn, row["evidence_id"], str(payload.get("description") or ""))
        if source is not None:
            break
    source_info = None
    if source is not None:
        # The chunk that holds the sentence may be an earlier one than any linked row.
        chunk = conn.execute(
            "SELECT session_id, captured_at FROM raw_evidence WHERE id = ?", (source.evidence_id,)
        ).fetchone()
        source_info = {
            **asdict(source),
            "captured_at": chunk["captured_at"] if chunk else None,
            "session": _session(order, chunk["session_id"]) if chunk else None,
        }

    return {
        "id": event.id,
        "event_type": event.event_type,
        "status": _status(event, transitions.get(event.id, {})),
        "superseded_by": event.superseded_by,
        **transitions.get(event.id, {"superseded_at": None, "supersede_reason": None, "archived_at": None}),
        "description": payload.get("description") or "",
        # Cascaded COMPONENT_STATUS claims record their explanation as "reason".
        "rationale": payload.get("rationale") or payload.get("reason") or None,
        "authority": payload.get("authority"),
        "confidence": payload.get("confidence"),
        # The weight stored at admission. The ranking recomputes its own weight and
        # does not read this one; `importance` is what the block ranks by.
        "stored_weight": event.weight,
        "importance": _importance(event, projection),
        "mention_count": event.mention_count,
        "created_at": event.created_at,
        "session": _session(order, event.session_id),
        "admission": {key: payload[key] for key in _ADMISSION_KEYS if payload.get(key)},
        "source": source_info,
        "provenance": [
            {
                "evidence_id": row["evidence_id"],
                "source_type": row["source_type"],
                "source_path": row["source_path"],
                "captured_at": row["captured_at"],
                "session": _session(order, row["session_id"]),
                "extractor_version": row["extractor_version"],
                "matched_phrase": row["matched_phrase"],
                "sentence_index": row["sentence_index"],
                "window_start": row["window_start"],
                "window_end": row["window_end"],
                "confidence": row["confidence"],
                "transformation_notes": row["transformation_notes"],
                "offsets_recorded": any(row[k] is not None for k in ("sentence_index", "window_start", "window_end")),
            }
            for row in provenance
        ],
        "commit": event.captured_at_sha,
        "history": [
            {
                "id": member.id,
                "event_type": member.event_type,
                "description": member.payload.get("description") or "",
                "created_at": member.created_at,
                "session": _session(order, member.session_id),
                "superseded_by": member.superseded_by,
                **transitions.get(member.id, {"superseded_at": None, "supersede_reason": None, "archived_at": None}),
                "live": is_live(member),
            }
            for member in chain
        ],
    }


def explain_claims(conn: sqlite3.Connection, project_id: str, subject: str, limit: int = 3) -> list[dict[str, Any]]:
    order = session_order(conn, project_id)
    claims = find_claims(conn, project_id, subject, limit=limit)
    projection = _ranking(conn, project_id) if claims else None
    return [_explain(conn, event, order, projection) for event in claims]


# ── text rendering ────────────────────────────────────────────────────────────

_LABEL_WIDTH = 12
_HISTORY_BEFORE = 3
_HISTORY_AFTER = 3


def _when(ms: int | None) -> str:
    if not ms:
        return "time not recorded"
    return datetime.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")


def _session_label(session: dict[str, Any]) -> str:
    if session["position"] is None:
        return f"session {session['id'][:8]} (order unknown)"
    return f"session {session['position']} of {session['total']}"


def _line(label: str, text: str) -> str:
    return f"  {label:<{_LABEL_WIDTH}} {text}"


def _transition_text(item: dict[str, Any]) -> str:
    if item["superseded_by"] is not None:
        at = _when(item["superseded_at"]) if item["superseded_at"] else "at a time not recorded"
        reason = item["supersede_reason"] or "reason not recorded"
        return f"superseded {at} by #{item['superseded_by']} · {reason}"
    if item.get("archived_at") or item.get("status") == "archived":
        return f"archived {_when(item.get('archived_at'))}"
    return ""


def _plural(n: int | None, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _importance_lines(claim: dict[str, Any]) -> list[str]:
    imp = claim["importance"]
    stored = claim["stored_weight"]
    stored_line = _line("", f"stored weight {stored:.2f} is not what ranks it" if isinstance(stored, (int, float))
                        else "no stored weight")
    if not imp["ranked"]:
        return [_line("importance", f"not ranked: {imp['reason']}"), stored_line]
    f = imp["factors"]
    files = "no files" if not imp["affected_files"] else _plural(len(imp["affected_files"]), "file")
    return [
        _line("importance", f"weight {imp['weight']:.2f} · rank {imp['rank']} of {imp['of']} {imp['bucket']}"),
        _line("", f"= base {f['base']:.2f} ({claim['event_type']}) × recency {f['recency']:.2f} "
                  f"({_plural(imp['sessions_ago'], 'session')} ago) × repetition {f['repetition']:.2f} "
                  f"({_plural(imp['mention_count'], 'mention')})"),
        _line("", f"  × centrality {f['centrality']:.2f} ({files}) × activity {f['activity']:.2f} ({files}) "
                  f"× type {f['type']:.2f}"),
        _line("", f"  × quality {f.get('quality', 1.0):.2f} "
                  f"({', '.join(label for label, _ in imp['quality_demotes']) or 'no demotes'})"),
        stored_line,
    ]


def _render_one(claim: dict[str, Any]) -> str:
    status = "active" if claim["status"] == "active" else _transition_text(claim)
    lines = [
        f"#{claim['id']}  {claim['event_type']}  {status}  ·  "
        f"{_session_label(claim['session'])}  ·  {_when(claim['created_at'])}",
        f"  {claim['description']}",
        _line("rationale", claim["rationale"] or "none recorded"),
    ]

    confidence = claim["confidence"]
    lines.append(_line("authority", " · ".join([
        claim["authority"] or "not recorded",
        f"confidence {confidence:.2f}" if isinstance(confidence, (int, float)) else "confidence not recorded",
        f"mentioned {claim['mention_count']}x",
    ])))

    admission = claim["admission"]
    lines.append(_line("admission", "admitted · " + (
        " · ".join(f"{k}: {v}" for k, v in admission.items()) if admission else "no gate annotation"
    )))
    lines.extend(_importance_lines(claim))

    if not claim["provenance"]:
        lines.append(_line("source", "no evidence linked"))
    elif claim["source"]:
        src = claim["source"]
        where = (f"evidence #{src['evidence_id']}" if src["scope"] == "chunk"
                 else f"the session transcript up to evidence #{src['evidence_id']}")
        session = _session_label(src["session"]) if src["session"] else "session not recorded"
        lines.append(_line("source", f'{src["role"]}: "{src["text"]}"'))
        lines.append(_line("", f"located in {where} · {session} · captured {_when(src['captured_at'])}"))
    else:
        lines.append(_line("source", "not located in its evidence (searched by claim text)"))

    extractors = sorted({p["extractor_version"] or "version not recorded" for p in claim["provenance"]})
    if extractors:
        offsets = ("sentence offsets recorded" if any(p["offsets_recorded"] for p in claim["provenance"])
                   else "sentence offsets not recorded by this extractor")
        lines.append(_line("extraction", f"{', '.join(extractors)} · {offsets}"))
    else:
        lines.append(_line("extraction", "not recorded (no evidence is linked to this claim)"))

    lines.append(_line("commit", claim["commit"][:12] if claim["commit"] else "not recorded"))

    history = claim["history"]
    if len(history) <= 1:
        lines.append(_line("history", "none — nothing replaced it and it replaced nothing"))
    else:
        # A same-session recency chain can run to dozens of narration threads; show
        # the claim's neighbourhood and count the rest (--json keeps every entry).
        here = next((i for i, item in enumerate(history) if item["id"] == claim["id"]), 0)
        start = max(0, here - _HISTORY_BEFORE)
        end = min(len(history), here + _HISTORY_AFTER + 1)
        if start > 0:
            lines.append(_line("history", f"... {start} earlier claim(s) in this chain (see --json)"))
        for index in range(start, end):
            item = history[index]
            marker = "  <- this claim" if item["id"] == claim["id"] else ""
            description = item["description"]
            if len(description) > 70:
                description = description[:67] + "..."
            lines.append(_line("history" if index == 0 else "",
                               f"#{item['id']} {item['event_type']} · {_session_label(item['session'])} · "
                               f"{_when(item['created_at'])} · \"{description}\"{marker}"))
            transition = _transition_text(item)
            if transition:
                lines.append(_line("", f"  {transition}"))
        if end < len(history):
            lines.append(_line("", f"... {len(history) - end} later claim(s) in this chain (see --json)"))
    return "\n".join(lines)


def render_claims(claims: list[dict[str, Any]]) -> str:
    return "\n\n".join(_render_one(claim) for claim in claims)
