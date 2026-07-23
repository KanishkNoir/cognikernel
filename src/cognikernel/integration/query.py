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
import re
import sqlite3
from pathlib import Path

from cognikernel.config import Config
from cognikernel.storage.connection import get_connection, get_db_path, resolve_project_id
from cognikernel.storage.migrations import run_migrations

_MAX_DESC = 140

# K2 surfacing precision: a graveyard entry is always a prohibition, but a
# CONSTRAINT_HARD is only a *prohibition* (something an edit could re-violate)
# when it carries an explicit negative/abandonment marker. Positive hard rules
# ("money is integer cents", "TargetConfig dataclass") are mandatory facts
# already in the block, not action-point bind targets — including them drove the
# surface rate to ~83% in calibration. Graveyard ∪ hard(negative) → ~0-27%.
_PROHIBITION_MARKER_RE = re.compile(
    r"\b(never|don't|do not|cannot|can't|must not|shall not|avoid|"
    r"instead of|rather than|deprecat\w*|forbid\w*|prohibit\w*|"
    r"stop using|migrate away|ruled out|abandon\w*|no longer|not use)\b"
)

# #56 selection: authority + scope ranking so the right prohibition surfaces.
_AUTH_RANK = {
    "user_stated": 3.0, "assistant_decided": 2.0,
    "assistant_answer_to_user_question": 1.0, "inferred_from_code": 0.0,
}
# A prohibition about WHERE state lives (multi-instance / shared / distributed)
# is a higher-value bind target than an implementation-mechanic one — the D5/D16
# lesson: BM25 buried "must be shared / local counters don't work" under a
# token-dense "RPM check sits outside the attempt loop".
_ARCH_MARKER_RE = re.compile(
    r"\b(multi-?instance|instances?|shared|distributed|redis|process-local|"
    r"in-process|\blocal\b|across|cluster|workers?|global state|single.?point|"
    r"per-process|per-request)\b", re.I
)


def _prohibition_priority(h: dict) -> float:
    """Rank a prohibition for action-point surfacing: authority + scope boosts.
    Graveyard ('do not retry') and architecture-scope prohibitions rank above
    implementation-detail ones so high term-overlap can't bury them."""
    pri = _AUTH_RANK.get(h.get("authority", ""), 1.0)
    if h.get("event_type") == "APPROACH_ABANDONED_DO_NOT_RETRY":
        pri += 0.5
    if _ARCH_MARKER_RE.search(f"{h.get('subject', '')} {h.get('description', '')}"):
        pri += 1.0
    return pri


# K2 floor fix #1 (probe replay 2026-07): fold inflectional suffixes when counting
# shared terms, LOCALLY — never in normalize_for_overlap itself, which would
# silently shift every calibrated Jaccard threshold in supersession/CK-1/echo.
# The replay's Toolbelt miss was one morpheme wide: a diff saying "re-implement"
# shares no token with a prohibition saying "never re-implemented".
_STEM_FOLDS = (("ations", "ate"), ("ation", "ate"), ("ments", "ment"),
               ("ings", "ing"), ("ing", ""), ("edly", ""), ("ies", "y"),
               ("ed", ""), ("es", ""), ("s", ""))


def _fold_stems(tokens: set[str]) -> set[str]:
    out = set()
    for t in tokens:
        if len(t) > 4:
            for suf, rep in _STEM_FOLDS:
                if t.endswith(suf) and len(t) - len(suf) + len(rep) >= 4:
                    t = t[: len(t) - len(suf)] + rep
                    break
        out.add(t)
    return out


# K2 floor fix #2 (probe replay 2026-07): dense rescue for THIN prohibitions.
# The Relay graveyard entry ("In-process counters are out — each instance sees
# only its slice of traffic") is on-topic for an in-process-counters diff but its
# stored text is so short it lands at 2 shared terms, under the floor. When the
# embedding model is resident, a candidate that fails the lexical floor is
# rescued iff cosine agrees strongly AND at least one shared content term anchors
# it (mirrors CK-1's dense-only arm: absolute floors, never rank-free semantics).
# Advisory-only surface + typed/marker-filtered pool keep this precision-safe.
_K2_DENSE_RESCUE_COS = 0.60
_K2_RESCUE_MIN_ANCHOR = 1


def _dense_rescue_scores(conn, diff_text: str, candidates: list[dict]) -> dict[int, float]:
    """cosine(diff, candidate) for each candidate id — {} when the model is cold.

    Stored event vectors are used when present; missing ones are embedded on the
    fly (pool is bounded by pretool_pool_size). Fail-open: any error -> {}.
    """
    try:
        from cognikernel.embedding.input import embedding_input
        from cognikernel.embedding.model import EMBEDDING_MODEL_VERSION, embed_text, is_ready
        from cognikernel.embedding.store import load_embeddings

        if not is_ready():
            return {}
        import numpy as np

        dv = embed_text(diff_text)
        if dv is None:
            return {}
        dv = np.asarray(dv, dtype="float32")
        stored = load_embeddings(conn, [h["id"] for h in candidates], EMBEDDING_MODEL_VERSION)
        scores: dict[int, float] = {}
        for h in candidates:
            vec = stored.get(h["id"])
            if vec is None:
                vec = embed_text(embedding_input(
                    {"description": h.get("description", ""), "subject": h.get("subject", "")},
                    h.get("event_type", "DECISION")))
            if vec is not None:
                scores[h["id"]] = float(dv @ np.asarray(vec, dtype="float32"))
        return scores
    except Exception:
        return {}


def _resolve(project_path: str, config: Config | None) -> tuple[str, Path]:
    config = config or Config.load(project_path=project_path)
    project_id = resolve_project_id(project_path, config)
    return project_id, get_db_path(config, project_id)


def _lexical_recall(
    conn: sqlite3.Connection, project_id: str, query: str, limit: int
) -> list[dict]:
    """Deterministic token-overlap fallback used when embeddings are unavailable.

    Jaccard over normalized description tokens — same normalization as supersession,
    so behavior is consistent with the lexical merge path.
    """
    from cognikernel.delta.supersede import normalize_for_overlap

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
    """Hybrid retrieval (BM25 ∪ dense → RRF) with a deterministic last resort.

    `hybrid_recall` handles the degradation ladder internally (both axes → RRF;
    one axis → that axis alone; model warm-up is kicked but never awaited). It
    returns [] only when NEITHER axis is available — no FTS5 in this SQLite
    build and no loaded embedding model — in which case the legacy Jaccard scan
    keeps recall correct on a minimal install.
    """
    from cognikernel.retrieval.hybrid import hybrid_recall

    hits = hybrid_recall(conn, project_id, query, k=k)
    if hits:
        return hits
    return _lexical_recall(conn, project_id, query, k)


def _fmt_hit(h: dict, extra: str = "") -> str:
    subj = f"[{h['subject']}] " if h.get("subject") else ""
    desc = (h.get("description") or "")[:_MAX_DESC]
    return f"- ({h['event_type']} · {h['score']:.2f}{extra}) {subj}{desc}"


def select_ck1_hits(
    hits: list[dict],
    prompt_text: str,
    config: Config,
    seen_ids: set[int] | None = None,
) -> list[dict]:
    """The full CK-1 selection pipeline as a pure function (harness-sweepable):
    self-echo filter → ledger redundancy filter → dual-evidence gate → cap."""
    from cognikernel.delta.supersede import normalize_for_overlap

    q_toks = normalize_for_overlap(prompt_text)
    if q_toks:
        def _echo(h: dict) -> bool:
            d_toks = normalize_for_overlap(h.get("description", ""))
            if not d_toks:
                return False
            inter = len(q_toks & d_toks)
            return inter / len(q_toks | d_toks) >= 0.6 or inter / len(d_toks) >= 0.8

        hits = [h for h in hits if not _echo(h)]
    if seen_ids:
        hits = [h for h in hits if h["id"] not in seen_ids]
    return _ck1_dual_evidence(hits, prompt_text, config)


def _ck1_dual_evidence(
    hits: list[dict],
    prompt_text: str,
    config: Config,
) -> list[dict]:
    """Rank-based dual-evidence gate (J4.2). Precision-first; silence default.

    Axis availability is inferred from the result set as a whole — a hit
    missing one rank when both axes ran means that axis did NOT independently
    surface it (insufficient evidence), which is different from the axis being
    down. Modes:
      both axes:  dense_rank ≤ N AND bm25_rank ≤ N   (independent agreement —
                  the rank-space analogue of the xenc∧jaccard supersession hybrid)
      BM25 only:  bm25_rank ≤ 2 AND ≥ ck1_min_term_overlap shared content terms
                  (an ABSOLUTE floor — ratio scores are why 0.625 was the old
                  cold-path ceiling)
      dense only: dense_rank ≤ 2 AND cosine ≥ 0.60 (self-comparable within one axis)
    """
    from cognikernel.delta.supersede import normalize_for_overlap

    dense_live = any(h.get("dense_rank") is not None for h in hits)
    bm25_live = any(h.get("bm25_rank") is not None for h in hits)
    q_toks = normalize_for_overlap(prompt_text)
    passed: list[dict] = []
    for h in hits:
        d, b = h.get("dense_rank"), h.get("bm25_rank")
        if dense_live and bm25_live:
            ok = (
                d is not None and b is not None
                and d <= config.ck1_dense_rank_max
                and b <= config.ck1_bm25_rank_max
            )
            # Lexical anchor even in dual mode: rank agreement alone is too
            # permissive on short/generic prompts in a small store (measured:
            # "yep i would like to replace the constraint" passed ≤5∧≤5).
            if ok and config.ck1_dual_anchor_terms:
                shared = q_toks & normalize_for_overlap(h.get("description", ""))
                ok = len(shared) >= config.ck1_dual_anchor_terms
        elif bm25_live:
            if b is None or b > 2:
                ok = False
            else:
                shared = q_toks & normalize_for_overlap(h.get("description", ""))
                ok = len(shared) >= config.ck1_min_term_overlap
        elif dense_live:
            ok = d is not None and d <= 2 and (h.get("cosine") or 0.0) >= 0.60
        else:
            ok = False
        if ok:
            passed.append(h)
    return passed[: config.ck1_max_events]


def recall_for_prompt(
    project_path: str,
    prompt_text: str,
    *,
    config: Config | None = None,
    session_id: str | None = None,
) -> str:
    """Per-prompt injection candidate for the UserPromptSubmit hook (CK-1).

    Returns a compact snippet (≤ config.query_injection_max_tokens tokens-ish)
    when a relevant, non-redundant prior fact clears the dual-evidence gate.
    Returns '' (silence) otherwise — silence is the default, not the fallback.

    Redundancy = the render ledger: anything this session already saw (block
    or earlier ck1 push) is skipped. This REPLACES the old type-based filter,
    which was empirically false — the block carries only a handful of the
    active hard constraints, and the types it excluded were exactly the ones
    most worth pushing.
    """
    try:
        config = config or Config.load(project_path=project_path)
        project_id, db_path = _resolve(project_path, config)
        if not db_path.exists():
            return ""
        max_chars = config.query_injection_max_tokens * 4  # tok → approx chars
        with get_connection(db_path) as conn:
            run_migrations(conn)
            from cognikernel.retrieval.hybrid import hybrid_recall

            hits = hybrid_recall(conn, project_id, prompt_text, k=8, n_per_axis=10)
            if not hits:
                return ""
            seen: set[int] = set()
            if session_id:
                from cognikernel.storage.render_ledger import rendered_event_ids

                seen = rendered_event_ids(conn, project_id, session_id)
            passed = select_ck1_hits(hits, prompt_text, config, seen)
            if not passed:
                return ""
            lines = ["[CogniKernel — relevant prior context]"]
            used = len(lines[0])
            injected: list[int] = []
            for h in passed:
                line = _fmt_hit(h)
                if used + len(line) + 1 > max_chars:
                    break
                lines.append(line)
                used += len(line) + 1
                injected.append(h["id"])
            if len(lines) <= 1:
                return ""
            if session_id and injected:
                from cognikernel.storage.render_ledger import record_rendered

                record_rendered(conn, project_id, session_id, injected, "ck1")
        return "\n".join(lines)
    except Exception:  # never raise; silence on any error
        return ""


def surface_prohibitions_for_edit(
    project_path: str,
    diff_text: str,
    *,
    file_path: str = "",
    config: Config | None = None,
    session_id: str | None = None,
) -> str:
    """K2 — PreToolUse JIT bind: surface a prohibition the edit would violate.

    The action-point analogue of `recall_for_prompt`. Given the new code a
    Write/Edit is about to write (`diff_text`), look up matching prohibitions in
    the type-restricted lexical pool (graveyard + hard constraints) and return a
    short advisory if one clears the gate. Returns '' (silence) otherwise —
    silence is the default. NEVER blocks; the caller surfaces this as
    `additionalContext` on an `allow` decision.

    Gate mirrors CK-1's BM25-only arm (precision-first): a prohibition must rank
    <= pretool_bm25_rank_max AND share >= pretool_min_term_overlap absolute
    content terms with the diff (over its subject+description). Already-surfaced
    events (any channel this session) are filtered via the render ledger.
    """
    try:
        config = config or Config.load(project_path=project_path)
        if not config.pretool_prohibition_surface_enabled:
            return ""
        if not diff_text or not diff_text.strip():
            return ""
        project_id, db_path = _resolve(project_path, config)
        if not db_path.exists():
            return ""

        from cognikernel.delta.supersede import normalize_for_overlap
        from cognikernel.storage.fts import prohibition_search

        diff_toks = normalize_for_overlap(diff_text)
        if not diff_toks:
            return ""
        with get_connection(db_path) as conn:
            run_migrations(conn)
            # #56: pull a BROAD pool, then re-rank by (authority, overlap) — NOT
            # by raw BM25 rank. The live Relay run proved rank≤1/BM25 surfaced a
            # token-dense impl-detail prohibition and buried the architecture one
            # (D5/D16) at BM25 ranks 6-9. Selection, not retrieval, was the gap.
            hits = prohibition_search(conn, project_id, diff_text, n=config.pretool_pool_size)
            if not hits:
                return ""
            seen: set[int] = set()
            if session_id:
                from cognikernel.storage.render_ledger import rendered_event_ids

                seen = rendered_event_ids(conn, project_id, session_id)
            eligible: list[tuple[dict, int]] = []
            for h in hits:
                if h["id"] in seen:
                    continue
                # A hard constraint is only a *prohibition* an edit can re-violate
                # when it carries a negative/abandonment marker; positive hard
                # rules are block facts, not bind targets. Graveyard always qualifies.
                if h.get("event_type") == "CONSTRAINT_HARD" and not (
                    _PROHIBITION_MARKER_RE.search((h.get("description") or "").lower())
                ):
                    continue
                p_toks = normalize_for_overlap(
                    f"{h.get('subject', '')} {h.get('description', '')}"
                )
                # Stem-folded overlap so one morpheme ("re-implement" vs
                # "re-implemented") can't hide a genuine bind.
                ov = len(_fold_stems(diff_toks) & _fold_stems(p_toks))
                eligible.append((h, ov))

            # Two tiers, floor first: a dense-rescued candidate is weaker evidence
            # than one clearing the lexical floor and must NEVER outrank it — the
            # first replay iteration let a rescued, arch-marker-boosted candidate
            # steal the cap-1 slot from the canonical prohibition.
            cand: list[tuple[float, int, dict]] = []
            rescued: list[tuple[float, int, dict]] = []
            below_floor = [(h, ov) for h, ov in eligible
                           if _K2_RESCUE_MIN_ANCHOR <= ov < config.pretool_min_term_overlap]
            rescue_scores = (
                _dense_rescue_scores(conn, diff_text, [h for h, _ in below_floor])
                if below_floor else {}
            )
            for h, ov in eligible:
                if ov >= config.pretool_min_term_overlap:
                    cand.append((_prohibition_priority(h), ov, h))
                elif (ov >= _K2_RESCUE_MIN_ANCHOR
                      and rescue_scores.get(h["id"], 0.0) >= _K2_DENSE_RESCUE_COS):
                    rescued.append((_prohibition_priority(h), ov, h))
            # Highest authority first, then strongest term overlap. A high-
            # authority architecture prohibition now beats a token-dense
            # impl-detail one even when BM25 ranked the latter first. Rescued
            # candidates trail the whole floor tier regardless of priority.
            cand.sort(key=lambda t: (t[0], t[1]), reverse=True)
            rescued.sort(key=lambda t: (t[0], t[1]), reverse=True)
            cand = cand + rescued
            passed: list[dict] = []
            seen_topics: set[str] = set()
            for _pri, _ov, h in cand:
                topic = h.get("decision_key") or " ".join(
                    sorted(normalize_for_overlap(
                        h.get("subject") or h.get("description") or ""))[:3])
                if topic and topic in seen_topics:
                    continue
                seen_topics.add(topic)
                passed.append(h)
                if len(passed) >= config.pretool_max_surface:
                    break
            if not passed:
                return ""
            lines = ["[CogniKernel — you previously ruled this out]"]
            for h in passed:
                approach = (h.get("description") or "")[:_MAX_DESC]
                rationale = (h.get("rationale") or "").strip()
                tail = f" — {rationale[:_MAX_DESC]}" if rationale else ""
                lines.append(f"- {approach}{tail}")
            lines.append(
                "Re-affirm in your message if this change is intentional; "
                "otherwise honor the prior decision."
            )
            if session_id:
                from cognikernel.storage.render_ledger import record_rendered

                record_rendered(
                    conn, project_id, session_id, [h["id"] for h in passed], "pretool"
                )
        return "\n".join(lines)
    except Exception:  # never raise into the hook; silence on any error
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
            from cognikernel.embedding.retrieval import find_related
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
