"""`cognikernel explain-recall <project> <query>` (S5 T-501).

Why memory retrieved what it did — and why it did not retrieve something else.

Two surfaces share one retrieval core, and both are explained here by calling the
same functions the live paths call:

  recall (MCP tool)   hybrid_candidates -> top k, else the legacy lexical scan
  CK-1 (per prompt)   top 8 of 10 per axis -> echo filter -> this session's
                      render ledger -> dual-evidence gate -> cap

Nothing is written. A claim that is superseded or archived is never a candidate:
retrieval searches only live claims, and the explanation says so rather than
reporting it as a miss.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
from typing import Any

from cognikernel.config import Config

_MAX_DESCRIPTION = 70
_NEAR_MISSES = 4


def _fastembed_installed() -> bool:
    return importlib.util.find_spec("fastembed") is not None


def _dense_status(available: bool) -> dict[str, Any]:
    """Status from the availability snapshot the search itself used: the model can
    finish loading after the search, and must not be reported as having ranked it."""
    from cognikernel.embedding.model import EMBEDDING_MODEL_VERSION

    if not _fastembed_installed():
        return {"status": "not installed", "model": EMBEDDING_MODEL_VERSION}
    return {"status": "available" if available else "not loaded", "model": EMBEDDING_MODEL_VERSION}


def _ck1(conn: sqlite3.Connection, project_id: str, query: str, config: Config,
         session_id: str | None) -> dict[str, Any]:
    from cognikernel.delta.supersede import normalize_for_overlap
    from cognikernel.integration import query as live
    from cognikernel.retrieval.hybrid import hybrid_candidates

    hits = hybrid_candidates(conn, project_id, query, n_per_axis=live.CK1_PER_AXIS)["fused"][:live.CK1_CANDIDATES]
    q_toks = normalize_for_overlap(query)
    echoes = [h["id"] for h in hits if q_toks and live.is_ck1_echo(q_toks, h)]
    remaining = [h for h in hits if h["id"] not in echoes]

    seen: set[int] = set()
    if session_id:
        from cognikernel.storage.render_ledger import rendered_event_ids

        seen = rendered_event_ids(conn, project_id, session_id)
    already_seen = [h["id"] for h in remaining if h["id"] in seen]
    remaining = [h for h in remaining if h["id"] not in seen]

    decisions = live.ck1_gate_decisions(remaining, query, config)
    passed = [h for h, ok, _ in decisions if ok]
    capped = passed[: config.ck1_max_events]
    # The same size-limit loop recall_for_prompt runs before it pushes anything.
    _, injected = live.fit_ck1_budget(capped, config)
    return {
        "limit": live.CK1_CANDIDATES,
        "per_axis": live.CK1_PER_AXIS,
        "candidates": [h["id"] for h in hits],
        "echo_filtered": echoes,
        "session": session_id,
        "already_seen": already_seen,
        "decisions": [
            {"id": h["id"], "passed": ok, "reason": reason,
             "dense_rank": h.get("dense_rank"), "bm25_rank": h.get("bm25_rank")}
            for h, ok, reason in decisions
        ],
        "cap": config.ck1_max_events,
        "over_cap": [h["id"] for h in passed[config.ck1_max_events:]],
        "budget_tokens": config.query_injection_max_tokens,
        "over_budget": [h["id"] for h in capped if h["id"] not in injected],
        "injected": injected,
    }


def _claim_verdict(conn: sqlite3.Connection, project_id: str, claim_id: int, explanation: dict[str, Any],
                   candidates: dict[str, Any]) -> dict[str, Any]:
    row = conn.execute(
        "SELECT id, event_type, payload, superseded_by, archived FROM events WHERE id = ? AND project_id = ?",
        (claim_id, project_id),
    ).fetchone()
    if row is None:
        return {"id": claim_id, "verdict": f"#{claim_id} is not a claim in this project"}
    base = {"id": claim_id, "event_type": row["event_type"],
            "description": json.loads(row["payload"]).get("description", "")}
    if row["superseded_by"] is not None:
        return {**base, "verdict": f"superseded by #{row['superseded_by']}: recall searches only live claims, "
                                   "so it is never a candidate (see `cognikernel why`)"}
    if row["archived"]:
        return {**base, "verdict": "archived: recall searches only live claims, so it is never a candidate"}

    limit, per_axis = explanation["fusion"]["limit"], explanation["fusion"]["per_axis"]
    position = next((c["rank"] for c in explanation["candidates"] if c["id"] == claim_id), None)
    dense_rank = next((i for i, h in enumerate(candidates["dense_hits"], 1) if h["id"] == claim_id), None)
    bm25_rank = next((i for i, h in enumerate(candidates["lexical_hits"], 1) if h["id"] == claim_id), None)

    if position is None and explanation["fallback"]:
        verdict = (f"not in the legacy lexical scan's top {limit} (neither axis returned any "
                   "candidate for this query, so recall used that scan)")
    elif position is None:
        verdict = (f"neither axis surfaced it within its top {per_axis} for this query; "
                   f"try --per-axis {per_axis * 5}, or words the claim itself uses")
    elif position <= limit and explanation["fallback"]:
        score = next(c["score"] for c in explanation["candidates"] if c["id"] == claim_id)
        verdict = f"returned by the legacy lexical scan at rank {position} (Jaccard score {score:.2f})"
    elif position <= limit:
        score = next(c["score"] for c in explanation["candidates"] if c["id"] == claim_id)
        verdict = f"returned by recall at rank {position} (fused score {score:.2f})"
    else:
        verdict = f"a candidate at rank {position}, below recall's cut of {limit}"

    ck1 = explanation["ck1"]
    decision = next((d for d in ck1["decisions"] if d["id"] == claim_id), None)
    if claim_id in ck1["injected"]:
        ck1_verdict = "CK-1 would push it"
    elif claim_id in ck1["echo_filtered"]:
        ck1_verdict = "CK-1 drops it as a restatement of the prompt"
    elif claim_id in ck1["already_seen"]:
        ck1_verdict = "CK-1 skips it: this session has already seen it"
    elif claim_id in ck1["over_cap"]:
        ck1_verdict = f"CK-1 passes it but is over its cap of {ck1['cap']}"
    elif claim_id in ck1["over_budget"]:
        ck1_verdict = f"CK-1 passes it but it does not fit the {ck1['budget_tokens']}-token injection limit"
    elif decision is not None:
        ck1_verdict = f"CK-1 gate rejects it: {decision['reason']}"
    else:
        ck1_verdict = f"not among CK-1's {ck1['limit']} candidates"
    return {**base, "verdict": f"{verdict}; {ck1_verdict}", "fused_rank": position,
            "dense_rank": dense_rank, "bm25_rank": bm25_rank}


def explain_recall(
    conn: sqlite3.Connection,
    project_id: str,
    query: str,
    config: Config,
    *,
    limit: int | None = None,
    per_axis: int | None = None,
    claim_id: int | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """`limit` and `per_axis` default to the live recall sizes in query.py."""
    from cognikernel.integration import query as live
    from cognikernel.retrieval.hybrid import _RRF_K, hybrid_candidates
    from cognikernel.storage.fts import build_match_query

    limit = live.RECALL_LIMIT if limit is None else limit
    per_axis = live.RECALL_PER_AXIS if per_axis is None else per_axis
    candidates = hybrid_candidates(conn, project_id, query, n_per_axis=per_axis)
    fused = candidates["fused"]
    fallback = None
    if not fused:
        # _recall_hits: hybrid returned nothing, so recall uses the Jaccard scan.
        fused = [{**h, "dense_rank": None, "bm25_rank": None, "cosine": None}
                 for h in live._lexical_recall(conn, project_id, query, limit)]
        fallback = "legacy lexical scan"

    explanation: dict[str, Any] = {
        "query": query,
        "terms": build_match_query(query),
        "axes": {
            "lexical": {
                "status": "available" if candidates["lexical_available"]
                else "unavailable (no FTS5 index in this SQLite build)",
                "matches": len(candidates["lexical_hits"]),
            },
            "dense": {**_dense_status(candidates["dense_available"]), "matches": len(candidates["dense_hits"])},
        },
        "fusion": {"method": "reciprocal rank fusion", "k": _RRF_K, "per_axis": per_axis, "limit": limit},
        "fallback": fallback,
        # "rrf": normalized fusion score; "jaccard": the legacy scan's token overlap.
        "score_kind": "jaccard" if fallback else "rrf",
        "candidates": [
            {
                "rank": rank,
                "id": h["id"],
                "event_type": h["event_type"],
                "description": h.get("description", ""),
                "score": h["score"],
                "dense_rank": h.get("dense_rank"),
                "bm25_rank": h.get("bm25_rank"),
                "cosine": h.get("cosine"),
                "in_recall": rank <= limit,
            }
            for rank, h in enumerate(fused, 1)
        ],
        "ck1": _ck1(conn, project_id, query, config, session_id),
        "claim": None,
    }
    if claim_id is not None:
        explanation["claim"] = _claim_verdict(conn, project_id, claim_id, explanation, candidates)
    return explanation


def _cell(value: Any, width: int) -> str:
    return f"{'-' if value is None else value:>{width}}"


def _short(text: str) -> str:
    return text if len(text) <= _MAX_DESCRIPTION else text[:_MAX_DESCRIPTION - 3] + "..."


def render_explain(data: dict[str, Any]) -> str:
    lexical, dense, fusion = data["axes"]["lexical"], data["axes"]["dense"], data["fusion"]
    lines = [
        f'explain-recall "{data["query"]}"',
        f"  terms           {data['terms'] or '(none: every word was too short or a stopword)'}",
        f"  lexical (BM25)  {lexical['status']} · {lexical['matches']} match(es) in its top {fusion['per_axis']}",
        f"  dense           {dense['status']} ({dense['model']}) · {dense['matches']} match(es)",
    ]
    if data["fallback"]:
        lines.append("  fallback        neither axis returned anything, so recall used the legacy lexical scan")
        lines.append(f"  recall          legacy lexical scan (score = Jaccard overlap), top {fusion['limit']} returned")
    else:
        if dense["status"] != "available" and lexical["status"] == "available":
            lines.append("                  without it, recall and CK-1 rank on BM25 alone, as a cold hook does")
        lines.append(f"  recall          {fusion['method']} (K={fusion['k']}, score = normalized fusion), "
                     f"top {fusion['limit']} returned")

    lines += ["", "  rank  claim     recall  score  dense  bm25  cosine  claim text"]
    if not data["candidates"]:
        lines.append("  (no candidates)")
    # Measured on a real store: one query fused 27 candidates. Show recall's top
    # k plus a few near misses, and the asked-about claim wherever it ranked.
    shown_through = fusion["limit"] + _NEAR_MISSES
    claim_id = (data.get("claim") or {}).get("id")
    shown = [c for c in data["candidates"] if c["rank"] <= shown_through or c["id"] == claim_id]
    for c in shown:
        cosine = f"{c['cosine']:.2f}" if isinstance(c["cosine"], (int, float)) else "-"
        lines.append(f"  {c['rank']:>4}  #{c['id']:<7} {'yes' if c['in_recall'] else 'cut':>6}  "
                     f"{c['score']:>5.2f}  {_cell(c['dense_rank'], 5)}  {_cell(c['bm25_rank'], 4)}  "
                     f"{cosine:>6}  {_short(c['description'])}")
    hidden = len(data["candidates"]) - len(shown)
    if hidden:
        lines.append(f"  ... {hidden} more candidate(s) below the cut (see --json)")

    ck1 = data["ck1"]
    lines += ["", f"CK-1 per-prompt push (top {ck1['limit']} candidates, {ck1['per_axis']} per axis)"]
    if not ck1["candidates"]:
        lines.append("  no candidates: nothing would be pushed")
    else:
        if ck1["echo_filtered"]:
            lines.append("  echo filter     drops " + ", ".join(f"#{i}" for i in ck1["echo_filtered"])
                         + " as restatements of the prompt")
        if ck1["session"] is None:
            lines.append("  session ledger  not applied (pass --session to skip what that session already saw)")
        elif ck1["already_seen"]:
            lines.append("  session ledger  skips " + ", ".join(f"#{i}" for i in ck1["already_seen"]))
        for d in ck1["decisions"]:
            lines.append(f"  #{d['id']:<7} {'pass' if d['passed'] else 'fail'}  {d['reason']}")
        if ck1["over_cap"]:
            lines.append(f"  cap {ck1['cap']}           drops " + ", ".join(f"#{i}" for i in ck1["over_cap"]))
        if ck1["over_budget"]:
            lines.append("  size limit      drops " +", ".join(f"#{i}" for i in ck1["over_budget"])
                         + f" (injection limit {ck1['budget_tokens']} tokens)")
        if ck1["injected"]:
            lines.append("  would push      " + ", ".join(f"#{i}" for i in ck1["injected"]))
        else:
            lines.append("  would push      nothing (silence is the default)")

    if data["claim"]:
        claim = data["claim"]
        lines += ["", f"#{claim['id']}: {claim['verdict']}"]
    return "\n".join(lines)
