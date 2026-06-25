"""Constraint supersession — keyword-overlap detection without embeddings.

Two complementary metrics:
  Jaccard similarity   — catches different word choice for the same concept
  Levenshtein (norm.)  — catches slight phrasing variations of the same sentence

OR rule: either metric triggering marks events as overlapping.
"""
from __future__ import annotations

import json
import re
import sqlite3
from typing import TYPE_CHECKING

# Subject-normalization primitives live in utils.subject (a dependency-free base)
# so extraction.decision_key can reuse them without importing delta — breaking the
# extraction<->delta layering cycle. Re-exported here for backward compatibility
# (delta.__init__ exports derive_subject; storage.fts imports STOPWORDS).
from memlora.utils.subject import STOPWORDS, derive_subject  # noqa: F401  (re-export)

if TYPE_CHECKING:
    from memlora.storage.events import Event

_SUPERSESSION_TYPES: frozenset[str] = frozenset({
    "CONSTRAINT_HARD",
    "CONSTRAINT_SOFT",
    "DECISION",
    "APPROACH_ABANDONED",
    "APPROACH_ABANDONED_DO_NOT_RETRY",
})

JACCARD_THRESHOLD: float = 0.6
LEVENSHTEIN_THRESHOLD: float = 0.15


def normalize_for_overlap(text: str) -> set[str]:
    """Lowercase, strip punctuation, remove stopwords and tokens ≤ 2 chars."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return {t for t in text.split() if len(t) > 2 and t not in STOPWORDS}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Jaccard on normalised token sets: |A ∩ B| / |A ∪ B|."""
    a = normalize_for_overlap(text_a)
    b = normalize_for_overlap(text_b)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def levenshtein_normalized(a: str, b: str) -> float:
    """Normalised edit distance in [0.0 (identical) … 1.0 (totally different)]."""
    a = a.lower().strip()
    b = b.lower().strip()
    if a == b:
        return 0.0
    if not a or not b:
        return 1.0
    m, n = len(a), len(b)
    if m < n:
        a, b, m, n = b, a, n, m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (a[i - 1] != b[j - 1]),
            )
        prev = curr
    return prev[n] / m


def descriptions_overlap(desc_a: str, desc_b: str) -> bool:
    """Return True if two descriptions express the same concept.

    Exact semantics: ``jaccard >= JACCARD_THRESHOLD OR levenshtein <= LEVENSHTEIN_THRESHOLD``.

    De-quadratic pruning (no behavior change): Jaccard (O(tokens)) is checked
    first and short-circuits. If it fails, a length bound rules out Levenshtein
    cheaply — normalized edit distance is at least ``|len_a − len_b| / max(len)``,
    so when that lower bound already exceeds the threshold the O(m·n) Levenshtein
    computation is skipped entirely. This is exact: it never changes the result,
    only avoids work on length-disparate (hence non-matching) pairs, which is the
    common case during a merge scan.
    """
    if jaccard_similarity(desc_a, desc_b) >= JACCARD_THRESHOLD:
        return True
    la, lb = len(desc_a.strip()), len(desc_b.strip())
    if la and lb and abs(la - lb) / max(la, lb) > LEVENSHTEIN_THRESHOLD:
        return False  # Levenshtein lower bound already exceeds the threshold
    return levenshtein_normalized(desc_a, desc_b) <= LEVENSHTEIN_THRESHOLD


# ── subject-keyed supersession (revived from a6b5d15) ────────────────────────
#
# `descriptions_overlap` (Jaccard + Levenshtein) misses the canonical case where
# a decision's *choice* changes but its *topic* stays the same — the choice verbs
# and tool names differ enough that token overlap falls below threshold. Extracting
# the *topic* the decision is about (independent of the specific choice) provides
# a third discriminating axis.
#
# Gated by SUBJECT_MATCH_MIN_JACCARD (0.3) so a shared *generic* topic alone
# cannot force a supersession — the two descriptions must also share enough tokens
# to indicate genuine relatedness. Combined predicate: `supersedes` = lexical OR
# same-subject-gated. Additive only: can add supersessions, never remove.

SUBJECT_MATCH_MIN_JACCARD: float = 0.3

# derive_subject (and its regex/stopword helpers) moved to utils.subject — see the
# re-export import at the top of this module.


def subject_supersedes(desc_a: str, desc_b: str) -> bool:
    """True when two descriptions share a derived subject and are related enough.

    Requires both descriptions to derive the same non-empty topic AND share at
    least SUBJECT_MATCH_MIN_JACCARD token overlap — so a shared *generic* topic
    alone (e.g. 'the API') cannot force an unrelated pair to supersede.
    """
    sa = derive_subject(desc_a)
    if not sa or sa != derive_subject(desc_b):
        return False
    return jaccard_similarity(desc_a, desc_b) >= SUBJECT_MATCH_MIN_JACCARD


def supersedes(desc_a: str, desc_b: str) -> bool:
    """Combined supersession predicate: textual overlap OR same-subject (gated)."""
    return descriptions_overlap(desc_a, desc_b) or subject_supersedes(desc_a, desc_b)


def events_overlap(event_a: Event, event_b: Event) -> bool:
    """Return True if two events express the same concept in different words."""
    if event_a.event_type != event_b.event_type:
        return False
    desc_a = event_a.payload.get("description", "")
    desc_b = event_b.payload.get("description", "")
    return supersedes(desc_a, desc_b)


def detect_supersession(
    conn: sqlite3.Connection,
    new_event: Event,
) -> list[int]:
    """Return IDs of existing events superseded by new_event.

    Queries same-type, non-superseded events then applies overlap detection.
    Returns empty list for event types that do not support supersession.
    """
    if new_event.event_type not in _SUPERSESSION_TYPES:
        return []

    rows = conn.execute(
        """
        SELECT id, payload FROM events
        WHERE project_id    = ?
          AND event_type    = ?
          AND archived      = 0
          AND superseded_by IS NULL
          AND content_hash != ?
        """,
        (new_event.project_id, new_event.event_type, new_event.content_hash),
    ).fetchall()

    new_desc = new_event.payload.get("description", "")
    superseded_ids: list[int] = []
    for row in rows:
        cand_desc = json.loads(row["payload"]).get("description", "")
        if supersedes(new_desc, cand_desc):
            superseded_ids.append(row["id"])

    return superseded_ids


def apply_supersession(
    conn: sqlite3.Connection,
    new_event_id: int,
    superseded_ids: list[int],
) -> int:
    """Mark each superseded event as replaced by new_event_id. Returns count updated."""
    for old_id in superseded_ids:
        conn.execute(
            "UPDATE events SET superseded_by = ? WHERE id = ?",
            (new_event_id, old_id),
        )
    return len(superseded_ids)


# ── gated supersession: temporal + authority + provenance, with optional semantic ─
#
# `find_superseded` is the single supersession entry point for the merge. The
# three structured gates are the ALWAYS-ON baseline — they apply whether or not
# embeddings are enabled, because they are correctness properties independent of
# the retrieval mechanism (decoupled from config.embedding_enabled by design):
#   - temporal direction: a new event only supersedes an OLDER one (created_at);
#   - authority precedence: a lower-trust event never supersedes a higher-trust
#     one (e.g. inferred-from-code must not overwrite a user-stated decision);
#   - provenance: a match within the SAME transcript (evidence_id) is a
#     restatement, not an evolution, so it is never superseded.
#
# On top of that gated floor, the *candidate* set is found by lexical overlap
# (descriptions_overlap) OR — when `use_embeddings` is True — a semantic cosine
# axis that also catches paraphrased corrections lexical overlap misses — a
# decision restated later in different, lexically-distinct words. The semantic
# axis is purely additive: with embeddings off (or the model absent) matching
# degrades to gated-lexical, never to ungated.

# SAFETY REVERT to 0.75. The CK-E6 sweep picked 0.65 on a generic labeled set, but
# re-validation on REAL same-project data (scripts/_mob_d9_revalidate.py) found it
# UNSAFE: the genuine correction target (#66, cosine 0.658) and unrelated decisions
# in the same project (#6 "composite PK" 0.654, #10 "UUID PK" 0.633) are
# NON-SEPARABLE by cosine — any threshold catching the real correction also
# false-supersedes unrelated decisions (deleting a still-valid decision: the
# precision failure we bias against). The eval missed this because its negatives
# were cross-domain (too easy); real decisions share domain vocabulary and cluster
# at 0.63-0.66. bge-small over BARE descriptions lacks the discriminator. 0.75 is
# the bleed-stop (no observed FP) but catches almost nothing semantic — a real fix
# needs STRUCTURE (subject-keyed candidates and/or required provenance/authority
# co-fire), not a threshold. Decision pending (see the integration follow-up).
SUPERSESSION_COSINE_THRESHOLD: float = 0.75

# Higher = more authoritative. A new event must be >= a candidate's precedence
# to supersede it. Mirrors extraction.authority string constants.
_AUTHORITY_PRECEDENCE: dict[str, int] = {
    "user_stated": 3,
    "assistant_decided": 2,
    "llm": 2,
    "assistant_answer_to_user_question": 1,
    "inferred_from_code": 0,
}
_AUTHORITY_DEFAULT = 2

# F1: types whose *choice* can evolve into a DIFFERENT type on the same subject —
# e.g. a soft constraint ("passwords hashed with bcrypt") is later restated as a
# decision ("switch to argon2id for password hashing"). Cross-type supersession is
# allowed ONLY within this family and ONLY when the derived subject matches (the
# stricter subject+Jaccard gate, never lexical-overlap-only or semantic-only), so
# a decision and a constraint on the same topic must be a genuine restatement of
# the same choice. The graveyard family (APPROACH_ABANDONED*) is excluded — that
# evolution is handled by merge._cross_type_dedup, not subject keying.
_CHOICE_FAMILY: frozenset[str] = frozenset({
    "CONSTRAINT_HARD",
    "CONSTRAINT_SOFT",
    "DECISION",
})


# R5: cap on cross-encoder forward passes per supersession check, to bound the
# Stop-hook merge cost. The cross-encoder reranks the top-K cosine candidates only.
_XENC_MAX_CANDIDATES: int = 80

# R5 HYBRID (precision-first). The cross-encoder ALONE over-supersedes on real stores —
# it scores ~0.97 for almost any SAME-TOPIC pair, conflating topical relatedness with
# supersession (validated: a default-alias change "superseded" 48 unrelated decisions).
# No threshold is safe alone. Requiring a LEXICAL co-fire (Jaccard floor, below
# lexical_supersedes' own 0.6) filters those topical false positives — they share the
# topic but not the changed VALUE's tokens — while still catching paraphrases that
# overlap moderately. Validated: real-store over-supersession 48/14 -> 1/0, gold P100
# guard_fp 0, recall a modest +8pts over lexical-only. The cross-encoder is a precision-
# gated RERANKER here, never a standalone recall expander.
_XENC_SUPERSEDE_MIN: float = 0.97   # high precision bar for the cross-encoder score
_XENC_JAC_MIN: float = 0.3          # required lexical co-fire


def find_superseded(
    conn: sqlite3.Connection,
    new_event: Event,
    *,
    use_embeddings: bool = True,
    use_cross_encoder: bool = False,
) -> list[int]:
    """Gated supersession finder (temporal + authority + provenance + lexical/semantic).

    Returns ids of active, same-type events that `new_event` supersedes. The
    three structured gates always apply. `use_embeddings` toggles only the
    semantic candidate axis: when False, no embedding model is loaded and
    matching is gated-lexical (the safe baseline for config.embedding_enabled =
    False); when True (and the model is available), cosine retrieval contributes
    additional candidates on top. The new event is assumed to be the most recent
    assertion.
    """
    if new_event.event_type not in _SUPERSESSION_TYPES:
        return []

    # Candidate types: always the same type; plus, when the new event is in the
    # choice family, the other choice-family types so a decision can supersede a
    # same-subject constraint and vice-versa (F1). Cross-type matches are held to
    # the stricter subject gate below.
    cand_types = {new_event.event_type}
    if new_event.event_type in _CHOICE_FAMILY:
        cand_types |= _CHOICE_FAMILY
    type_placeholders = ",".join("?" * len(cand_types))

    rows = conn.execute(
        f"""
        SELECT id, payload, created_at, evidence_id, event_type FROM events
        WHERE project_id    = ?
          AND event_type    IN ({type_placeholders})
          AND archived      = 0
          AND superseded_by IS NULL
          AND content_hash != ?
        """,
        (new_event.project_id, *sorted(cand_types), new_event.content_hash),
    ).fetchall()
    if not rows:
        return []

    new_desc = new_event.payload.get("description", "")
    new_auth = _AUTHORITY_PRECEDENCE.get(
        new_event.payload.get("authority", ""), _AUTHORITY_DEFAULT
    )
    new_created = new_event.created_at
    new_evidence = new_event.evidence_id

    # Candidate cosine scores (shared by the semantic + cross-encoder axes). Embeddings
    # are stored regardless of config, so we score once and reuse. Computed only when an
    # embedding-backed axis is on; empty when the model/vectors are unavailable.
    cand_cos: dict[int, float] = {}
    if use_embeddings or use_cross_encoder:
        try:
            import numpy as _np

            from memlora.embedding.input import embedding_input
            from memlora.embedding.model import EMBEDDING_MODEL_VERSION, embed_text
            from memlora.embedding.store import load_embeddings

            query_vec = embed_text(embedding_input(new_event.payload, new_event.event_type))
            if query_vec is not None:
                qv = _np.asarray(query_vec, dtype="float32")
                cand_emb = load_embeddings(conn, [r["id"] for r in rows], EMBEDDING_MODEL_VERSION)
                cand_cos = {cid: float(qv @ _np.asarray(v, dtype="float32")) for cid, v in cand_emb.items()}
        except Exception:
            cand_cos = {}

    # Semantic axis (optional, additive): candidates above the safe cosine threshold.
    sem_matches: set[int] = (
        {cid for cid, c in cand_cos.items() if c >= SUPERSESSION_COSINE_THRESHOLD}
        if use_embeddings else set()
    )

    # Cross-encoder axis (R5, optional + additive): a learned PAIRWISE scorer that catches
    # paraphrased corrections lexical+cosine miss WITHOUT the same-area false positives a
    # bare cosine causes (the F5 fix). Bi-encoder RETRIEVE -> cross-encoder RERANK: score
    # only the top-_XENC_MAX_CANDIDATES same-type candidates by cosine, so cost is bounded
    # regardless of store size. Additive above the always-on gates; fail-open (no model ->
    # empty). Precision-calibrated threshold ships with the model. Same-type only.
    xenc_matches: set[int] = set()
    if use_cross_encoder:
        try:
            from memlora.delta import supersede_xenc

            if supersede_xenc.is_available():
                by_id = {r["id"]: r for r in rows if r["event_type"] == new_event.event_type}
                ranked = sorted(
                    (cid for cid in by_id if cid in cand_cos),
                    key=lambda c: cand_cos[c], reverse=True,
                )
                shortlist = ranked[:_XENC_MAX_CANDIDATES] or list(by_id)[:_XENC_MAX_CANDIDATES]
                for cid in shortlist:
                    cd = json.loads(by_id[cid]["payload"]).get("description", "")
                    p = supersede_xenc.prob_supersedes(new_desc, cd)
                    # HYBRID: high cross-encoder score AND a lexical co-fire (filters the
                    # topical false positives the cross-encoder produces alone).
                    if (p is not None and p >= _XENC_SUPERSEDE_MIN
                            and jaccard_similarity(new_desc, cd) >= _XENC_JAC_MIN):
                        xenc_matches.add(cid)
        except Exception:
            xenc_matches = set()

    superseded: list[int] = []
    for row in rows:
        # Provenance gate (E2): only supersede across a DIFFERENT transcript. A
        # match within the same evidence is a duplicate capture of one statement
        # (restatement), not a later evolution — so it must not be superseded.
        if new_evidence is not None and row["evidence_id"] == new_evidence:
            continue

        cand_payload = json.loads(row["payload"])
        cand_desc = cand_payload.get("description", "")

        # Temporal gate: only supersede an event that is not newer than this one.
        c_created = row["created_at"]
        if c_created is not None and new_created is not None and c_created > new_created:
            continue

        # Authority gate: a less-authoritative event must not supersede a more-
        # authoritative one.
        cand_auth = _AUTHORITY_PRECEDENCE.get(
            cand_payload.get("authority", ""), _AUTHORITY_DEFAULT
        )
        if new_auth < cand_auth:
            continue

        if row["event_type"] == new_event.event_type:
            # Same type: full predicate — cross-encoder OR semantic OR lexical OR subject.
            if (row["id"] in sem_matches or row["id"] in xenc_matches
                    or supersedes(new_desc, cand_desc)):
                superseded.append(row["id"])
        else:
            # Cross type (F1): require the stronger structural signal — same derived
            # subject + Jaccard floor. Lexical-overlap-only / semantic-only is NOT
            # enough across types, to protect precision.
            if subject_supersedes(new_desc, cand_desc):
                superseded.append(row["id"])

    return superseded
