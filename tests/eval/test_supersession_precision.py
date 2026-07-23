"""CK-E6 — supersession precision/recall eval + threshold sweep.

Validates the semantic supersession decision (`delta.supersede.find_superseded`
with embeddings ON) against a labeled set spanning the outcome classes. A false
supersession silently hides a still-valid decision, so we bias to **precision**.

  - pytest gate: `test_precision_at_current_threshold` asserts precision at the
    shipped SUPERSESSION_COSINE_THRESHOLD meets target on this set (model-guarded).
  - sweep report: `python tests/eval/test_supersession_precision.py` prints
    precision/recall/confusion per threshold + the false positives, to pick the value.

Cases are generic/domain-neutral on purpose (no benchmark-specific decisions), and
descriptions are deliberately bare (no `subject`) to measure the realistic case
where trie-extracted events lack structured subjects.

LIMITATION (found post-hoc, do not trust the band without fixing this): the
negatives below are cross-DOMAIN (caching vs input validation, etc.) and are easy
to separate. Re-validation on real same-PROJECT data
(scripts/_mob_d9_revalidate.py) showed unrelated decisions within one project share
domain vocabulary and cluster with genuine corrections at ~0.63-0.66 — NON-separable
by cosine. So a clean precision/recall band HERE does not imply real-world
precision. This eval must be hardened with same-domain, distinct-decision negatives
before any threshold recommendation from it is trusted. See the
SUPERSESSION_COSINE_THRESHOLD comment in delta/supersede.py.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import pytest

from cognikernel.delta import supersede
from cognikernel.delta.supersede import find_superseded
from cognikernel.embedding.model import is_available
from cognikernel.storage.events import Event
from cognikernel.storage.migrations import run_migrations

PRECISION_TARGET = 0.9  # a false supersession deletes a valid decision — bias here.


@dataclass
class Case:
    cls: str
    should_supersede: bool
    old_desc: str
    new_desc: str
    event_type: str = "DECISION"
    old_authority: str = "assistant_decided"
    new_authority: str = "assistant_decided"
    same_evidence: bool = False     # True → same transcript
    new_is_newer: bool = True       # False → the existing (old) event is the newer one


CASES: list[Case] = [
    # ── should-supersede: genuine cross-session corrections, lexically distinct ──
    Case("paraphrase", True, "Store user sessions in an in-memory cache",
         "Move session storage to a Redis-backed store"),
    Case("paraphrase", True, "Serialize wire payloads as JSON",
         "Adopt Protocol Buffers for service-to-service payloads"),
    Case("paraphrase", True, "Generate record identifiers as auto-increment integers",
         "Switch primary identifiers to random UUIDs"),
    Case("paraphrase", True, "Write application logs to local files",
         "Ship logs to a centralized aggregation service"),
    Case("paraphrase", True, "Deploy the service as a single monolith",
         "Split the service into independently deployable microservices"),
    # lexically-overlapping correction (easy recall — even lexical catches it):
    Case("near_lexical", True, "Use a connection pool of size 10",
         "Use a connection pool of size 50"),

    # ── unrelated, same type: must NOT supersede (semantic false-positive test) ──
    Case("unrelated", False, "Cache rendered templates for one hour",
         "Validate all inputs at the API boundary"),
    Case("unrelated", False, "Paginate list endpoints with a cursor",
         "Encrypt all secrets at rest"),
    Case("unrelated", False, "Run background work on a thread pool",
         "Gzip-compress HTTP responses over a size threshold"),
    Case("unrelated", False, "Rate-limit public endpoints per client",
         "Render dates in ISO-8601 in API responses"),

    # ── restatement within the SAME transcript: must NOT supersede (provenance) ──
    Case("restatement_same_evidence", False, "Adopt feature flags for gradual rollout",
         "We will use feature flags to control rollout", same_evidence=True),

    # ── in-session pivot: same transcript → NOT superseded by current design ─────
    Case("in_session_pivot", False, "Expose the public API over REST",
         "On reflection, expose the public API over GraphQL instead", same_evidence=True),

    # ── temporal: the existing event is NEWER → must NOT be superseded ───────────
    Case("temporal_older", False, "Encode timestamps as Unix epoch seconds",
         "Encode timestamps as ISO-8601 strings", new_is_newer=False),

    # ── authority: a lower-trust event must NOT supersede a higher-trust one ──────
    Case("lower_authority", False, "Authenticate requests with signed tokens",
         "Authenticate requests with opaque session cookies",
         old_authority="user_stated", new_authority="inferred_from_code"),
]


def _make_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


def _seed_case(conn: sqlite3.Connection, case: Case) -> tuple[Event, int]:
    """Insert the existing (old) event (+ its embedding); return (new_event, old_id)."""
    old_created = 1_000 if case.new_is_newer else 9_000
    new_created = 5_000
    new_evidence = 1 if case.same_evidence else 2

    old_payload = {"description": case.old_desc, "authority": case.old_authority}
    conn.execute(
        """INSERT INTO events (project_id, session_id, created_at, event_type,
                               payload, content_hash, weight, mention_count, evidence_id)
           VALUES ('p1', 's', ?, ?, ?, 'old_hash', 1.0, 1, 1)""",
        (old_created, case.event_type, json.dumps(old_payload)),
    )
    old_id = conn.execute("SELECT id FROM events WHERE content_hash='old_hash'").fetchone()["id"]
    conn.commit()

    if is_available():
        from cognikernel.embedding.input import embedding_input
        from cognikernel.embedding.model import EMBEDDING_MODEL_VERSION, embed_text
        from cognikernel.embedding.store import upsert_embedding
        vec = embed_text(embedding_input(old_payload, case.event_type))
        if vec is not None:
            upsert_embedding(conn, old_id, vec, EMBEDDING_MODEL_VERSION)
            conn.commit()

    new_event = Event(
        project_id="p1", session_id="s", event_type=case.event_type,
        payload={"description": case.new_desc, "authority": case.new_authority},
        content_hash="new_hash", created_at=new_created, evidence_id=new_evidence,
    )
    return new_event, old_id


def evaluate(threshold: float) -> dict:
    """Run every case at `threshold`; return confusion matrix + precision/recall."""
    original = supersede.SUPERSESSION_COSINE_THRESHOLD
    supersede.SUPERSESSION_COSINE_THRESHOLD = threshold
    tp = fp = tn = fn = 0
    false_positives: list[str] = []
    false_negatives: list[str] = []
    try:
        for case in CASES:
            conn = _make_conn()
            try:
                new_event, old_id = _seed_case(conn, case)
                predicted = old_id in find_superseded(conn, new_event, use_embeddings=True)
            finally:
                conn.close()
            if case.should_supersede and predicted:
                tp += 1
            elif case.should_supersede and not predicted:
                fn += 1
                false_negatives.append(f"[{case.cls}] {case.old_desc!r} -> {case.new_desc!r}")
            elif not case.should_supersede and predicted:
                fp += 1
                false_positives.append(f"[{case.cls}] {case.old_desc!r} -> {case.new_desc!r}")
            else:
                tn += 1
    finally:
        supersede.SUPERSESSION_COSINE_THRESHOLD = original

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {
        "threshold": threshold, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall,
        "false_positives": false_positives, "false_negatives": false_negatives,
    }


@pytest.mark.skipif(not is_available(), reason="embedding model not installed")
def test_precision_at_current_threshold() -> None:
    from cognikernel.delta.supersede import SUPERSESSION_COSINE_THRESHOLD
    m = evaluate(SUPERSESSION_COSINE_THRESHOLD)
    assert m["precision"] >= PRECISION_TARGET, (
        f"precision {m['precision']:.2f} < {PRECISION_TARGET} at threshold "
        f"{SUPERSESSION_COSINE_THRESHOLD}; false positives: {m['false_positives']}"
    )


@pytest.mark.skipif(not is_available(), reason="embedding model not installed")
def test_gates_hold_regardless_of_threshold() -> None:
    """The structured gates (provenance/temporal/authority) must block their cases
    even at a permissive threshold — they are not threshold-dependent."""
    gated = {"restatement_same_evidence", "in_session_pivot", "temporal_older", "lower_authority"}
    supersede_orig = supersede.SUPERSESSION_COSINE_THRESHOLD
    supersede.SUPERSESSION_COSINE_THRESHOLD = 0.0  # most permissive
    try:
        for case in CASES:
            if case.cls not in gated:
                continue
            conn = _make_conn()
            try:
                new_event, old_id = _seed_case(conn, case)
                assert old_id not in find_superseded(conn, new_event, use_embeddings=True), case.cls
            finally:
                conn.close()
    finally:
        supersede.SUPERSESSION_COSINE_THRESHOLD = supersede_orig


def _sweep() -> None:
    if not is_available():
        print("embedding model unavailable — install the `embedding` extra to run the sweep.")
        return
    print(f"{'thresh':>7} {'prec':>6} {'recall':>7} {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}")
    for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        m = evaluate(t)
        print(f"{t:>7.2f} {m['precision']:>6.2f} {m['recall']:>7.2f} "
              f"{m['tp']:>3} {m['fp']:>3} {m['tn']:>3} {m['fn']:>3}")
    print("\nDetail at the shipped threshold:")
    from cognikernel.delta.supersede import SUPERSESSION_COSINE_THRESHOLD
    m = evaluate(SUPERSESSION_COSINE_THRESHOLD)
    print(f"  threshold={SUPERSESSION_COSINE_THRESHOLD}  precision={m['precision']:.2f}  recall={m['recall']:.2f}")
    for fp in m["false_positives"]:
        print(f"  FP: {fp}")
    for fn in m["false_negatives"]:
        print(f"  FN: {fn}")


if __name__ == "__main__":
    _sweep()
