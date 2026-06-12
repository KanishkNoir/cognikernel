"""J2.3 + J3.1 — golden-record consolidation: canonical pick, lineage, guards."""
from __future__ import annotations

from memlora.storage.consolidate import consolidate_by_key


def _rec(id_, desc, *, key="", etype="DECISION", authority="assistant_decided",
         created=0, weight=0.5, mentions=1, session="s1"):
    return {
        "id": id_, "event_type": etype, "weight": weight,
        "mention_count": mentions, "session_id": session,
        "content_hash": f"h{id_}", "created_at": created, "decision_key": key,
        "payload": {"description": desc, "authority": authority},
    }


class TestGrouping:
    def test_keyless_pass_through(self) -> None:
        recs = [_rec(1, "a"), _rec(2, "b")]
        assert consolidate_by_key(recs) == recs

    def test_exact_multitoken_key_groups(self) -> None:
        out = consolidate_by_key([
            _rec(1, "TTFT timeout: default 10 s, per provider", key="timeout ttft", created=1),
            _rec(2, "TTFT timeout: 10 s, configurable per provider", key="timeout ttft", created=2),
        ])
        assert len(out) == 1
        assert out[0]["id"] == 2  # newest wins at same authority

    def test_single_token_key_needs_jaccard(self) -> None:
        """The measured wrong-merge class: generic one-token keys ('key',
        'write') must not fuse unrelated topics."""
        out = consolidate_by_key([
            _rec(1, "Key format: rly_<43 chars base62> from os.urandom", key="key", created=1),
            _rec(2, "Key: SHA-256 of a canonical JSON of payload fields", key="key", created=2),
        ])
        assert len(out) == 2  # unrelated — both render

    def test_single_token_key_groups_when_related(self) -> None:
        out = consolidate_by_key([
            _rec(1, "Retry: 2 attempts per deployment, base=100 ms, max=500 ms", key="retry", created=1),
            _rec(2, "Retry: 3 attempts per deployment, full jitter, max=500 ms", key="retry", created=2),
        ])
        assert len(out) == 1
        assert out[0]["id"] == 2


class TestCanonicalPick:
    def test_authority_beats_recency(self) -> None:
        out = consolidate_by_key([
            _rec(1, "limiter counters live in Redis sliding window", key="counter limit",
                 authority="user_stated", created=1),
            _rec(2, "limiter counters live in a process-local dict window", key="counter limit",
                 authority="inferred_from_code", created=99),
        ])
        assert len(out) == 1
        assert out[0]["id"] == 1  # user_stated outranks inferred despite age

    def test_recency_breaks_authority_ties(self) -> None:
        out = consolidate_by_key([
            _rec(1, "cache hit needs exact SHA match of fields", key="cache hit", created=1),
            _rec(2, "cache hit needs cosine 0.97 semantic match of fields", key="cache hit", created=2),
        ])
        assert out[0]["id"] == 2


class TestGoldenRecordQuantities:
    def test_mention_sum_weight_max(self) -> None:
        out = consolidate_by_key([
            _rec(1, "span name relay.provider_call with gen_ai attributes", key="name span",
                 created=1, weight=0.9, mentions=3),
            _rec(2, "span name relay.provider_call plus outcome attributes", key="name span",
                 created=2, weight=0.4, mentions=2),
        ])
        assert len(out) == 1
        assert out[0]["mention_count"] == 5
        assert out[0]["weight"] == 0.9
        assert out[0]["payload"]["provenance_count"] == 2

    def test_lineage_carries_distinct_values_only(self) -> None:
        out = consolidate_by_key([
            _rec(1, "cache hit requires exact SHA-256 match", key="cache hit", created=1),
            _rec(2, "cache hit requires exact SHA-256 match", key="cache hit", created=2),
            _rec(3, "cache hit requires cosine 0.97 between embeddings now", key="cache hit", created=3),
        ])
        assert len(out) == 1
        lineage = out[0]["payload"].get("lineage", [])
        # The two exact restatements fold silently; one distinct old value remains.
        assert len(lineage) == 1
        assert "SHA-256" in lineage[0]["description"]

    def test_lineage_capped_at_two(self) -> None:
        out = consolidate_by_key([
            _rec(i, f"retry policy generation {i} uses {v}", key="policy retry", created=i)
            for i, v in enumerate(["alpha backoff", "beta jitter", "gamma window", "delta cap"], 1)
        ])
        assert len(out) == 1
        assert len(out[0]["payload"]["lineage"]) == 2

    def test_input_recs_not_mutated(self) -> None:
        recs = [
            _rec(1, "TTL: 600 seconds for the completion cache", key="ttl", created=1),
            _rec(2, "TTL: 3600 seconds for the completion cache", key="ttl", created=2),
        ]
        consolidate_by_key(recs)
        assert "lineage" not in recs[1]["payload"]
        assert recs[1]["mention_count"] == 1
