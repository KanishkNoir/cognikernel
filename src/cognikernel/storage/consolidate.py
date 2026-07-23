"""Golden-record consolidation at READ (J2.3 + J3.1).

Groups same-key choice-family projection recs and emits ONE canonical rec per
group — the MDM golden-record pattern, mirroring the existing per-path
component collapse (projections.py) so greedy/backstop behavior stays
predictable. Latest-wins canonical pick: within the highest authority tier
present, the newest event speaks for the topic; demoted DISTINCT values stay
visible as lineage on the canonical line.

Read-time only and reversible — no archived/superseded writes; raw events are
untouched (lossless). An event with no key passes through exactly as before.

Grouping rule (v1, locked by measurement on the gamma DB — see
scripts/measure_groups.py): EXACT key equality only. One-token keys must also
pass a description-Jaccard floor, because generic single tokens ('key',
'write', 'safety') otherwise fuse unrelated topics — the measured group-of-5
wrong-merge. Subset/df-gated matching was measured and rejected: it bought 2
extra groups at the cost of that wrong-merge class. Evolution chains whose
links share no lexical surface (the F-C value-evolution case) are NOT
collapsible by deterministic keys — that stays on the structural-supersession
track, documented in the sprint plan.
"""
from __future__ import annotations

from typing import Any

from cognikernel.delta.supersede import jaccard_similarity

# Higher = more authoritative; mirrors delta.supersede._AUTHORITY_PRECEDENCE.
# (Consolidating the three scattered authority tables into one shared home is
# tracked in the sprint plan; supersede's is imported to avoid a fourth copy.)
from cognikernel.delta.supersede import _AUTHORITY_PRECEDENCE, _AUTHORITY_DEFAULT

_SINGLE_TOKEN_JACCARD_FLOOR = 0.3
_LINEAGE_MAX = 2


def _authority_rank(rec: dict[str, Any]) -> int:
    return _AUTHORITY_PRECEDENCE.get(
        rec["payload"].get("authority", ""), _AUTHORITY_DEFAULT
    )


def _canonical_sort_key(rec: dict[str, Any]) -> tuple[int, int, int]:
    """Latest-wins: highest authority tier, then newest, then highest id."""
    return (
        _authority_rank(rec),
        rec.get("created_at") or 0,
        rec.get("id") or 0,
    )


def _norm_desc(rec: dict[str, Any]) -> str:
    return " ".join((rec["payload"].get("description") or "").lower().split())


def consolidate_by_key(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse same-key recs to one canonical each. Keyless recs pass through.

    Mirrors the component collapse quantities: mention_count sums across the
    group (repetition signal preserved for composite weighting), weight takes
    the max (nothing drops out of the greedy budget *because* it was grouped).
    """
    by_key: dict[str, list[dict[str, Any]]] = {}
    out: list[dict[str, Any]] = []
    for rec in recs:
        key = rec.get("decision_key") or ""
        if not key:
            out.append(rec)
            continue
        by_key.setdefault(key, []).append(rec)

    for key, group in by_key.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        if " " not in key:
            # Single-token key: generic tokens fuse unrelated topics. Keep only
            # members lexically related to the newest one; the rest pass through.
            group.sort(key=_canonical_sort_key, reverse=True)
            head, related = group[0], [group[0]]
            for rec in group[1:]:
                if jaccard_similarity(_norm_desc(head), _norm_desc(rec)) >= _SINGLE_TOKEN_JACCARD_FLOOR:
                    related.append(rec)
                else:
                    out.append(rec)
            group = related
            if len(group) == 1:
                out.append(group[0])
                continue

        group.sort(key=_canonical_sort_key, reverse=True)
        canonical = dict(group[0])
        canonical["payload"] = dict(canonical["payload"])
        canonical["mention_count"] = sum(r.get("mention_count", 1) for r in group)
        canonical["weight"] = max(r.get("weight", 0.0) for r in group)
        canonical["payload"]["provenance_count"] = len(group)

        # Lineage carries DISTINCT demoted values: a candidate folds silently
        # when it restates the canonical OR an already-accepted lineage entry.
        accepted: list[str] = [_norm_desc(canonical)]
        lineage: list[dict[str, str]] = []
        for rec in group[1:]:
            d = _norm_desc(rec)
            if any(d == a or jaccard_similarity(d, a) >= 0.85 for a in accepted):
                continue
            if len(lineage) < _LINEAGE_MAX:
                lineage.append({
                    "description": (rec["payload"].get("description") or "")[:80],
                    "session_id": rec.get("session_id", ""),
                })
                accepted.append(d)
        if lineage:
            canonical["payload"]["lineage"] = lineage
        out.append(canonical)

    return out
