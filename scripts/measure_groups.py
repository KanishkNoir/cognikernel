"""J2.3 design measurement — read-time union-find grouping of choice-family events.

Edges: (a) decision-key exact/subset match, (b) the existing pairwise
supersedes() predicate (lexical overlap OR subject-keyed). Reports gold-chain
coverage, group stats, runtime, and the largest groups for wrong-merge review.

Usage: python scripts/measure_groups.py [--db PATH]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from memlora.delta.supersede import supersedes
from memlora.extraction.decision_key import CHOICE_FAMILY, derive_decision_key

GOLD_CHAINS = {
    "D4 default-alias": [27, 157, 172, 206, 325],
    "D2 retry-policy": [150, 153, 177, 209, 240],
    "D8 cache-hit": [97, 227, 265, 301, 302, 329],
}


class DSU:
    def __init__(self) -> None:
        self.p: dict[int, int] = {}

    def find(self, x: int) -> int:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        self.p[self.find(a)] = self.find(b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=r".bench_dbs\gamma_cogni.db")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, event_type, payload FROM events "
        "WHERE archived=0 AND superseded_by IS NULL"
    ).fetchall()
    evs = []
    for r in rows:
        if r["event_type"] not in CHOICE_FAMILY:
            continue
        payload = json.loads(r["payload"])
        key = derive_decision_key(payload, r["event_type"])
        evs.append({
            "id": r["id"], "type": r["event_type"],
            "desc": payload.get("description", ""), "key": key,
            "ktoks": frozenset(key.split()) if key else frozenset(),
        })

    # Token document frequency across choice-family keys: generic tokens
    # ('key', 'config') may not carry a subset edge — the MDM blocking-key rule.
    df: Counter = Counter()
    for e in evs:
        df.update(e["ktoks"])
    n_generic_cap = max(3, len(evs) // 33)  # ~3% of choice events
    generic = {t for t, n in df.items() if n > n_generic_cap}

    t0 = time.perf_counter()
    dsu = DSU()
    n_key_edges = n_pred_edges = 0
    for i in range(len(evs)):
        a = evs[i]
        for j in range(i + 1, len(evs)):
            b = evs[j]
            if not (a["ktoks"] and b["ktoks"]):
                continue
            small, big = (a["ktoks"], b["ktoks"]) if len(a["ktoks"]) <= len(b["ktoks"]) else (b["ktoks"], a["ktoks"])
            if small == big or (small <= big and not (small <= generic)):
                dsu.union(a["id"], b["id"])
                n_key_edges += 1
    dt = time.perf_counter() - t0
    print(f"generic tokens (df > {n_generic_cap}): {sorted(generic)[:20]}")

    groups: dict[int, list[dict]] = {}
    for e in evs:
        groups.setdefault(dsu.find(e["id"]), []).append(e)
    sizes = Counter(len(v) for v in groups.values())
    n_grouped = sum(len(v) for v in groups.values() if len(v) > 1)
    print(f"choice events: {len(evs)}  pairwise time: {dt*1000:.0f} ms")
    print(f"edges: key={n_key_edges} predicate={n_pred_edges}")
    print(f"groups: {len(groups)} (size hist: {dict(sorted(sizes.items()))})")
    print(f"events in multi-groups: {n_grouped} "
          f"({100*n_grouped/max(len(evs),1):.0f}% — render-line reduction potential)")

    print("\nlargest groups (wrong-merge review):")
    for root, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:6]:
        print(f"  --- group of {len(members)}")
        for m in members[:8]:
            print(f"    #{m['id']:<4} [{m['type'][:15]:<15}] key={m['key']!r:<28} {m['desc'][:75]}")

    print("\n--- gold chains ---")
    for name, ids in GOLD_CHAINS.items():
        roots = Counter()
        for i in ids:
            if i in dsu.p or any(e["id"] == i for e in evs):
                roots[dsu.find(i)] += 1
        dom = roots.most_common(1)[0][1] if roots else 0
        print(f"{name}: dominant group covers {dom}/{len(ids)} "
              f"(distinct groups: {len(roots)})")


if __name__ == "__main__":
    main()
