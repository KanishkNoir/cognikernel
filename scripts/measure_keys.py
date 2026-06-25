"""J2.2 measurement — decision-key coverage + gold-chain grouping on a real DB.

Usage: python scripts/measure_keys.py [--db PATH]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from memlora.extraction.decision_key import CHOICE_FAMILY, derive_decision_key

# The three measured evolution chains (gamma): event ids that SHOULD co-key.
GOLD_CHAINS = {
    "D4 default-alias": [27, 157, 172, 206, 325],
    "D2 retry-policy": [150, 153, 177, 209, 240],
    "D8 cache-hit": [97, 227, 265, 301, 302, 329],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=r".bench_dbs\gamma_cogni.db")
    args = ap.parse_args()
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        "SELECT id, event_type, payload FROM events WHERE archived=0 AND superseded_by IS NULL"
    ).fetchall()
    keys: dict[int, str] = {}
    n_choice = n_keyed = 0
    key_sizes: Counter = Counter()
    for r in rows:
        payload = json.loads(r["payload"])
        k = derive_decision_key(payload, r["event_type"])
        keys[r["id"]] = k
        if r["event_type"] in CHOICE_FAMILY:
            n_choice += 1
            if k:
                n_keyed += 1
                key_sizes[k] += 1

    print(f"choice-family events: {n_choice}; keyed: {n_keyed} "
          f"({100*n_keyed/max(n_choice,1):.0f}%)")
    print(f"distinct keys: {len(key_sizes)}; groups >1: "
          f"{sum(1 for v in key_sizes.values() if v > 1)}")
    print("\nlargest groups:")
    for k, n in key_sizes.most_common(8):
        print(f"  {n:>2}  {k!r}")

    print("\n--- gold chains ---")
    for name, ids in GOLD_CHAINS.items():
        got = [(i, keys.get(i, "<gone>")) for i in ids]
        kc = Counter(k for _, k in got if k and k != "<gone>")
        dominant, dom_n = (kc.most_common(1)[0] if kc else ("", 0))
        print(f"\n{name}: dominant key {dominant!r} covers {dom_n}/{len(ids)}")
        for i, k in got:
            d = ""
            row = conn.execute(
                "SELECT json_extract(payload,'$.description') FROM events WHERE id=?", (i,)
            ).fetchone()
            if row:
                d = (row[0] or "")[:70]
            print(f"  #{i:<4} key={k!r:<35} {d}")


if __name__ == "__main__":
    main()
