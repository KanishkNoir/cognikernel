"""Phase A statement-quality audit — generator for human-verification subset.

Four hosted LLMs independently labeled 60 memory statements. 58 statements were
labeled by all four. A human now hand-labels a 20-statement subset to measure
model-vs-human agreement. The human must not see any model's answer.

Reads the four label files, restricts to IDs present in all four, computes
each id's vote bin (how many models marked it defective), allocates 20 slots
proportionally to bin size, and selects deterministically.

Writes:
  research/statement_audit/verify_<stamp>.jsonl      (labeler opens this, blinded)
  research/statement_audit/verify_key_<stamp>.json   (answer key, do NOT open yet)

Usage: uv run python scripts/audit_verify_subset.py [--seed 1106] \
           [--labels-glob "labels_*_20260726-004114.jsonl"] \
           [--pool research/statement_audit/pool_20260725-193802.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("research/statement_audit")

# Label files are written as labels_<model-slug>_<stamp>.jsonl by
# audit_label_llm.py. Extracting the slug back out lets the answer key be
# keyed by model rather than by an unlabeled position in a list.
_LABEL_FILENAME_RX = re.compile(r"^labels_(.+)_\d{8}-\d{6}\.jsonl$")


def model_id_from_label_filename(path: str) -> str:
    """The model slug embedded in a labels_<slug>_<stamp>.jsonl filename.

    Falls back to the bare filename stem if it doesn't match the expected
    pattern, so an oddly-named file still gets a usable (if less clean) key
    instead of crashing.
    """
    m = _LABEL_FILENAME_RX.match(Path(path).name)
    return m.group(1) if m else Path(path).stem


def vote_bin(label_sets: list[set[str]]) -> int:
    """Count how many labelers marked the statement defective.

    Returns 0..len(label_sets). A labeler votes defective if their set
    contains any code other than CLEAN.
    """
    defective_count = 0
    for labels in label_sets:
        # Defective if any code other than CLEAN is present
        if any(code != "CLEAN" for code in labels):
            defective_count += 1
    return defective_count


def proportional_allocation(bin_counts: dict[int, int], n: int) -> dict[int, int]:
    """Allocate n slots across bins proportionally to bin size using largest-remainder.

    Returns a dict mapping bin -> allocation count.
    - Allocations sum to min(n, total_members)
    - No non-empty bin gets 0 unless n < number_of_non_empty_bins
    - Each bin's allocation is capped at its member count

    The seed passed to select_subset must be the only source of randomness in
    this pipeline, so every step here iterates `bin_counts` in sorted (bin_id)
    order rather than dict-insertion order — insertion order upstream derives
    from iterating a Python set, which varies with PYTHONHASHSEED. Without
    this, largest-remainder ties (in the sort below and in the bump-repair
    step) would be broken by hash-seed-dependent order instead of the seed.
    """
    if not bin_counts or n == 0:
        return {k: 0 for k in bin_counts}

    # Cap n at total members available
    total_members = sum(bin_counts.values())
    n_capped = min(n, total_members)

    # Step 1: Largest-remainder method (proportional allocation)
    alloc = {}
    total_assigned = 0
    remainders = {}

    for bin_id, count in sorted(bin_counts.items()):
        # Proportional share
        share = (count / total_members) * n_capped
        floor_val = int(share)
        # Cap at actual member count
        alloc[bin_id] = min(floor_val, count)
        remainders[bin_id] = share - floor_val
        total_assigned += alloc[bin_id]

    # Distribute remaining slots by largest remainder, tie-broken by bin_id
    remaining = n_capped - total_assigned
    if remaining > 0:
        # Sort by remainder (largest first), then by bin_id for deterministic ties
        sorted_bins = sorted(remainders.items(), key=lambda x: (-x[1], x[0]))
        for i in range(remaining):
            bin_id = sorted_bins[i][0]
            # Only assign if we haven't hit the bin's member count
            if alloc[bin_id] < bin_counts[bin_id]:
                alloc[bin_id] += 1

    # Step 2: Repair — any non-empty bin with 0 allocation gets bumped to 1
    for bin_id, count in sorted(bin_counts.items()):
        if count > 0 and alloc[bin_id] == 0:
            # Take from the bin with largest allocation, tie-broken by bin_id
            max_bin = max((b for b in sorted(alloc) if b != bin_id),
                         key=lambda b: alloc[b])
            if alloc[max_bin] > 0:
                alloc[max_bin] -= 1
                alloc[bin_id] = 1

    return alloc


def select_subset(rows_by_bin: dict[int, list[str]], alloc: dict[int, int],
                  seed: int) -> list[str]:
    """Deterministically select ids from each bin using seed.

    Sorts ids within each bin and bin keys to ensure result is independent
    of input dict/file ordering.
    """
    rng = random.Random(seed)
    selected = []

    # Iterate bins in sorted order for determinism
    for bin_id in sorted(rows_by_bin.keys()):
        ids = rows_by_bin[bin_id]
        count = alloc.get(bin_id, 0)

        if count > 0:
            # Sort ids within bin for determinism
            sorted_ids = sorted(ids)
            # Shuffle deterministically
            shuffled = list(sorted_ids)
            rng.shuffle(shuffled)
            # Take first 'count' after shuffle
            selected.extend(shuffled[:count])

    # Final shuffle to avoid bin-order dependency in output
    rng.shuffle(selected)
    return selected


def verify_rows(selected_ids: list[str], pool_by_id: dict) -> list[dict]:
    """Build verify_*.jsonl rows: only pool schema, no model labels."""
    rows = []
    for sid in selected_ids:
        if sid not in pool_by_id:
            sys.exit(f"ERROR: id {sid} not found in pool — blinding violation")
        pool_row = pool_by_id[sid]
        rows.append({
            "id": sid,
            "event_type": pool_row["event_type"],
            "text": pool_row["text"],
            "labels": [],
            "notes": "",
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1106, help="random seed")
    ap.add_argument("--labels-glob", default="labels_*_20260726-004114.jsonl",
                    help="glob (relative to research/statement_audit/) matching "
                         "the label files to read; must match exactly 4")
    ap.add_argument("--pool", type=Path,
                    default=OUT_DIR / "pool_20260725-193802.jsonl",
                    help="pool jsonl providing event_type/text for the "
                         "blinded verify rows")
    args = ap.parse_args()

    # Find and read the label files
    label_pattern = OUT_DIR / args.labels_glob
    label_files = sorted(glob.glob(str(label_pattern)))

    if len(label_files) != 4:
        sys.exit(f"ERROR: expected exactly 4 label files matching {label_pattern}, "
                 f"found {len(label_files)}")

    # Read labels from each file, keyed by model id (not a bare positional
    # list) so the answer key can name which model said what.
    label_sets_by_id: dict[str, dict[str, set[str]]] = collections.defaultdict(dict)
    all_files_ids = None

    for label_file in label_files:
        model_id = model_id_from_label_filename(label_file)
        file_ids = set()
        with open(label_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    sid = row["id"]
                    labels = set(row.get("labels") or [])
                    label_sets_by_id[sid][model_id] = labels
                    file_ids.add(sid)

        if all_files_ids is None:
            all_files_ids = file_ids
        else:
            all_files_ids = all_files_ids.intersection(file_ids)

    print(f"found {len(label_files)} label files")
    print(f"ids present in all four: {len(all_files_ids)}")

    # Read pool to get event_type and text
    pool_file = args.pool
    pool_by_id: dict[str, dict] = {}

    with open(pool_file, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                pool_by_id[row["id"]] = row

    # Compute vote bins for each id in the intersection
    rows_by_bin: dict[int, list[str]] = collections.defaultdict(list)

    for sid in all_files_ids:
        label_sets = list(label_sets_by_id[sid].values())
        bin_num = vote_bin(label_sets)
        rows_by_bin[bin_num].append(sid)

    # Show bin distribution
    print("\nbin distribution before allocation:")
    for bin_id in sorted(rows_by_bin.keys()):
        print(f"  bin {bin_id} (votes): {len(rows_by_bin[bin_id])} statements")

    # Allocate 20 slots proportionally
    bin_counts = {b: len(ids) for b, ids in rows_by_bin.items()}
    alloc = proportional_allocation(bin_counts, 20)

    print("\nallocation (20 total):")
    for bin_id in sorted(alloc.keys()):
        if alloc[bin_id] > 0:
            print(f"  bin {bin_id}: {alloc[bin_id]} statements")

    # Select subset deterministically
    selected_ids = select_subset(rows_by_bin, alloc, args.seed)

    # Generate output rows
    rows = verify_rows(selected_ids, pool_by_id)

    # Build answer key — restricted to the 20 selected ids (not all 58, which
    # would leak the un-selected statements' model labels into a file that
    # should only ever cover what the human actually labeled), keyed by model
    # id (not an unnamed positional list), with each model's label list
    # sorted for determinism.
    answer_key = {}
    for sid in sorted(selected_ids):
        per_model = label_sets_by_id[sid]
        bin_num = vote_bin(list(per_model.values()))
        answer_key[sid] = {
            "labels": {model_id: sorted(labels)
                      for model_id, labels in sorted(per_model.items())},
            "vote_bin": bin_num,
        }

    # Write outputs
    stamp = time.strftime("%Y%m%d-%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    verify_path = OUT_DIR / f"verify_{stamp}.jsonl"
    with verify_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    key_path = OUT_DIR / f"verify_key_{stamp}.json"
    with key_path.open("w", encoding="utf-8") as f:
        json.dump(answer_key, f, indent=2, ensure_ascii=False)

    print(f"\nwrote {verify_path} ({len(rows)} to label)")
    print(f"wrote {key_path}")
    print("\n" + "=" * 70)
    print("WARNING: The human labeler must NOT open verify_key_*.json")
    print("before completing the verify_*.jsonl labels. Opening the key file")
    print("will compromise the agreement measurement.")
    print("=" * 70)


if __name__ == "__main__":
    main()
