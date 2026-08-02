"""Deterministic A/B for the path-recall fix: old regex vs new, same inputs.

Research tooling — NOT part of the shipped package.

This measures exactly what Task 4 changed, and it needs no transcript recovery:
it replays both patterns over the `raw_evidence` blobs already stored in every
local project DB. Only 22 of 139 stored sessions still have a JSONL transcript
on disk, so this is the stronger evidence for this specific claim.

Counts paths EXTRACTABLE (matched and canonicalized to a usable relative path),
not merely matched, because a match that canonicalizes to '' is dropped by
file_mentions and would inflate the "after" column dishonestly.

Read-only. Usage:
    python scripts/path_recall_ab.py
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import zlib
from collections import Counter
from pathlib import Path

from cognikernel.extraction.file_mentions import _FILE_PATTERN as NEW_PATTERN
from cognikernel.extraction.transcript import transcript_from_source
from cognikernel.utils.paths import canonicalize_path, is_bare_basename

# The pattern exactly as it stood before the fix (git 409707f and earlier).
OLD_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9_./\\])"
    r"(?:[a-zA-Z0-9_][a-zA-Z0-9_/.-]*/)*"
    r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*\."
    r"(?:py|ts|tsx|js|jsx|mjs|json|yaml|yml|sql|md|toml|env|cfg|ini|go|rs|java|cs)"
    r"(?![a-zA-Z0-9_])",
    re.ASCII,
)


def extractable(pattern: re.Pattern, text: str) -> set[str]:
    """Paths this pattern would actually turn into a component, post-canonicalization.

    Deduplicates the raw match strings BEFORE canonicalizing. A transcript
    mentions the same path hundreds of times, and canonicalizing each occurrence
    made this scan quadratic in repetition for no extra information.
    """
    raw = {m.group(0) for m in pattern.finditer(text)}
    out: set[str] = set()
    for candidate in raw:
        p = canonicalize_path(candidate)
        if p and not is_bare_basename(p):
            out.add(p)
    return out


def shape(path: str) -> str:
    """Bucket a newly-recovered path by why the old pattern missed it."""
    if path.startswith("."):
        return "dot-directory"
    if "\\" in path:
        return "windows-separator"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projects-dir",
                    default=str(Path.home() / ".cognikernel" / "projects"))
    ap.add_argument("--out", default="docs/metrics/path_recall_ab.json")
    args = ap.parse_args()

    stores = sorted(Path(args.projects_dir).glob("*.db"))
    old_total = new_total = 0
    blobs = 0
    stores_improved = 0
    shapes: Counter = Counter()
    examples: list[str] = []

    for db in stores:
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            rows = conn.execute(
                "SELECT content_encoding, content_blob, source_type FROM raw_evidence"
            ).fetchall()
        except Exception:
            continue

        store_old = store_new = 0
        print(f"  {db.name}: {len(rows)} blobs", flush=True)
        for encoding, blob, source_type in rows:
            if blob is None:
                continue
            try:
                raw = zlib.decompress(blob) if encoding == "zlib" else blob
                text = raw.decode("utf-8", errors="replace")
                # Decode exactly as production does. Feeding RAW JSONL here is
                # invalid: its literal "\n" and "\r" escapes are backslash+letter,
                # which the Windows-separator support reads as directory
                # separators, fusing prose and paths into strings like
                # 'skeleton/n/nsrc/conductor/driver.py'. Production never sees
                # that — jsonl_to_transcript parses each line first, so the
                # escapes are already real newlines by the time the pattern runs.
                text = transcript_from_source(source_type, text)
            except Exception:
                continue
            blobs += 1
            o = extractable(OLD_PATTERN, text)
            n = extractable(NEW_PATTERN, text)
            store_old += len(o)
            store_new += len(n)
            for p in (n - o):
                shapes[shape(p)] += 1
                if len(examples) < 10 and p not in examples:
                    examples.append(p)
        conn.close()

        old_total += store_old
        new_total += store_new
        if store_new > store_old:
            stores_improved += 1

    gain = new_total - old_total
    report = {
        "stores_scanned": len(stores),
        "evidence_blobs": blobs,
        "paths_extractable_old": old_total,
        "paths_extractable_new": new_total,
        "absolute_gain": gain,
        "relative_gain_pct": (100.0 * gain / old_total) if old_total else 0.0,
        "stores_improved": stores_improved,
        "recovered_by_shape": dict(shapes),
        "examples": examples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"stores {len(stores)}   evidence blobs {blobs}")
    print(f"paths extractable  old : {old_total}")
    print(f"paths extractable  new : {new_total}")
    print(f"gain                   : +{gain} "
          f"({report['relative_gain_pct']:.1f}%) across {stores_improved} stores")
    print(f"recovered by shape     : {dict(shapes)}")
    for e in examples[:5]:
        print(f"   e.g. {e}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
