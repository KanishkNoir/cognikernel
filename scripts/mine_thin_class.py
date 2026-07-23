"""Mine candidate sentences for the thin eval classes (P0).

THREAD (n=6), CONSTRAINT_SOFT (n=8), APPROACH_ABANDONED (n=10) drive half of
macro-F1 but are too small to measure — single-item flips swing the ranking.
This mines cue-matched candidates from ALL real stores, decontaminated against
the current eval + manual labels, for hand-curation into manual_labels.jsonl.

Sentences are surfaced exactly as the pipeline sees them (tokenize -> skip code
-> sanitize -> content-word floor), so a label assigned here matches inference.

Writes: research/model_eval/_thin_candidates.txt  (idx|guess|store|text)

Usage: uv run python scripts/mine_thin_class.py [--per-class 60]
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "scripts")

from build_model_eval import iter_store_sentences  # noqa: E402
from cognikernel.delta.supersede import normalize_for_overlap  # noqa: E402

EVAL = Path("research/model_eval")
PROJECTS = Path.home() / ".cognikernel" / "projects"

_CUES = {
    "THREAD": re.compile(
        r"\b(next session|next time|still (need|to do|todo|have to|gotta)|"
        r"remaining|in progress|half[- ]done|pick up|left off|to be done|"
        r"come back to|follow[- ]up|not (yet|done)|will (add|wire|implement) next|"
        r"active thread|open (item|question|work))\b", re.I),
    "CONSTRAINT_SOFT": re.compile(
        r"\b(prefer|convention|naming|style|id rather|i'd rather|should probably|"
        r"sparingly|consistent(ly)?|keep .* (under|short|small)|tend to|"
        r"as a rule of thumb|readab|lint|format(ting)?|organi[sz]e)\b", re.I),
    "APPROACH_ABANDONED_DO_NOT_RETRY": re.compile(
        r"\b(instead of|rather than|ruled out|abandon\w*|scrap\w*|drop the|"
        r"no longer|gave up|deprecat\w*|rejected|not use|don'?t use|avoid|"
        r"decided against|won'?t use|tried .* before|never again|do not (adopt|retry|use)|"
        r"stop using|migrate away)\b", re.I),
}
# Down-rank obvious non-fits per class (e.g. a hard "must never" is HARD not SOFT).
_HARD_NEG = re.compile(r"\b(must|never|always|shall|required|mandatory)\b", re.I)


def _existing_sigs() -> set[frozenset]:
    sigs = set()
    for name in ("salience_eval.jsonl", "manual_labels.jsonl"):
        p = EVAL / name
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                t = json.loads(line).get("text", "")
                s = frozenset(normalize_for_overlap(t))
                if s:
                    sigs.add(s)
    return sigs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=60)
    args = ap.parse_args()

    existing = _existing_sigs()
    seen: set[frozenset] = set()
    found: dict[str, list[tuple[str, str]]] = {k: [] for k in _CUES}

    dbs = sorted(glob.glob(str(PROJECTS / "*.db")))
    for dbp in dbs:
        pid = Path(dbp).stem
        try:
            for desc, role in iter_store_sentences(pid):
                if len(desc) < 25 or len(desc) > 220:
                    continue
                sig = frozenset(normalize_for_overlap(desc))
                if not sig or sig in existing or sig in seen:
                    continue
                for cls, rx in _CUES.items():
                    if len(found[cls]) >= args.per_class:
                        continue
                    if rx.search(desc):
                        # SOFT should not fire on hard-rule wording.
                        if cls == "CONSTRAINT_SOFT" and _HARD_NEG.search(desc):
                            continue
                        found[cls].append((pid[:8], desc))
                        seen.add(sig)
                        break
        except Exception:
            continue

    out = EVAL / "_thin_candidates.txt"
    with open(out, "w", encoding="utf-8") as f:
        idx = 0
        for cls in ("THREAD", "CONSTRAINT_SOFT", "APPROACH_ABANDONED_DO_NOT_RETRY"):
            for store, text in found[cls]:
                f.write(f"{idx:03d}|{cls}|{store}|{text}\n")
                idx += 1
    print(f"wrote {sum(len(v) for v in found.values())} candidates -> {out}")
    for cls, v in found.items():
        print(f"  {cls:34} {len(v)}")


if __name__ == "__main__":
    main()
