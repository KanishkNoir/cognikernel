#!/usr/bin/env python3
"""Supersession quality harness — scores a pairwise predicate against a frozen gold fixture.

This is the regression gate for Stage 5 supersession (the text decision only). It asks,
for each labeled pair, whether the NEWER description `a` should supersede the OLDER `b`,
and compares the predicate's answer to the gold relation.

  precision   of pairs the predicate flags as superseding, fraction that truly should
  recall      of pairs that truly should supersede, fraction the predicate catches
  guard_fp    GUARD negatives (same-domain different-decision) wrongly flagged — the
              dangerous class: a false supersession HIDES a still-valid decision

It is version-agnostic: it scores the current lexical+subject `supersedes()` today and
any future cross-encoder on the SAME pairs. It never trains on anything — the fixture
(tests/fixtures/supersession_pairs_gold.json) is the held-out eval.

Default predicate is cognikernel.delta.supersede.supersedes(new, old). Swap with --predicate
to score an alternative (an importable "module:function" taking (new_desc, old_desc) -> bool).

Usage:
    python scripts/eval_supersession.py
    python scripts/eval_supersession.py --check          # exit 1 if any target unmet
    python scripts/eval_supersession.py --predicate mypkg.mod:my_supersedes
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Callable

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_GOLD = _ROOT / "tests" / "fixtures" / "supersession_pairs_gold.json"

# Make the package importable when run from a checkout without install.
sys.path.insert(0, str(_ROOT / "src"))


def load_predicate(spec: str | None) -> Callable[[str, str], bool]:
    """Return a (new_desc, old_desc) -> bool predicate. Default: lexical+subject supersedes()."""
    if not spec:
        from cognikernel.delta.supersede import supersedes
        return supersedes
    mod_name, _, fn_name = spec.partition(":")
    if not fn_name:
        raise SystemExit(f"--predicate must be 'module:function', got {spec!r}")
    mod = importlib.import_module(mod_name)
    return getattr(mod, fn_name)


def score(pairs: list[dict], rel_map: dict[str, dict], predicate: Callable[[str, str], bool]) -> dict:
    tp = fp = fn = tn = 0
    guard_fp = 0
    fp_examples: list[str] = []
    fn_examples: list[str] = []
    guard_examples: list[str] = []

    for p in pairs:
        should = bool(rel_map[p["relation"]]["should_supersede"])
        got = bool(predicate(p["a"], p["b"]))

        if got and should:
            tp += 1
        elif got and not should:
            fp += 1
            fp_examples.append(f'{p["id"]} [{p["relation"]}] {p["a"]!r} SUPERSEDES {p["b"]!r}')
            if p.get("guard"):
                guard_fp += 1
                guard_examples.append(f'{p["id"]} {p["a"]!r} SUPERSEDES {p["b"]!r}')
        elif not got and should:
            fn += 1
            fn_examples.append(f'{p["id"]} [{p["relation"]}] {p["a"]!r} | {p["b"]!r}')
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "n": len(pairs), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "guard_fp": guard_fp,
        "fp_examples": fp_examples, "fn_examples": fn_examples, "guard_examples": guard_examples,
    }


def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def report(s: dict, targets: dict) -> bool:
    checks = [
        ("precision     ", f"{s['precision']:.0%}", s["precision"] >= targets["precision_min"], f">= {targets['precision_min']:.0%}"),
        ("recall        ", f"{s['recall']:.0%}", s["recall"] >= targets["recall_min"], f">= {targets['recall_min']:.0%}"),
        ("guard_fp      ", f"{s['guard_fp']}", s["guard_fp"] <= targets["guard_false_positives_max"], f"<= {targets['guard_false_positives_max']}"),
    ]
    print("\n=== Supersession scorecard (pairwise, text-only) ===")
    print(f"{'metric':<16} {'value':>6}   {'target':<10} result")
    print("-" * 48)
    all_ok = True
    for name, val, ok, tgt in checks:
        all_ok = all_ok and ok
        print(f"{name} {val:>6}   {tgt:<10} {_flag(ok)}")

    print(f"\nconfusion:  tp={s['tp']} fp={s['fp']} fn={s['fn']} tn={s['tn']}  (n={s['n']})")
    if s["guard_examples"]:
        print("GUARD false supersessions (deletes a valid decision):")
        for ex in s["guard_examples"]:
            print(f"   {ex}")
    if s["fn_examples"]:
        print("missed supersessions (recall gap - cross-encoder target):")
        for ex in s["fn_examples"]:
            print(f"   {ex}")
    print(f"\nOVERALL: {'ALL TARGETS MET' if all_ok else 'TARGETS UNMET'}")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Score a supersession predicate against the gold pair fixture.")
    ap.add_argument("--gold", default=str(_DEFAULT_GOLD), help="Gold fixture path")
    ap.add_argument("--predicate", default=None, help="module:function (new_desc, old_desc) -> bool")
    ap.add_argument("--check", action="store_true", help="Exit non-zero if any target is unmet")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    predicate = load_predicate(args.predicate)
    s = score(gold["pairs"], gold["relations"], predicate)
    ok = report(s, gold["targets"])
    return 0 if (ok or not args.check) else 1


if __name__ == "__main__":
    sys.exit(main())
