#!/usr/bin/env python3
"""Extraction quality harness — scores a produced event set against a frozen gold fixture.

This is the regression gate for the extraction pipeline (Stage 2). It measures the
exact failure modes found in the Relay S1 analysis:

  precision        fraction of emitted events that are clean, substantive, correctly-typed facts
  clean_recall     ground-truth facts recalled by a correctly-typed, faithful event   (token + type)
  present_recall   ground-truth facts present at all (even if mistyped/truncated)      (token only)
  hard_miss        ground-truth facts with NO covering event
  noise_rate       emitted events that are false positives (echo/meta/dup/fragment/truncated/mistyped)
  echo/meta/dup/trunc/mistype  the individual false-positive classes

Scoring is token+type based, so it is version-agnostic: it scores the legacy regex
extractor and any future encoder head on the same yardstick. It never trains on
anything — the gold fixture (tests/fixtures/relay_s1_gold.json) is the held-out eval.

Usage:
    python scripts/eval_extraction.py --db <project.db> [--session <id>]
    python scripts/eval_extraction.py --json events.json
    python scripts/eval_extraction.py --db <db> --gold tests/fixtures/relay_s1_gold.json
    python scripts/eval_extraction.py --db <db> --check     # exit 1 if any v1 target unmet

An event is {event_type, text}. With --db, text = payload['description'].
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

_DEFAULT_GOLD = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "relay_s1_gold.json"

_WORD = re.compile(r"[a-z0-9]{3,}")
_TRUNC = re.compile(r"(?:…|\.\.\.)\s*\.?\s*$")          # trailing … or ...
_THREAD_VERB = re.compile(
    r"\b(open|next session|implement|build|work item|work thread|active work|todo|to-do|create|tackle)\b",
    re.IGNORECASE,
)
_REJECT_VERB = re.compile(
    r"\b(reject|ruled out|never use|do not use|don't use|avoid|not used|instead of|too heavy|abandon|no\b)",
    re.IGNORECASE,
)


def normalize(text: str) -> str:
    """Lowercase, NFKC-fold, collapse whitespace. Keeps unicode (10⁻⁸ stays)."""
    t = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", t).strip().lower()


def content_words(text: str) -> int:
    return len(_WORD.findall(text.lower()))


def _sig_match(norm_text: str, sig: dict) -> bool:
    """A signature is {'any': [...]} and/or {'all': [...]} of substrings (normalized)."""
    if not sig:
        return False
    ok = True
    if "all" in sig:
        ok = ok and all(normalize(tok) in norm_text for tok in sig["all"])
    if "any" in sig:
        ok = ok and any(normalize(tok) in norm_text for tok in sig["any"])
    return ok


# ── event loading ─────────────────────────────────────────────────────────────

def load_events_from_db(db_path: str, session_id: str | None) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    q = "select event_type, payload from events"
    args: tuple = ()
    if session_id:
        q += " where session_id = ?"
        args = (session_id,)
    q += " order by id"
    out: list[dict] = []
    for r in conn.execute(q, args):
        try:
            p = json.loads(r["payload"])
            text = p.get("description") or p.get("text") or ""
        except Exception:
            text = str(r["payload"])
        out.append({"event_type": r["event_type"], "text": text})
    conn.close()
    return out


def load_events_from_json(path: str) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [{"event_type": e["event_type"], "text": e.get("text") or e.get("description") or ""} for e in data]


# ── scoring ─────────────────────────────────────────────────────────────────

def classify_false_positive(ev: dict, norm: str, seen: set, gold: dict) -> str | None:
    """Return the FP class for an event, or None if it's a true positive (clean fact)."""
    if norm in seen:
        return "dup"
    for sig in gold["noise_signatures"]["echo"]:
        if normalize(sig) in norm:
            return "echo"
    for sig in gold["noise_signatures"]["meta"]:
        if normalize(sig) in norm:
            return "meta"
    if _TRUNC.search(ev["text"]):
        return "truncated"
    if content_words(norm) < 4:
        return "fragment"
    et = ev["event_type"]
    if et in ("THREAD_OPEN", "THREAD_CLOSE") and not _THREAD_VERB.search(norm):
        return "mistyped_thread"
    if et == "APPROACH_ABANDONED_DO_NOT_RETRY" and not _REJECT_VERB.search(norm):
        return "false_graveyard"
    return None


def score(events: list[dict], gold: dict) -> dict:
    compat = gold["type_compat"]
    facts = gold["ground_truth_facts"]

    norms = [normalize(e["text"]) for e in events]

    # ── recall ──
    present_ids, clean_ids = set(), set()
    for f in facts:
        ctypes = compat[f["kind"]]
        for ev, nt in zip(events, norms):
            if _sig_match(nt, f.get("present", {})):
                present_ids.add(f["id"])
            if _sig_match(nt, f.get("clean", {})) and ev["event_type"] in ctypes:
                clean_ids.add(f["id"])
    n = len(facts)
    miss_ids = [f["id"] for f in facts if f["id"] not in present_ids]

    # ── precision + FP breakdown ──
    fp_counts: dict[str, int] = {}
    tp = 0
    seen: set = set()
    fp_examples: dict[str, str] = {}
    for ev, nt in zip(events, norms):
        cls = classify_false_positive(ev, nt, seen, gold)
        seen.add(nt)
        if cls is None:
            tp += 1
        else:
            fp_counts[cls] = fp_counts.get(cls, 0) + 1
            fp_examples.setdefault(cls, ev["text"][:80])
    total = len(events)
    precision = tp / total if total else 0.0

    return {
        "events": total,
        "precision": precision,
        "tp": tp,
        "fp": total - tp,
        "fp_breakdown": fp_counts,
        "fp_examples": fp_examples,
        "present_recall": len(present_ids) / n,
        "clean_recall": len(clean_ids) / n,
        "hard_miss": len(miss_ids),
        "miss_ids": miss_ids,
        "clean_ids": sorted(clean_ids),
        "noise_rate": (total - tp) / total if total else 0.0,
    }


# ── reporting ─────────────────────────────────────────────────────────────────

def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def report(s: dict, gold: dict) -> bool:
    t = gold["targets_v1"]
    fp = s["fp_breakdown"]
    echo = fp.get("echo", 0); meta = fp.get("meta", 0); dup = fp.get("dup", 0)
    trunc = fp.get("truncated", 0)

    checks = [
        ("precision        ", f"{s['precision']:.0%}", s["precision"] >= t["precision_min"], f">= {t['precision_min']:.0%}"),
        ("clean_recall     ", f"{s['clean_recall']:.0%}", s["clean_recall"] >= t["clean_recall_min"], f">= {t['clean_recall_min']:.0%}"),
        ("hard_miss        ", f"{s['hard_miss']}", s["hard_miss"] <= t["hard_miss_max"], f"<= {t['hard_miss_max']}"),
        ("noise_rate       ", f"{s['noise_rate']:.0%}", s["noise_rate"] <= t["noise_rate_max"], f"<= {t['noise_rate_max']:.0%}"),
        ("echo_events      ", f"{echo}", echo <= t["echo_events"], f"<= {t['echo_events']}"),
        ("meta_events      ", f"{meta}", meta <= t["meta_events"], f"<= {t['meta_events']}"),
        ("dup_events       ", f"{dup}", dup <= t["dup_events"], f"<= {t['dup_events']}"),
        ("truncated_events ", f"{trunc}", trunc <= t["truncated_events"], f"<= {t['truncated_events']}"),
        ("event_count      ", f"{s['events']}", t["event_count_min"] <= s["events"] <= t["event_count_max"], f"{t['event_count_min']}-{t['event_count_max']}"),
    ]

    print(f"\n=== Extraction scorecard — {gold['session_label']} ===")
    print(f"{'metric':<18} {'value':>8}   {'target':<12} result")
    print("-" * 52)
    all_ok = True
    for name, val, ok, tgt in checks:
        all_ok = all_ok and ok
        print(f"{name} {val:>8}   {tgt:<12} {_flag(ok)}")

    print(f"\nrecall detail:  present {s['present_recall']:.0%} ({len(gold['ground_truth_facts'])-s['hard_miss']}/{len(gold['ground_truth_facts'])})"
          f"  clean {s['clean_recall']:.0%}  clean_facts={s['clean_ids']}")
    print(f"hard misses:    {s['miss_ids']}")
    print(f"precision:      tp={s['tp']} fp={s['fp']}  breakdown={s['fp_breakdown']}")
    if s["fp_examples"]:
        print("fp examples:")
        for cls, ex in s["fp_examples"].items():
            print(f"   {cls:<16} {ex!r}")

    b = gold["baseline"]
    print(f"\nhand-scored baseline (analysis): precision {b['hand_precision']:.0%} "
          f"clean_recall {b['hand_clean_recall']:.0%} noise {b['hand_noise_rate']:.0%} "
          f"events {b['events_produced']}")
    print(f"OVERALL: {'ALL TARGETS MET' if all_ok else 'TARGETS UNMET'}")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Score extracted events against the gold fixture.")
    ap.add_argument("--db", help="Path to a memlora project .db")
    ap.add_argument("--session", help="Filter events to this session_id")
    ap.add_argument("--json", help="Path to a JSON list of {event_type, text} events")
    ap.add_argument("--gold", default=str(_DEFAULT_GOLD), help="Gold fixture path")
    ap.add_argument("--check", action="store_true", help="Exit non-zero if any v1 target is unmet")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    if args.db:
        events = load_events_from_db(args.db, args.session)
    elif args.json:
        events = load_events_from_json(args.json)
    else:
        ap.error("one of --db or --json is required")
        return 2

    s = score(events, gold)
    ok = report(s, gold)
    return 0 if (ok or not args.check) else 1


if __name__ == "__main__":
    sys.exit(main())
