"""Estimate the label-ceiling: inter-annotator agreement on the frozen eval (P0).

The teacher (gpt-4o-mini) independently re-labels every eval sentence with NO
access to the gold label; agreement with the curated gold is a proxy for the
human-agreement ceiling. If two reasonable annotators only agree ~X% on the
DECISION↔CONSTRAINT boundary, a model scoring X% has hit the ceiling and chasing
higher is fighting label noise, not model capacity.

Reports overall agreement, Cohen's kappa, per-class agreement, and the specific
ambiguous confusions (DECISION↔CONSTRAINT_HARD, signal↔NOISE).

Usage: uv run python scripts/label_ceiling.py [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path("research/model_eval/salience_eval.jsonl")
CACHE = Path("research/model_eval/ceiling_cache")
API_URL = "https://api.openai.com/v1/chat/completions"
LABELS = ("NOISE", "DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
          "APPROACH_ABANDONED_DO_NOT_RETRY", "THREAD")

_SYSTEM = (
    "You are an expert annotator labeling ONE sentence from an AI-coding-assistant "
    "session into exactly one category.\n"
    "- DECISION: a concrete choice made (tool/value/design).\n"
    "- CONSTRAINT_HARD: an inviolable rule (must/never/always; security/correctness).\n"
    "- CONSTRAINT_SOFT: a convention or style preference.\n"
    "- APPROACH_ABANDONED_DO_NOT_RETRY: an approach explicitly rejected/ruled out.\n"
    "- THREAD: open/ongoing work — in progress or planned next.\n"
    "- NOISE: narration, a question, an explanation, an acknowledgment, or a meta "
    "reference that merely quotes a prior decision.\n"
    'Return JSON {"label": "<one>"}.'
)
_ALIAS = {"ABANDONED": "APPROACH_ABANDONED_DO_NOT_RETRY", "APPROACH_ABANDONED": "APPROACH_ABANDONED_DO_NOT_RETRY",
          "THREAD_OPEN": "THREAD", "THREAD_CLOSE": "THREAD", "HARD": "CONSTRAINT_HARD", "SOFT": "CONSTRAINT_SOFT"}


def _key() -> str:
    ef = Path(".env")
    if ef.exists():
        for line in ef.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENAI_API_KEY"):
                return line.partition("=")[2].strip().strip('"').strip("'")
    return os.environ.get("OPENAI_API_KEY", "")


def annotate(model: str, text: str) -> str | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{hashlib.sha256((model+text).encode()).hexdigest()[:20]}.json"
    if cf.exists():
        lab = json.loads(cf.read_text(encoding="utf-8")).get("label", "")
    else:
        body = json.dumps({"model": model, "messages": [
            {"role": "system", "content": _SYSTEM}, {"role": "user", "content": f"Sentence: {text}"}],
            "temperature": 0, "response_format": {"type": "json_object"}}).encode()
        req = urllib.request.Request(API_URL, data=body, headers={
            "Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    content = json.loads(json.loads(r.read())["choices"][0]["message"]["content"])
                cf.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
                lab = content.get("label", "")
                break
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and 400 <= exc.code < 500:
                    sys.exit(f"API {exc.code}: {exc.read().decode()[:200]}")
                time.sleep(2 * (attempt + 1))
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            return None
    lab = str(lab).strip().upper().replace(" ", "_")
    lab = _ALIAS.get(lab, lab)
    return lab if lab in LABELS else "NOISE"


def _kappa(y1, y2) -> float:
    n = len(y1)
    po = sum(a == b for a, b in zip(y1, y2)) / n
    c1, c2 = Counter(y1), Counter(y2)
    pe = sum((c1[l] / n) * (c2[l] / n) for l in LABELS)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    items = [json.loads(l) for l in DATA.read_text(encoding="utf-8").splitlines()]
    if args.limit:
        items = items[: args.limit]

    gold, llm = [], []
    per_class = defaultdict(lambda: [0, 0])
    confusion = Counter()
    for i, it in enumerate(items):
        a = annotate(args.model, it["text"])
        if a is None:
            continue
        g = it["label"]
        gold.append(g); llm.append(a)
        per_class[g][1] += 1
        if a == g:
            per_class[g][0] += 1
        else:
            confusion[(g, a)] += 1
        if (i + 1) % 80 == 0:
            print(f"  {i+1}/{len(items)}", flush=True)

    n = len(gold)
    agree = sum(a == b for a, b in zip(gold, llm)) / n
    # NOISE-vs-signal binary agreement (the capture boundary)
    gb = [("SIG" if x != "NOISE" else "N") for x in gold]
    lb = [("SIG" if x != "NOISE" else "N") for x in llm]
    bin_agree = sum(a == b for a, b in zip(gb, lb)) / n

    print(f"\n=== label ceiling (teacher vs curated gold, n={n}) ===")
    print(f"  overall agreement : {agree:.3f}   (a model at ~{agree:.2f} has hit the human ceiling)")
    print(f"  Cohen's kappa     : {_kappa(gold, llm):.3f}")
    print(f"  signal-vs-NOISE   : {bin_agree:.3f}  (the capture boundary)")
    print("  per-class agreement:")
    for lab in LABELS:
        c, t = per_class[lab]
        print(f"    {lab:34} {c}/{t} = {c/max(1,t):.2f}")
    print("  top disagreements:")
    for (g, a), c in confusion.most_common(8):
        print(f"    gold {g} -> teacher {a}: {c}")
    Path("research/model_eval/label_ceiling.json").write_text(json.dumps({
        "n": n, "agreement": round(agree, 3), "kappa": round(_kappa(gold, llm), 3),
        "signal_vs_noise": round(bin_agree, 3),
        "per_class": {l: {"agree": round(c / max(1, t), 3), "n": t} for l, (c, t) in per_class.items()},
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
