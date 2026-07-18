"""Humanize clean synthetic sentences into messy real-developer register (Spike A1).

Research: templated/clean synthetic data yields NARROW generalization, and our
adversarial retrain failed to move memory_meta because synthetic shapes don't
match real transcripts. Rather than (only) label real data, cheaply close the
synthetic->real gap: rewrite each clean synthetic sentence into how a developer
ACTUALLY types — lowercase, typos, shorthand, fragments, filler, mid-thought
interruptions — while PRESERVING the exact decision/fact and therefore its label.

A DECISION rewritten casually is still a DECISION; NOISE stays NOISE. The label
rides along unchanged; only the surface register changes.

Reads : research/train_corpus/synth_sentences.jsonl  (clean synthetic)
Writes: research/train_corpus/humanized_sentences.jsonl  {text,label,register,domain,source}

Only the "too-clean" registers are humanized (plain/label_value/table_row/
explanation/narration/instruction and the present-tense adversarial ones); the
casual_* registers are already messy and pass through unchanged.

Usage: uv run python scripts/humanize_corpus.py [--per-batch 10] [--cap-per-cell 60]
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
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CORPUS = Path("research/train_corpus")
CACHE_DIR = CORPUS / "humanize_cache"
API_URL = "https://api.openai.com/v1/chat/completions"

# Registers already in natural/messy register — skip (pass through unchanged).
_ALREADY_MESSY = {"casual_chat", "casual_update", "casual", "question", "ack"}

_SYSTEM = (
    "You rewrite sentences from AI-coding-assistant sessions into how a real "
    "developer actually types in chat: often lowercase, occasional typos or "
    "shorthand (fn, btw, pls, prob, config), dropped punctuation, fragments, "
    "filler ('ok so', 'yeah', 'hmm'), sometimes mid-thought. CRITICAL: preserve "
    "the exact technical decision/fact/intent and its polarity — do NOT add, "
    "drop, or soften any decision, constraint, rejection, or value. A rule stays "
    "a rule; a rejection stays a rejection. Only the surface style changes."
)


def _api_key() -> str:
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENAI_API_KEY"):
                _, _, v = line.partition("=")
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    return os.environ.get("OPENAI_API_KEY", "")


def _call(model: str, user: str, cache_key: str) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cf = CACHE_DIR / f"{cache_key}.json"
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        "temperature": 1.0,
        "response_format": {"type": "json_object"},
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read())
            content = json.loads(resp["choices"][0]["message"]["content"])
            cf.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
            return content
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and 400 <= exc.code < 500:
                sys.exit(f"OpenAI API {exc.code}: {exc.read().decode()[:300]}")
            time.sleep(2.0 * (attempt + 1))
        except Exception as exc:
            print(f"    api error ({exc}); retry")
            time.sleep(2.0 * (attempt + 1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--per-batch", type=int, default=10)
    ap.add_argument("--cap-per-cell", type=int, default=60,
                    help="max sentences humanized per (label,register) — keeps the spike bounded")
    args = ap.parse_args()

    src = [json.loads(l) for l in (CORPUS / "synth_sentences.jsonl").read_text(encoding="utf-8").splitlines()]
    # Pass messy registers through unchanged; humanize the rest, capped per cell.
    passthrough = [s for s in src if s["register"] in _ALREADY_MESSY]
    to_hum: list[dict] = []
    per_cell: dict[tuple, int] = defaultdict(int)
    for s in src:
        if s["register"] in _ALREADY_MESSY:
            continue
        key = (s["label"], s["register"])
        if per_cell[key] < args.cap_per_cell:
            per_cell[key] += 1
            to_hum.append(s)

    out: list[dict] = list(passthrough)  # keep the already-messy ones
    calls = 0
    for i in range(0, len(to_hum), args.per_batch):
        batch = to_hum[i:i + args.per_batch]
        numbered = "\n".join(f"{j+1}. {b['text']}" for j, b in enumerate(batch))
        user = (
            f"Rewrite each numbered sentence in messy real-developer chat style, "
            f"preserving the exact decision/fact/polarity.\n{numbered}\n"
            f'Return JSON: {{"items": ["rewrite 1", "rewrite 2", ...]}} in the same order, '
            f"same count ({len(batch)})."
        )
        ck = hashlib.sha256(f"{args.model}|{numbered}".encode()).hexdigest()[:20]
        res = _call(args.model, user, ck)
        calls += 1
        items = (res or {}).get("items", [])
        for b, rew in zip(batch, items):
            if isinstance(rew, str) and 10 <= len(rew) <= 400:
                out.append({"text": rew.strip(), "label": b["label"],
                            "register": f"hum_{b['register']}", "domain": b.get("domain", ""),
                            "source": f"humanized:{args.model}"})
    with open(CORPUS / "humanized_sentences.jsonl", "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"humanized {len(to_hum)} clean sentences in {calls} calls; "
          f"{len(passthrough)} already-messy passed through")
    print(f"total humanized_sentences.jsonl: {len(out)}")
    print("  by label:", dict(Counter(i['label'] for i in out)))


if __name__ == "__main__":
    main()
