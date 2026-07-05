"""Build chain-of-thought SFT data to instruction-tune LFM2.5-230M (Spike B).

Zero-shot the 230M model can't map to our 6-way taxonomy — but it UNDERSTANDS
(got the meta-framing cases right). CoT fine-tuning is the fix: teach it to
REASON to the label. Reasoning-distillation — a strong teacher (gpt-4o-mini)
writes a short <thought> justifying each GOLD label, so the small model learns
the teacher's reasoning, not just the answer.

Target format (user's <thought><action> structure):
  user:      "Sentence: <text>"
  assistant: "<thought><one-sentence rationale></thought><action><LABEL></action>"

Source: the labeled training corpus (NOT the frozen eval — decontaminated
already). Balanced per label so minority classes get real CoT coverage.

Reads : research/train_corpus/train_sentences_hum.jsonl (or --corpus)
Writes: research/train_corpus/cot_sft.jsonl  {messages:[user, assistant]}

Usage: uv run python scripts/build_cot_sft.py [--per-label 200] [--per-batch 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
random.seed(1106)

CORPUS = Path("research/train_corpus")
CACHE_DIR = CORPUS / "cot_cache"
API_URL = "https://api.openai.com/v1/chat/completions"
LABELS = ("NOISE", "DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
          "APPROACH_ABANDONED_DO_NOT_RETRY", "THREAD")

_DEF = {
    "DECISION": "a concrete choice made for the project (tool, value, design)",
    "CONSTRAINT_HARD": "an inviolable rule/invariant (must/never/always; security or correctness)",
    "CONSTRAINT_SOFT": "a convention or preference (naming, style, layout)",
    "APPROACH_ABANDONED_DO_NOT_RETRY": "an approach explicitly rejected / ruled out",
    "THREAD": "open or ongoing work — in progress or planned next",
    "NOISE": "not a durable project fact — narration, a question, an explanation, an "
             "acknowledgment, or a META reference that merely quotes/mentions a prior decision",
}
_SYSTEM = (
    "You write ONE short sentence of reasoning (first-person analyst thinking aloud) that "
    "explains why a given sentence has its assigned label, citing the distinguishing cue. "
    "Be concise and concrete; do not restate the whole sentence. Return JSON {\"thought\": \"...\"}."
)


def _api_key() -> str:
    ef = Path(".env")
    if ef.exists():
        for line in ef.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENAI_API_KEY"):
                _, _, v = line.partition("=")
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    return os.environ.get("OPENAI_API_KEY", "")


def _call(model: str, user: str, ck: str) -> dict | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cf = CACHE_DIR / f"{ck}.json"
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))
    body = json.dumps({
        "model": model, "messages": [{"role": "system", "content": _SYSTEM},
                                     {"role": "user", "content": user}],
        "temperature": 0.7, "response_format": {"type": "json_object"},
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
                sys.exit(f"OpenAI API {exc.code}: {exc.read().decode()[:200]}")
            time.sleep(2.0 * (attempt + 1))
        except Exception:
            time.sleep(2.0 * (attempt + 1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--corpus", default="research/train_corpus/train_sentences_hum.jsonl")
    ap.add_argument("--per-label", type=int, default=200)
    ap.add_argument("--per-batch", type=int, default=8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.corpus).read_text(encoding="utf-8").splitlines()]
    by_label: dict[str, list] = defaultdict(list)
    for r in rows:
        if r.get("label") in LABELS and r.get("text"):
            by_label[r["label"]].append(r["text"])
    sample: list[tuple[str, str]] = []
    for lab in LABELS:
        texts = by_label[lab]
        random.shuffle(texts)
        for t in texts[: args.per_label]:
            sample.append((t, lab))
    random.shuffle(sample)
    print(f"building CoT for {len(sample)} examples "
          f"({ {l: min(len(by_label[l]), args.per_label) for l in LABELS} })")

    out = []
    calls = 0
    for i in range(0, len(sample), args.per_batch):
        batch = sample[i:i + args.per_batch]
        for text, label in batch:
            user = (f"Label = {label} ({_DEF[label]}).\nSentence: {text}\n"
                    f"Write the one-sentence reasoning for why this is {label}.")
            ck = hashlib.sha256(f"{args.model}|{label}|{text}".encode()).hexdigest()[:20]
            res = _call(args.model, user, ck)
            calls += 1
            thought = (res or {}).get("thought", "").strip()
            if not thought:
                thought = f"This reads as {label.lower().replace('_', ' ')}."
            assistant = f"<thought>{thought}</thought><action>{label}</action>"
            out.append({"messages": [
                {"role": "user", "content": f"Sentence: {text}"},
                {"role": "assistant", "content": assistant},
            ]})
        if (i // args.per_batch) % 20 == 0:
            print(f"  {i+len(batch)}/{len(sample)}", flush=True)

    with open(CORPUS / "cot_sft.jsonl", "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(out)} CoT SFT examples ({calls} teacher calls) -> cot_sft.jsonl")
    print("sample:", out[0]["messages"][1]["content"][:160])


if __name__ == "__main__":
    main()
