#!/usr/bin/env python3
"""Generate synthetic training data for the v1 salience+type head via the OpenAI API.

Why synthetic (not labeling real transcripts):
  - No eval leakage — the model writes fresh sentences, never the held-out Relay S1
    gold (we also exclude any exact match as a belt-and-suspenders guard).
  - No privacy exposure — your private transcripts are never sent to OpenAI.
  - Balance + diversity — we control the per-class counts and span many domains,
    and we explicitly enumerate the NOISE taxonomy that real prose is full of and
    a small hand-seed set under-represents (the cause of the all-sentence
    over-acceptance we saw).

Output: tests/fixtures/salience_train_generated.jsonl  ({text, label} per line),
which scripts/train_salience_head.py reads alongside the hand seeds.

Usage:
    OPENAI_API_KEY=...  python scripts/gen_training_data.py
        [--model gpt-4o-mini] [--out tests/fixtures/salience_train_generated.jsonl]
        [--temperature 0.9] [--chunk 30]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "tests" / "fixtures" / "salience_train_generated.jsonl"
_S1_EVENTS = _ROOT / "tests" / "fixtures" / "relay_s1_baseline_events.json"

DOMAINS = [
    "a FastAPI + PostgreSQL + React task-management app with JWT auth",
    "a self-hosted multi-provider LLM gateway / proxy",
    "a typed agent-tooling SDK monorepo (registry, retries, MCP)",
    "a RAG ingestion + retrieval pipeline with a vector DB",
    "a durable backend-automation pipeline (webhooks, outbox, retries, DLQ)",
    "a Go microservice with gRPC and Redis",
    "a data-engineering ETL pipeline with Airflow and Parquet",
    "a mobile app backend with push notifications and billing",
]

# Per-class generation spec: a crisp definition + what to avoid, so the model
# produces on-distribution examples and the classes stay separable.
SPECS: dict[str, dict] = {
    "DECISION": {
        "target": 150,
        "definition": (
            "a concrete technical CHOICE that was made — a library, value, scheme, "
            "default, format, or approach the team adopted. One declarative sentence."
        ),
        "avoid": "questions, options still being weighed, or hard 'must/never' rules (those are constraints).",
    },
    "CONSTRAINT_HARD": {
        "target": 150,
        "definition": (
            "a hard, non-negotiable requirement or invariant — phrased with must / "
            "never / always / cannot, often about security, correctness, or money. "
            "One declarative sentence."
        ),
        "avoid": "soft preferences ('prefer', 'should'), choices, or questions.",
    },
    "CONSTRAINT_SOFT": {
        "target": 130,
        "definition": (
            "a softer convention, preference, or default — phrased with prefer / "
            "should / generally / by default. A style or organizational guideline."
        ),
        "avoid": "hard 'must/never' invariants, one-off choices of a specific library, or questions.",
    },
    "APPROACH_ABANDONED_DO_NOT_RETRY": {
        "target": 110,
        "definition": (
            "an explicit REJECTION of an approach — something tried and dropped, or "
            "ruled out, with or without a reason. Phrased with rejected / ruled out / "
            "abandoned / will not use / not worth it."
        ),
        "avoid": "neutral choices, or merely mentioning a technology without rejecting it.",
    },
    "THREAD": {
        "target": 90,
        "definition": (
            "an active WORK ITEM or next step to carry forward — 'next we'll…', "
            "'open a thread to…', 'still need to…', 'the active task is…'."
        ),
        "avoid": "completed work, decisions, or constraints.",
    },
    "NOISE": {
        "target": 320,
        "definition": (
            "NOT a durable memory fact. Generate a realistic MIX across ALL of these "
            "sub-types, roughly evenly:\n"
            "  (a) user questions / instructions ('Design the schema and list the constraints.')\n"
            "  (b) first-person assistant narration ('Let me read the router first.', \"I'll add a test.\")\n"
            "  (c) hypothetical examples ('If the caller sets a 30s timeout, the retry never fires.')\n"
            "  (d) justifications / tradeoff musings ('It's the right tradeoff because the read is rare.')\n"
            "  (e) meta-talk about the session / memory ('Here is the record for the Stop hook to capture.')\n"
            "  (f) code-doc descriptions ('NonRetryableError(status) signals the router to stop.')\n"
            "  (g) section headers / labels ('Proposed Stack and Hard Constraints.')\n"
            "  (h) sentence fragments ('Never retrievable after that.', 'No discipline required.')\n"
            "  (i) status chatter ('All eighteen tests pass.', 'Nothing to do here.')\n"
            "  (j) transitional / framing sentences ('Below is the serialization function.')"
        ),
        "avoid": "anything that is actually a clean decision, constraint, rejection, or work item.",
    },
}

_SYS = (
    "You generate labeled training sentences for a classifier that decides whether a "
    "single sentence pulled from an AI-coding-assistant transcript is a durable project-"
    "memory fact, and of what type. Output ONLY valid JSON. Sentences must be realistic, "
    "varied in phrasing and length, and specific (real library names, numbers, env vars). "
    "Do not number them. Do not repeat near-identical sentences."
)


def _load_api_key() -> str:
    """Prefer the project .env (the user's real key) over a stale process env var."""
    env_file = _ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY"):
                _, _, val = line.partition("=")
                val = val.strip().strip('"').strip("'")
                if val:
                    return val
    return (os.environ.get("OPENAI_API_KEY") or "").strip()


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", t or "")).strip().lower()


def _excluded_texts() -> set[str]:
    out: set[str] = set()
    try:
        for e in json.loads(_S1_EVENTS.read_text(encoding="utf-8")):
            out.add(_norm(e.get("text", "")))
    except Exception:
        pass
    return out


def gen_for_class(client, model, label, spec, chunk, temperature) -> list[str]:
    target = spec["target"]
    got: list[str] = []
    seen: set[str] = set()
    di = 0
    safety = 0
    while len(got) < target and safety < 200:
        safety += 1
        domain = DOMAINS[di % len(DOMAINS)]
        di += 1
        n = min(chunk, target - len(got) + 5)
        user = (
            f"Domain context: {domain}.\n"
            f"Generate {n} distinct example sentences of class {label}.\n"
            f"Class {label} means: {spec['definition']}\n"
            f"Avoid: {spec['avoid']}\n"
            'Return JSON: {"examples": ["...", "..."]}'
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": _SYS},
                          {"role": "user", "content": user}],
            )
            data = json.loads(resp.choices[0].message.content)
            items = data.get("examples") or data.get("sentences") or []
        except Exception as exc:  # noqa: BLE001
            print(f"  [{label}] call failed: {exc}", file=sys.stderr)
            continue
        for s in items:
            if not isinstance(s, str):
                continue
            s = s.strip()
            key = _norm(s)
            if not key or key in seen:
                continue
            seen.add(key)
            got.append(s)
        print(f"  [{label}] {len(got)}/{target}", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return got[:target]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--out", default=str(_OUT))
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--only", help="comma-separated subset of classes to (re)generate")
    args = ap.parse_args()

    key = _load_api_key()
    if not key:
        print("OPENAI_API_KEY not set (env or .env)", file=sys.stderr)
        return 2
    import httpx
    from openai import OpenAI

    # Pass a clean httpx client: some openai/httpx version combos crash because the
    # SDK forwards a removed `proxies=` kwarg when it builds its own client.
    client = OpenAI(api_key=key, http_client=httpx.Client(timeout=60.0))
    excluded = _excluded_texts()
    only = set(args.only.split(",")) if args.only else None

    rows: list[tuple[str, str]] = []
    for label, spec in SPECS.items():
        if only and label not in only:
            continue
        print(f"generating {label} (target {spec['target']})…", file=sys.stderr)
        for text in gen_for_class(client, args.model, label, spec, args.chunk, args.temperature):
            if _norm(text) in excluded:
                continue
            rows.append((text, label))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for text, label in rows:
            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")

    print(f"\nwrote {len(rows)} examples to {out.relative_to(_ROOT)}")
    from collections import Counter
    for lbl, c in Counter(l for _, l in rows).most_common():
        print(f"  {lbl:<32} {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
