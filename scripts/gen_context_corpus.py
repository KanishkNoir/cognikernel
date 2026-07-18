"""Generate context-carrying training data for P2 (role + prev-sentence).

The label-ceiling study isolated two frontiers that CONTEXT disambiguates:
  - meta-framing: "the recall surfaces the earlier decision to use X" is NOISE,
    but only obvious when the PREV line is "let me check what we decided" — the
    same words after "I've weighed the options" would be a DECISION.
  - THREAD's remainder + user-directive NOISE: who is speaking + what preceded.

So this emits {role, prev, current, label, register} — a plausible PRECEDING
line + the target line + the speaker — concentrated on the pairs where prev
flips the label. compose_head_input(current, role, prev) turns each into the
head's training string; the SAME function composes at eval + inference, so the
format can never drift (the CoT-spike lesson).

Cells are (label, register, instruction, [ {role, prev, current} exemplars ]).

Writes: research/train_corpus/context_examples.jsonl
Usage: uv run python scripts/gen_context_corpus.py --per-cell 10 --domains 20
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
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
random.seed(20260708)

CORPUS = Path("research/train_corpus")
CACHE = CORPUS / "context_cache"
API_URL = "https://api.openai.com/v1/chat/completions"

DOMAINS = [
    "fintech payments", "embedded firmware", "game engine", "iOS app",
    "data pipeline", "ML infra", "e-commerce frontend", "healthcare records",
    "IoT fleet", "developer CLI", "compiler tooling", "logistics routing",
    "video streaming", "social feed backend", "search infra", "auth provider",
    "usage billing", "observability stack", "robotics control", "email delivery",
]

_SYSTEM = (
    "You generate realistic 2-line exchanges from an AI-coding-assistant session: a "
    "PREV line (what was just said) and a CURRENT line, each tagged with the speaker "
    "role (user or assistant). The CURRENT line's label depends on the PREV context. "
    "Keep them natural and specific to the project."
)

# The high-value contrastive cells — prev flips or clarifies the label.
CELLS = [
    ("NOISE", "meta_after_recall",
     "CURRENT is meta-commentary that QUOTES/REFERENCES a prior decision (=> NOISE), "
     "made natural by a PREV line about checking memory/recall. role=assistant.",
     [{"role": "assistant", "prev": "Let me check what we already decided about the datastore.",
       "current": "Memory shows we standardized on Postgres over SQLite last session."}]),
    ("DECISION", "decide_after_weighing",
     "Same SURFACE as the meta cell but CURRENT is a NEW decision (=> DECISION), set up "
     "by a PREV line weighing options. role=assistant. Contrast to meta_after_recall.",
     [{"role": "assistant", "prev": "Weighing Postgres vs a document store for this workload.",
       "current": "Going with Postgres — the relational constraints matter more than schema flexibility here."}]),
    ("NOISE", "user_directive",
     "CURRENT is a USER request/directive that names decision-shaped nouns but ASKS for "
     "work (=> NOISE), prev is assistant offering. role=user.",
     [{"role": "assistant", "prev": "I can set up the schema now.",
       "current": "list every hard constraint and which columns must be uuid primary keys"}]),
    ("THREAD", "open_after_plan",
     "CURRENT names concrete UNFINISHED work (=> THREAD), prev sets up the plan/status. "
     "role=assistant.",
     [{"role": "assistant", "prev": "Auth models and hashing are done.",
       "current": "The login endpoint and refresh flow are the pending pieces for next session."}]),
    ("CONSTRAINT_HARD", "rule_after_context",
     "CURRENT is an inviolable rule (=> CONSTRAINT_HARD), prev gives the setup. role=assistant.",
     [{"role": "assistant", "prev": "On storing money values:",
       "current": "amounts are always integer cents — floats never touch money anywhere."}]),
    ("APPROACH_ABANDONED_DO_NOT_RETRY", "reject_after_eval",
     "CURRENT explicitly rejects an approach (=> ABANDONED), prev is the evaluation. role=assistant.",
     [{"role": "assistant", "prev": "We looked at LangChain for the request path.",
       "current": "Rejected — wrong abstraction; we hand-write thin per-provider adapters instead."}]),
    ("NOISE", "question_in_context",
     "CURRENT is a user question (=> NOISE), prev is assistant explaining. role=user.",
     [{"role": "assistant", "prev": "The limiter reads its caps from Redis each window.",
       "current": "wait where does it fall back if redis is down?"}]),
    ("NOISE", "narration_in_context",
     "CURRENT is assistant narration with no durable fact (=> NOISE), prev is user asking. role=assistant.",
     [{"role": "user", "prev": "can you add pagination to the audit log?",
       "current": "Let me look at how the audit endpoint currently returns rows before changing it."}]),
]


def _key() -> str:
    ef = Path(".env")
    if ef.exists():
        for line in ef.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENAI_API_KEY"):
                return line.partition("=")[2].strip().strip('"').strip("'")
    return os.environ.get("OPENAI_API_KEY", "")


def _call(model: str, user: str, ck: str) -> dict | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    cf = CACHE / f"{ck}.json"
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))
    body = json.dumps({"model": model, "messages": [
        {"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        "temperature": 1.0, "response_format": {"type": "json_object"}}).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "Authorization": f"Bearer {_key()}", "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                content = json.loads(json.loads(r.read())["choices"][0]["message"]["content"])
            cf.write_text(json.dumps(content, ensure_ascii=False), encoding="utf-8")
            return content
        except urllib.error.HTTPError as exc:
            if exc.code != 429 and 400 <= exc.code < 500:
                sys.exit(f"API {exc.code}: {exc.read().decode()[:200]}")
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--domains", type=int, default=20)
    args = ap.parse_args()
    domains = random.sample(DOMAINS, min(args.domains, len(DOMAINS)))

    out = []
    calls = 0
    for label, register, instruction, exemplars in CELLS:
        for domain in domains:
            user = (
                f"Generate {args.per_cell} 2-line exchanges for a {domain} project.\n"
                f"CURRENT line label: {label}. {instruction}\n"
                f"Style example (different domain, do NOT copy): {json.dumps(exemplars)}\n"
                'Return JSON: {"items": [{"role": "user|assistant", "prev": "...", "current": "..."}, ...]}'
            )
            ck = hashlib.sha256(f"{args.model}|{label}|{register}|{domain}|{args.per_cell}".encode()).hexdigest()[:20]
            res = _call(args.model, user, ck)
            calls += 1
            for it in (res or {}).get("items", []):
                if isinstance(it, dict) and it.get("current") and it.get("prev"):
                    out.append({
                        "text": str(it["current"]).strip(), "prev": str(it["prev"]).strip(),
                        "role": str(it.get("role", "assistant")).strip().lower(),
                        "label": label, "register": f"ctx_{register}",
                        "domain": domain, "source": f"context:{args.model}",
                    })
    with open(CORPUS / "context_examples.jsonl", "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(out)} context examples ({calls} calls)")
    print("  by label:", dict(Counter(i["label"] for i in out)))
    print("  by role: ", dict(Counter(i["role"] for i in out)))


if __name__ == "__main__":
    main()
