#!/usr/bin/env python3
"""Generate a HARD-NEGATIVE TWIN corpus for the salience/type head via OpenAI (WS-A1).

The frozen-backbone + linear head confuses the classes that are *semantically
adjacent*: hard vs soft constraint, decision vs rejection, decision vs constraint,
and real-fact vs noise. Independent per-class examples (gen_training_data.py) don't
teach those boundaries — the model needs MINIMAL PAIRS that share the topic
vocabulary and differ ONLY along the confused axis. Those shared-topic twins are the
hard negatives a contrastive fine-tune (SetFit, WS-B2) needs to separate the geometry.

Each generated "twin set" is about ONE topic; we emit each member as a {text, label}
row (the same format train_salience_head.py reads), so the corpus drops straight into
training alongside the hand seeds + gen_training_data.py output.

Axes (the confusions we measured):
    strength    CONSTRAINT_HARD  vs CONSTRAINT_SOFT             (deontic strength)
    polarity    DECISION         vs APPROACH_ABANDONED_DO_NOT_RETRY (adopt vs reject)
    modality    DECISION         vs CONSTRAINT_HARD             (a choice vs an invariant)
    salience_d  DECISION         vs NOISE                       (fact vs question/musing)
    salience_c  CONSTRAINT_HARD  vs NOISE                       (rule vs narration)
    thread      THREAD           vs NOISE                       (open work vs status chatter)

Global generalization: the same broad domain pool as the pairwise generator — the
model must learn the axis, not a stack. No eval leakage: drops anything matching the
Relay S1 baseline events.

Output: tests/fixtures/salience_twins_generated.jsonl  ({text, label} / line).

Usage:
    python scripts/gen_twin_corpus.py [--per 180] [--model gpt-4o-mini]
        [--chunk 10] [--temperature 0.9] [--only strength,polarity]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import unicodedata
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_OUT = _ROOT / "tests" / "fixtures" / "salience_twins_generated.jsonl"
_S1_EVENTS = _ROOT / "tests" / "fixtures" / "relay_s1_baseline_events.json"

# Reuse the broad GLOBAL pool (kept in sync with gen_supersession_pairs.py).
DOMAINS = [
    "a Python FastAPI + PostgreSQL backend",
    "a TypeScript React single-page app",
    "a React Native mobile app with offline sync",
    "a native iOS app in Swift with Core Data",
    "an Android app in Kotlin with Jetpack Compose",
    "a Go microservice with gRPC and Redis",
    "a Rust systems daemon with tokio",
    "a Rust embedded firmware project (no_std)",
    "a C++ game engine with an ECS",
    "a Unity game in C#",
    "a Java Spring Boot enterprise service",
    "a .NET Core Web API",
    "a Ruby on Rails monolith",
    "an Elixir Phoenix realtime app",
    "a PHP Laravel e-commerce site",
    "a Node.js Express REST API",
    "a Next.js full-stack app on serverless",
    "a Python data-engineering ETL pipeline with Airflow",
    "a PyTorch model-training repo",
    "an MLOps inference-serving platform",
    "a Kubernetes platform / Helm + Terraform IaC repo",
    "a CLI tool distributed as a single binary",
    "a cross-platform desktop app in Electron",
    "a Flutter mobile app",
    "a Solidity smart-contract project",
    "a browser extension (Manifest V3)",
    "a scientific-computing library in Julia",
    "a Haskell backend service",
    "an Android/Kotlin + Go full-stack fintech app",
    "a SwiftUI macOS menu-bar utility",
    "a streaming data platform with Kafka and Flink",
    "a static-site generator written in Rust",
]

# Aspect pool — sampled per chunk to force TOPICAL spread. Without it the model
# gravitates to a few defaults (auth, ORMs, timeouts); facets push the corpus across
# the whole surface of software work so the head generalizes to any project.
FACETS = [
    "data modeling / schema / migrations",
    "authentication / authorization / sessions",
    "API design / endpoints / versioning",
    "error handling / retries / timeouts",
    "logging / metrics / tracing / observability",
    "configuration / secrets / environment variables",
    "testing / CI / release process",
    "performance / caching / memory usage",
    "concurrency / async / threading / locking",
    "security / encryption / input validation",
    "storage / files / serialization formats",
    "networking / protocols / messaging",
    "build system / packaging / dependency management",
    "deployment / infrastructure / scaling",
    "UI / state management / rendering",
    "internationalization / accessibility",
    "data pipelines / batch vs streaming",
    "model training / evaluation / inference",
    "billing / payments / rate limiting / quotas",
    "background jobs / scheduling / queues",
    "code style / naming / project layout",
    "third-party integrations / webhooks / APIs",
]

# Each axis: the two labels to emit (the JSON keys the model returns) + how to write
# the twin so the members share a topic but differ only along the axis.
AXES: dict[str, dict] = {
    "strength": {
        "target": 200,
        "labels": ["CONSTRAINT_HARD", "CONSTRAINT_SOFT"],
        "instruction": (
            "Pick one technical topic. Write CONSTRAINT_HARD: a non-negotiable rule "
            "(must / never / always / cannot) about it. Write CONSTRAINT_SOFT: a softer "
            "preference (prefer / should / ideally / by default) about the SAME topic. "
            "They must share the topic vocabulary and differ ONLY in deontic strength."
        ),
    },
    "polarity": {
        "target": 180,
        "labels": ["DECISION", "APPROACH_ABANDONED_DO_NOT_RETRY"],
        "instruction": (
            "Pick one technology/library/approach. Write DECISION: a sentence that "
            "ADOPTS it as the chosen option. Write APPROACH_ABANDONED_DO_NOT_RETRY: a "
            "sentence that RULES IT OUT / abandoned it / will not use it. Name the SAME "
            "technology in both."
        ),
    },
    "modality": {
        "target": 150,
        "labels": ["DECISION", "CONSTRAINT_HARD"],
        "instruction": (
            "Pick one topic. Write DECISION: a concrete choice the team made about it. "
            "Write CONSTRAINT_HARD: an invariant that must hold about the SAME topic. "
            "The decision is a chosen option; the constraint is a rule."
        ),
    },
    "salience_d": {
        "target": 150,
        "labels": ["DECISION", "NOISE"],
        "instruction": (
            "Pick one topic. Write DECISION: a clean durable decision about it. Write "
            "NOISE: a sentence about the SAME topic that is NOT a durable fact — a "
            "question, a hypothetical ('if ...'), first-person narration ('let me ...'), "
            "or a tradeoff musing. Same topic vocabulary so it's a real hard negative."
        ),
    },
    "salience_c": {
        "target": 120,
        "labels": ["CONSTRAINT_HARD", "NOISE"],
        "instruction": (
            "Pick one topic. Write CONSTRAINT_HARD: a real invariant about it. Write "
            "NOISE: a sentence about the SAME topic that only NARRATES or asks (status "
            "chatter, a question, a section header, a code-doc description) — not a rule."
        ),
    },
    "thread": {
        "target": 120,
        "labels": ["THREAD", "NOISE"],
        "instruction": (
            "Pick a work area. Write THREAD: an OPEN next step / active work item to "
            "carry forward ('still need to ...', 'next we will ...'). Write NOISE: a "
            "sentence about the same area that is COMPLETED-work mention or status "
            "chatter ('all tests pass', 'nothing to do here') — not an open thread."
        ),
    },
}

_SYS = (
    "You generate MINIMAL-PAIR training sentences for a classifier that types sentences "
    "pulled from an AI-coding-assistant transcript. Each twin set is about ONE topic and "
    "contains one sentence per requested class, sharing the topic's vocabulary but "
    "differing only along the requested distinction. Use real library names, numbers, and "
    "env vars. Output ONLY valid JSON. Vary phrasing and length. No near-duplicates."
)


def _load_api_key() -> str:
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


def gen_for_axis(client, model, axis, spec, chunk, temperature, excluded) -> list[tuple[str, str]]:
    """Return [(text, label)] rows for one axis (target counts whole twin sets)."""
    labels = spec["labels"]
    target_sets = spec["target"]
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    n_sets = 0
    safety = 0
    while n_sets < target_sets and safety < 400:
        safety += 1
        domain = random.choice(DOMAINS)
        facet = random.choice(FACETS)
        n = min(chunk, target_sets - n_sets + 3)
        shape = ", ".join(f'"{l}": "..."' for l in labels)
        user = (
            f"Domain context: {domain}.\n"
            f"Focus this batch on the aspect: {facet}.\n"
            f"Generate {n} distinct twin sets for axis '{axis}'.\n"
            f"{spec['instruction']}\n"
            "Make them VERY diverse: each set a DIFFERENT specific topic within the "
            "aspect, and vary sentence length and register (terse vs detailed).\n"
            f'Return JSON: {{"sets": [{{{shape}}}, ...]}}'
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
            sets = data.get("sets") or []
        except Exception as exc:  # noqa: BLE001
            print(f"  [{axis}] call failed: {exc}", file=sys.stderr)
            continue
        for st in sets:
            if not isinstance(st, dict):
                continue
            members = []
            ok = True
            for lbl in labels:
                txt = (st.get(lbl) or "").strip()
                key = _norm(txt)
                if not txt or key in seen or key in excluded:
                    ok = False
                    break
                members.append((txt, lbl, key))
            if not ok:
                continue
            for txt, lbl, key in members:
                seen.add(key)
                rows.append((txt, lbl))
            n_sets += 1
        print(f"  [{axis}] {n_sets}/{target_sets} sets", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--out", default=str(_OUT))
    ap.add_argument("--per", type=int, default=None, help="override target twin SETS per axis")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--only", help="comma-separated subset of axes")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    key = _load_api_key()
    if not key:
        print("OPENAI_API_KEY not set (env or .env)", file=sys.stderr)
        return 2
    import httpx
    from openai import OpenAI

    client = OpenAI(api_key=key, http_client=httpx.Client(timeout=60.0))
    excluded = _excluded_texts()
    only = set(args.only.split(",")) if args.only else None

    rows: list[tuple[str, str]] = []
    for axis, spec in AXES.items():
        if only and axis not in only:
            continue
        if args.per is not None:
            spec = {**spec, "target": args.per}
        print(f"generating {axis} (target {spec['target']} sets)...", file=sys.stderr)
        rows.extend(gen_for_axis(client, args.model, axis, spec, args.chunk, args.temperature, excluded))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with out.open(mode, encoding="utf-8") as f:
        for text, label in rows:
            f.write(json.dumps({"text": text, "label": label}, ensure_ascii=False) + "\n")

    try:
        disp = out.relative_to(_ROOT)
    except ValueError:
        disp = out
    print(f"\n{'appended' if args.append else 'wrote'} {len(rows)} rows to {disp}")
    from collections import Counter
    for lbl, c in Counter(l for _, l in rows).most_common():
        print(f"  {lbl:<32} {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
