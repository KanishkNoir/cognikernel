#!/usr/bin/env python3
"""Generate a synthetic PAIRWISE supersession corpus via the OpenAI API (WS-A2).

Feeds the cross-encoder supersession head (WS-C1). Each row is a pair of memory
descriptions plus the relation the NEWER `a` bears to the OLDER `b`:

    supersedes  — same topic, the CHOICE changed (correction). a replaces b.
    refines     — same rule, an updated / more-specific value. a replaces b.
    subsumes    — a is a general rule that absorbs the specific b. a replaces b.
    contradicts — a directly negates b (polarity flip). most-recent wins.
    unrelated   — DIFFERENT concerns; superseding would DELETE a valid decision.

Design goals (match the rest of the sprint):
  - GLOBAL generalization. Pairs are drawn over a broad pool of languages, project
    types, and domains so the model learns the RELATION, not a domain. Never tune
    the corpus to one stack.
  - PRECISION bias. `unrelated` is over-weighted and, crucially, includes SAME-AREA
    hard negatives (two different decisions both about "the database", "auth", "the
    cache", ...). These are the pairs a bare cosine threshold false-supersedes; the
    cross-encoder must learn to hold them apart.
  - NO eval leakage. Any generated pair matching the held-out gold fixture
    (tests/fixtures/supersession_pairs_gold.json) is dropped.

Output: tests/fixtures/supersession_pairs_generated.jsonl  ({a, b, relation} / line).
This is TRAINING data — kept separate from the held-out gold pair fixture.

Usage:
    python scripts/gen_supersession_pairs.py [--per 400] [--model gpt-4o-mini]
        [--chunk 12] [--temperature 0.9] [--only supersedes,unrelated]
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
_OUT = _ROOT / "tests" / "fixtures" / "supersession_pairs_generated.jsonl"
_GOLD = _ROOT / "tests" / "fixtures" / "supersession_pairs_gold.json"

# Broad, GLOBAL pool so the relation — not the domain — is what's learned. Mixes
# languages, project types, and business domains. Sampled per chunk.
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

# Aspect pool — sampled per chunk to force TOPICAL spread across the whole surface
# of software work (kept in sync with gen_twin_corpus.py).
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

# Per-relation generation spec. The model is asked to write a realistic minimal
# pair in one sampled domain. `a` is always the NEWER assertion.
RELATION_SPECS: dict[str, dict] = {
    "supersedes": {
        "target": 350,
        "instruction": (
            "Write a pair about the SAME topic where a later decision CHANGES the "
            "earlier choice (a correction). `b` states the original choice; `a` is the "
            "newer decision that adopts a DIFFERENT option for the same purpose. Vary "
            "phrasing so token overlap is sometimes low (paraphrase, synonyms)."
        ),
    },
    "refines": {
        "target": 250,
        "instruction": (
            "Write a pair stating the SAME rule/setting where `a` updates `b` to a "
            "more specific or different VALUE (a number, threshold, version, or scope). "
            "Keep the subject identical; only the value changes."
        ),
    },
    "subsumes": {
        "target": 200,
        "instruction": (
            "Write a pair where `b` is a SPECIFIC rule and `a` is a more GENERAL rule "
            "that fully absorbs it (e.g. b='do not use Redis', a='do not depend on any "
            "external managed services'). `a` must logically cover `b`."
        ),
    },
    "contradicts": {
        "target": 200,
        "instruction": (
            "Write a pair about the SAME subject with OPPOSITE polarity — `b` forbids/"
            "requires something and `a` reverses it (allow vs forbid, must vs must-not)."
        ),
    },
    "unrelated": {
        "target": 600,
        "instruction": (
            "Write a pair of TWO GENUINELY DIFFERENT decisions or constraints that a "
            "naive similarity check might confuse. Strongly prefer SAME-AREA hard "
            "negatives: both about the database (but different choices), both about "
            "auth, both about caching, both carrying a number/seconds/version, both "
            "naming a primary key, etc. — yet about DIFFERENT concerns so neither "
            "should replace the other. Include some easy cross-concern pairs too."
        ),
    },
}

_SYS = (
    "You generate labeled PAIRS of project-memory sentences to train a model that "
    "decides whether a newer decision supersedes an older one. Each sentence is one "
    "short declarative statement a developer might record (a decision, constraint, or "
    "rejected approach), with real library names, numbers, and env vars. Output ONLY "
    "valid JSON. Make pairs realistic, varied in phrasing and length, and specific. "
    "Do not repeat near-identical pairs."
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


def _pair_key(a: str, b: str) -> str:
    return f"{_norm(a)}||{_norm(b)}"


def _excluded_pairs() -> set[str]:
    """Both orderings of every gold pair, so leakage is caught regardless of a/b order."""
    out: set[str] = set()
    try:
        gold = json.loads(_GOLD.read_text(encoding="utf-8"))
        for p in gold["pairs"]:
            out.add(_pair_key(p["a"], p["b"]))
            out.add(_pair_key(p["b"], p["a"]))
    except Exception:
        pass
    return out


def gen_for_relation(client, model, relation, spec, chunk, temperature, excluded) -> list[tuple[str, str]]:
    target = spec["target"]
    got: list[tuple[str, str]] = []
    seen: set[str] = set()
    safety = 0
    while len(got) < target and safety < 400:
        safety += 1
        domain = random.choice(DOMAINS)  # random sampling spreads pairs across the global pool
        facet = random.choice(FACETS)
        n = min(chunk, target - len(got) + 4)
        user = (
            f"Domain context: {domain}.\n"
            f"Focus this batch on the aspect: {facet}.\n"
            f"Generate {n} distinct PAIRS for relation '{relation}'.\n"
            f"{spec['instruction']}\n"
            "`a` is the NEWER statement, `b` is the OLDER statement.\n"
            "Make them VERY diverse: each pair a DIFFERENT specific topic within the "
            "aspect, varying sentence length and phrasing.\n"
            'Return JSON: {"pairs": [{"a": "...", "b": "..."}, ...]}'
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
            items = data.get("pairs") or []
        except Exception as exc:  # noqa: BLE001
            print(f"  [{relation}] call failed: {exc}", file=sys.stderr)
            continue
        for it in items:
            if not isinstance(it, dict):
                continue
            a = (it.get("a") or "").strip()
            b = (it.get("b") or "").strip()
            if not a or not b:
                continue
            key = _pair_key(a, b)
            if key in seen or key in excluded:
                continue
            seen.add(key)
            got.append((a, b))
        print(f"  [{relation}] {len(got)}/{target}", end="\r", file=sys.stderr)
    print(file=sys.stderr)
    return got[:target]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--out", default=str(_OUT))
    ap.add_argument("--per", type=int, default=None, help="override target pairs per relation")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--only", help="comma-separated subset of relations")
    ap.add_argument("--append", action="store_true", help="append to the output file instead of overwriting")
    args = ap.parse_args()

    key = _load_api_key()
    if not key:
        print("OPENAI_API_KEY not set (env or .env)", file=sys.stderr)
        return 2
    import httpx
    from openai import OpenAI

    client = OpenAI(api_key=key, http_client=httpx.Client(timeout=60.0))
    excluded = _excluded_pairs()
    only = set(args.only.split(",")) if args.only else None

    rows: list[tuple[str, str, str]] = []
    for relation, spec in RELATION_SPECS.items():
        if only and relation not in only:
            continue
        if args.per is not None:
            spec = {**spec, "target": args.per}
        print(f"generating {relation} (target {spec['target']})...", file=sys.stderr)
        for a, b in gen_for_relation(client, args.model, relation, spec, args.chunk, args.temperature, excluded):
            rows.append((a, b, relation))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    with out.open(mode, encoding="utf-8") as f:
        for a, b, relation in rows:
            f.write(json.dumps({"a": a, "b": b, "relation": relation}, ensure_ascii=False) + "\n")

    try:
        disp = out.relative_to(_ROOT)
    except ValueError:
        disp = out
    print(f"\n{'appended' if args.append else 'wrote'} {len(rows)} pairs to {disp}")
    from collections import Counter
    for rel, c in Counter(r for *_, r in rows).most_common():
        print(f"  {rel:<14} {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
