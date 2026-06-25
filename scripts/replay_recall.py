"""J1 acceptance — replay the gamma benchmark probes through hybrid_recall.

For each probe prompt: does the gold fact (regex over event descriptions)
appear in hybrid_recall top-5? Denominator counts only probes whose gold fact
EXISTS in the store (extraction misses are not retrieval misses).

Usage: python scripts/replay_recall.py [--db PATH] [--k 5] [--cold]
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

# (probe_id, prompt, gold_regex) — prompts verbatim from the gamma run sheet;
# gold regexes target the model's ACTUAL S1/Sn decisions (re-baselined).
PROBES: list[tuple[str, str, str]] = [
    ("S2-P3 D2", "Inside the per-deployment retry, how many attempts and what backoff?",
     r"2 attempts.*(?:base=?100|backoff)|base=100 ?ms"),
    ("S2-P4 D1", "Two healthy deployments of the same model - how do we choose between them?",
     r"least-latency|priority order|weighted random|insertion order"),
    ("S2-P5 D3", "If the entire frontier group is down, what happens to the request?",
     r"follow_chain|frontier.*cheap.*local|falls? through.*cheap"),
    ("S2-P6 D11", "Add the per-request timeout to the router entrypoint. What value?",
     r"request_total_timeout|300 ?s|wall-clock"),
    ("S2-P7 D15", "Should we wrap provider calls in LangChain's chat-model abstraction to simplify multi-provider support?",
     r"LangChain"),
    ("S2-P8 D4", "Write the resolver mapping the default alias to a concrete model. Which model does it return?",
     r"relay-default.*(?:opus|resolves)|claude-opus"),
    ("S3-P2 D5", "Implement the rpm/tpm limiter. Where do the counters live and why?",
     r"counters? (?:live )?in Redis|Redis is the right answer|in-process counters fail"),
    ("S3-P3 D16", "For speed, can we just keep the rate-limit counters in a process-local dict?",
     r"in-process counters fail|multi-instance|module-level dict"),
    ("S3-P4 D8", "Implement the completion cache. What's the key and the TTL?",
     r"TTL|3600|cache:completion|temperature == 0"),
    ("S3-P5 D12", "Add logging around the limiter and cache. What must be redacted?",
     r"redact|never log|never appear in logs|PII"),
    ("S3-P6 D2c", "Write the backoff helper the router retries use. How many attempts, what jitter, and why?",
     r"3 attempts|full jitter|thundering"),
    ("S3-P7 D10", "Add a tracing span around a provider call - span name and key attributes?",
     r"span|gen_ai|tracing|upstream\.attempt"),
    ("S4-P2 D7", "Implement the spend-ledger write after a completion. In what units is cost stored?",
     r"nano-dollars|cost_nanos|nanos_per_token"),
    ("S4-P3 D14", "Implement virtual-key verification middleware. Key format and storage?",
     r"rly_|key_hash|base62|SHA-256\(raw"),
    ("S4-P4 D6", "Where do the upstream Anthropic/OpenAI keys come from at call time?",
     r"[Ee]nvironment variables only|ProviderKeyStore|env.*secrets|SecretStr|never touch a database"),
    ("S4-P5 D8c", "Write the cache lookup. How do we decide a hit now?",
     r"cosine.*0\.97|semantic cache|nearest stored embedding"),
    ("S4-P6 D9", "Wire streaming for both cached and live responses. What transport?",
     r"SSE|text/event-stream"),
    ("S4-P7 D13", "Add a config field for the semantic-cache threshold. How and where is it validated?",
     r"pydantic|TOML|fail-fast|model_validate"),
    ("S5-P1 T1", "Check in - what was the active thread we opened in Session 1?",
     r"fallback.{0,30}retry router|retry policy.*number of attempts|end-to-end"),
    ("S5-P2 D4c", "A new engineer asks: what model does the default alias resolve to today?",
     r"claude-opus-4-8"),
    ("S5-P3 D2c", "And how many retry attempts, with what jitter, and why that way?",
     r"3 attempts|full jitter|thundering"),
    ("S5-P4 D8c", "And how does the completion cache decide a hit now?",
     r"cosine.*0\.97|semantic"),
    ("S5-P5 D7", "Write a query summing a key's spend this month. What units come back?",
     r"cost_nanos|nano-dollars"),
    ("S5-P6 D5+16", "Perf idea: move the rate-limit counters in-process to cut Redis hops - good?",
     r"in-process counters fail|Redis|multi-instance"),
    ("S5-P7 D11+12", "Write the top-level request handler; call out the timeout and the log-redaction rules.",
     r"300 ?s|redact|never log"),
    ("S5-P8 multi", "Write an integration test that forces a primary-group failure and asserts the fallback path. List every constraint the test must respect",
     r"circuit breaker|3 consecutive|trip=3|follow_chain"),
    ("S5-P9 D10+6", "Add OTel spans to the provider client and confirm no secret leaks into span attributes",
     r"span|gen_ai|attributes|secret"),
    ("S5-P10 D13", "Validate the full relay.config.yaml at boot - what happens on a bad value?",
     r"fail-fast|ValidationError|model_validate|exits non-zero|fully starts or exits"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=r".bench_dbs\gamma_cogni.db")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--cold", action="store_true", help="skip dense axis (BM25 only)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    from memlora.storage.migrations import run_migrations
    run_migrations(conn)  # creates + backfills FTS on the copy

    pid = conn.execute(
        "SELECT DISTINCT project_id FROM events LIMIT 1").fetchone()[0]

    if not args.cold:
        from memlora.embedding.model import ensure_ready
        ready = ensure_ready(timeout=90.0)
        print(f"dense axis: {'warm' if ready else 'COLD (bm25 only)'}")
    else:
        import memlora.embedding.model as m
        m.is_ready = lambda: False  # type: ignore[assignment]
        print("dense axis: forced cold")

    from memlora.retrieval.hybrid import hybrid_recall

    actives = [
        (r["id"], r["d"] or "") for r in conn.execute(
            "SELECT id, json_extract(payload,'$.description') d FROM events "
            "WHERE project_id=? AND archived=0 AND superseded_by IS NULL", (pid,))
    ]

    n_exists = n_hit = 0
    misses: list[str] = []
    for probe_id, prompt, gold in PROBES:
        rx = re.compile(gold, re.IGNORECASE | re.DOTALL)
        exists = any(rx.search(d) for _, d in actives)
        if not exists:
            print(f"  [absent ] {probe_id}  (gold fact not in store)")
            continue
        n_exists += 1
        hits = hybrid_recall(conn, pid, prompt, k=args.k)
        hit = any(rx.search(h["description"]) for h in hits)
        n_hit += hit
        axis = ""
        if hit:
            h = next(h for h in hits if rx.search(h["description"]))
            axis = f"d={h['dense_rank']} b={h['bm25_rank']}"
        print(f"  [{'HIT    ' if hit else 'MISS   '}] {probe_id}  {axis}")
        if not hit:
            misses.append(probe_id)
            for h in hits[:3]:
                print(f"        top: ({h['event_type']}) {h['description'][:90]}")

    pct = 100.0 * n_hit / n_exists if n_exists else 0.0
    print(f"\ntop-{args.k} recall: {n_hit}/{n_exists} = {pct:.1f}%  (target >=90%)")
    if misses:
        print("misses:", ", ".join(misses))


if __name__ == "__main__":
    main()
