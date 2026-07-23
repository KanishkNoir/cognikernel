"""J7 — block budget elasticity: distinct gold facts vs budget + real cost model.

Renders the gamma block at several budgets (sandboxed copy) and counts how many
DISTINCT ground-truth facts the block carries. Cost model from the run's real
telemetry: delta cache-read tokens = (B-1500) x calls/session at 0.1x input
price, vs the measured cost of agent pull round-trips the bigger block avoids.

Usage: python scripts/measure_budget_elasticity.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "src"))
sys.stdout.reconfigure(encoding="utf-8")

PID = "79e384f12a00c402"
SRC_DB = _REPO / ".bench_dbs" / "gamma_cogni.db"
PROJECT = r"C:\Users\Admin\OneDrive\Desktop\GAMMA_COGNI"
TDIR = Path(r"C:\Users\Admin\.claude\projects\C--Users-Admin-OneDrive-Desktop-GAMMA-COGNI")

# Distinct ground-truth facts (chains deduped to their CURRENT value).
GOLD_FACTS = {
    "D1 routing": r"priority|least-latency|weighted random|failover budget",
    "D2 retry (current)": r"3 attempts|full jitter",
    "D3 fallback chain": r"frontier.*cheap.*local|follow_chain",
    "D4 default alias (current)": r"claude-opus-4-8|relay-default.*[Oo]pus",
    "D5 rate limits redis": r"counters.*Redis|Redis.*counter|in-process counters fail",
    "D6 keys env-only": r"environment variables only|never touch a database|ProviderKeyStore",
    "D7 nano-dollars": r"nano-dollar|cost_nanos",
    "D8 cache (current)": r"cosine.*0\.97|semantic cach",
    "D9 SSE": r"SSE|text/event-stream",
    "D10 span": r"relay\.provider_call|gen_ai\.",
    "D11 timeout 300": r"300 ?s|request_total_timeout|RELAY_REQUEST_TIMEOUT",
    "D12 redaction": r"redact|never log|never appear in logs|never contains a key",
    "D13 config fail-fast": r"fail-?fast|ValidationError|model_validate|pydantic",
    "D14 virtual keys": r"rly_|key_hash|base62",
    "D15 no langchain": r"LangChain",
    "D16 no in-process limiter": r"in-process counters fail|multi-instance",
    "T1 thread": r"fallback.{0,30}retry router|retry policy",
}

BUDGETS = [1500, 2500, 3500, 5000]


def main() -> None:
    home = Path(tempfile.mkdtemp(prefix="cognikernel_j7_"))
    (home / "projects").mkdir()
    shutil.copy(SRC_DB, home / "projects" / f"{PID}.db")
    os.environ["COGNIKERNEL_DIR"] = str(home)
    os.environ["COGNIKERNEL_DISABLE_AUTO_WARM"] = "1"

    import dataclasses

    from cognikernel.config import Config
    from cognikernel.injection.template import count_tokens_accurate
    from cognikernel.integration.session import render_state
    from cognikernel.storage.projections import invalidate_projection

    base = Config.load(project_path=PROJECT)

    print("budget | tokens | facts in block | missing")
    print("-" * 70)
    results = {}
    for b in BUDGETS:
        conn = sqlite3.connect(home / "projects" / f"{PID}.db")
        conn.row_factory = sqlite3.Row
        invalidate_projection(conn, PID)
        conn.close()
        cfg = dataclasses.replace(base, token_budget=b)
        block = render_state(PROJECT, cfg)
        toks = count_tokens_accurate(block)
        present = [k for k, rx in GOLD_FACTS.items()
                   if re.search(rx, block, re.IGNORECASE | re.DOTALL)]
        missing = [k for k in GOLD_FACTS if k not in present]
        results[b] = (toks, len(present), missing)
        print(f"{b:>6} | {toks:>6} | {len(present)}/{len(GOLD_FACTS)}"
              f"          | {', '.join(m.split()[0] for m in missing)}")

    # Cost model from the run's real telemetry.
    print("\n--- cost model (gamma run telemetry) ---")
    sessions = []
    for p in TDIR.glob("*.jsonl"):
        if p.stat().st_size < 100_000:
            continue
        calls = cache_read = 0
        for line in open(p, encoding="utf-8"):
            try:
                u = (json.loads(line).get("message") or {}).get("usage") or {}
            except Exception:
                continue
            if u:
                calls += 1
                cache_read += u.get("cache_read_input_tokens", 0)
        sessions.append((p.stem[:8], calls, cache_read))
    total_calls = sum(c for _, c, _ in sessions)
    total_cr = sum(cr for _, _, cr in sessions)
    print(f"sessions: {len(sessions)}  api calls: {total_calls}  "
          f"cache_read total: {total_cr/1e6:.1f}M")
    for b in BUDGETS[1:]:
        extra = (b - 1500) * total_calls
        pct = 100 * extra / max(total_cr, 1)
        print(f"  budget {b}: +{extra/1e6:.2f}M cache-read tokens across the run "
              f"(+{pct:.1f}% of measured cache reads; 0.1x input price)")
    print("\npull cost reference: a recall MCP round-trip ≈ 1-2k tokens at FULL "
          "price + a turn of latency; the gamma run made ~60 pulls (S4 alone ~30).")


if __name__ == "__main__":
    main()
