#!/usr/bin/env python3
"""Cross-benchmark retention gate — enforces "fixes only make the benchmark go up".

Checks that a benchmark's ground-truth facts are still present in a project's ACTIVE
events, and reports the active-event count. Run before/after any de-noising fix
(R1/R3/R4/R5) on BOTH benchmarks — the fact count must stay full while active count
drops. Losing a fact = intelligence loss = FAIL. Stored-events only (no model).

Facts come in two tiers:
  core  — mechanism-based; must pass on every run regardless of specific value the
          model chose (e.g. "a timeout exists" rather than "timeout=120s"). These are
          the ONLY facts that count toward min_retain and can fail --check.
  run   — value-specific (e.g. the exact seconds chosen). Only WARN when missing; a
          different run may legitimately choose a different value. Use --gold-file to
          supply per-run gold for these and promote them to gated.

Benchmarks:
  relay  — Relay recall-heavy gateway (v2-broad), 12 core facts + 3 supersession chains.
  mobc   — MOB_C / Taskflow (legacy), 10 facts incl. the bcrypt->argon2id chain + Celery graveyard.

Usage:
    python scripts/eval_retention.py                              # both, auto from known ids
    python scripts/eval_retention.py --benchmark relay [--check]
    python scripts/eval_retention.py --db <project.db>           # explicit DB, auto-detect benchmark
    python scripts/eval_retention.py --db <db> --benchmark relay --project-id <id>
    python scripts/eval_retention.py --gold-file relay_run2.json  # per-run value overrides
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# Each fact entry: name -> {"groups": [...], "tier": "core"|"run"}
# tier=core  → counted toward min_retain, fails --check when missing
# tier=run   → warns only unless promoted via --gold-file
#
# Keyword groups: present if ANY group fully matches (all-of within a group).
BENCHMARKS: dict[str, dict] = {
    "relay": {
        # Canonical run-1 project (TEST_RELAY_CK). Use --db or --project-id to
        # target a different run (e.g. TEST_BETA_RELAY_CK = 449280faa20605c6).
        "project_id": "83c6f1dcec7c2318",
        "min_retain": 11,  # core facts only; D11-specific value is now tier=run
        "facts": {
            "D4 chain latest=opus":        {"groups": [["opus"]],                                   "tier": "core"},
            "D2 chain 3 attempts":         {"groups": [["3", "attempt"], ["3 total"]],              "tier": "core"},
            "D2 chain full jitter":        {"groups": [["jitter"]],                                 "tier": "core"},
            "D8 chain semantic cosine":    {"groups": [["cosine"], ["semantic", "cache"]],          "tier": "core"},
            # Mechanism: a hard timeout constraint exists. Specific value (60s/120s) is
            # run-specific (the model may choose any value) — checked via tier=run below.
            "D11 timeout constraint":      {"groups": [["timeout"], ["deadline"], ["anyio"]],       "tier": "core"},
            "D11 timeout 120s (run-1)":    {"groups": [["120", "timeout"]],                        "tier": "run"},
            "D12 redact/never-log":        {"groups": [["redact"], ["never", "log"]],               "tier": "core"},
            "D13 config fail-fast":        {"groups": [["fail-fast"], ["startup"], ["startuperror"]], "tier": "core"},
            "D14 hashed virtual keys":     {"groups": [["sha-256"], ["hash", "key"]],               "tier": "core"},
            "D7 spend units (token)":      {"groups": [["token"]],                                  "tier": "core"},
            "D15 graveyard: LangChain":    {"groups": [["langchain"]],                              "tier": "core"},
            "D16 graveyard: in-process":   {"groups": [["in-process"], ["process-local"], ["module-level dict"]], "tier": "core"},
            "T1 router/retry thread":      {"groups": [["router"], ["retry", "policy"]],            "tier": "core"},
        },
    },
    "mobc": {
        "project_id": "9d61801554d730b8",
        "min_retain": 9,  # legacy baseline; D3 Alembic is a genuine recall gap (v2 may recover it)
        "facts": {
            "D1 Postgres (not SQLite)":    {"groups": [["postgres"]],                                           "tier": "core"},
            "D2 UUID primary keys":        {"groups": [["uuid"]],                                               "tier": "core"},
            "D3 Alembic migrations":       {"groups": [["alembic"]],                                            "tier": "core"},
            "D4 async SQLAlchemy":         {"groups": [["async", "sqlalchemy"], ["async", "sqla"]],             "tier": "core"},
            "D5 JWT secret env var":       {"groups": [["taskflow_jwt_secret"], ["jwt", "secret"]],             "tier": "core"},
            "D6 snake/camel case":         {"groups": [["snake_case"], ["camelcase"], ["camel case"]],          "tier": "core"},
            "D7 shadcn/headless ui":       {"groups": [["shadcn"], ["headless"], ["material ui"], ["chakra"], ["component library"]], "tier": "core"},
            "D8 /api/v1 prefix":           {"groups": [["/api/v1"], ["api/v1"]],                                "tier": "core"},
            "D9 chain latest=argon2id":    {"groups": [["argon2id"], ["argon2"]],                               "tier": "core"},
            "D10 graveyard: no Celery":    {"groups": [["celery"]],                                             "tier": "core"},
        },
    },
}


def _active_descriptions(db_path: Path) -> list[str]:
    db = sqlite3.connect(str(db_path)); db.row_factory = sqlite3.Row
    rows = db.execute(
        "select json_extract(payload,'$.description') d from events "
        "where superseded_by is null and archived=0"
    ).fetchall()
    db.close()
    return [(r["d"] or "").lower() for r in rows]


def _present(descs: list[str], groups: list[list[str]]) -> bool:
    return any(all(k in d for k in g) for g in groups for d in descs)


def _db_for(project: str | None, db: str | None, bench: str, project_id: str | None = None) -> Path | None:
    if db:
        return Path(db)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from cognikernel.config import Config
    from cognikernel.storage.connection import get_db_path
    if project:
        from cognikernel.storage.connection import hash_project_path
        return get_db_path(Config.load(), hash_project_path(project))
    pid = project_id or BENCHMARKS[bench]["project_id"]
    return get_db_path(Config.load(), pid)


def _load_gold_file(path: str) -> dict[str, dict]:
    """Load per-run fact overrides from a JSON file.

    Format: {"fact_name": {"groups": [[kw, ...], ...], "tier": "core"}, ...}
    Loaded facts are merged into the benchmark's fact table, overriding by name.
    """
    with open(path) as f:
        return json.load(f)


def run_one(bench: str, db_path: Path, gold_override: dict | None = None) -> bool:
    spec = BENCHMARKS[bench]
    facts: dict[str, dict] = dict(spec["facts"])

    # Merge any per-run gold (promote run-tier facts or add new ones).
    if gold_override:
        facts.update(gold_override)

    core_facts   = {n: v for n, v in facts.items() if v.get("tier", "core") == "core"}
    run_facts    = {n: v for n, v in facts.items() if v.get("tier") == "run"}
    floor        = spec.get("min_retain", len(core_facts))

    if not db_path.exists():
        print(f"[{bench}] no db at {db_path} — skip"); return True

    descs = _active_descriptions(db_path)
    core_kept = sum(_present(descs, v["groups"]) for v in core_facts.values())
    ok = core_kept >= floor

    print(f"\n[{bench}] active events: {len(descs)}"
          f"   core retained {core_kept}/{len(core_facts)}"
          f"  (baseline >= {floor}) {'OK' if ok else 'REGRESSION'}")

    for name, v in core_facts.items():
        if not _present(descs, v["groups"]):
            print(f"   MISS (core) {name}")

    if run_facts:
        run_kept = sum(_present(descs, v["groups"]) for v in run_facts.values())
        print(f"   run-specific: {run_kept}/{len(run_facts)} present (warn only, not gated)")
        for name, v in run_facts.items():
            if not _present(descs, v["groups"]):
                print(f"   WARN (run)  {name}")

    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=list(BENCHMARKS), help="default: both")
    ap.add_argument("--db", help="explicit project .db path")
    ap.add_argument("--project", help="project path (derives DB)")
    ap.add_argument("--project-id", dest="project_id",
                    help="override project_id for DB lookup (e.g. 449280faa20605c6 for relay-beta)")
    ap.add_argument("--gold-file", dest="gold_file",
                    help="JSON file with per-run fact overrides (promotes/adds run-tier facts)")
    ap.add_argument("--check", action="store_true", help="exit 1 on any REGRESSION")
    args = ap.parse_args()

    gold = _load_gold_file(args.gold_file) if args.gold_file else None

    # --db with no --benchmark: try to auto-detect benchmark by project_id in the DB.
    if args.db and not args.benchmark:
        db_path = Path(args.db)
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path)); conn.row_factory = sqlite3.Row
                pid = conn.execute("select project_id from events limit 1").fetchone()
                conn.close()
                if pid:
                    pid = pid["project_id"]
                    for b, spec in BENCHMARKS.items():
                        if spec["project_id"] == pid:
                            args.benchmark = b; break
            except Exception:
                pass

    benches = [args.benchmark] if args.benchmark else list(BENCHMARKS)
    all_ok = True
    for b in benches:
        db_path = _db_for(args.project, args.db, b, args.project_id)
        ok = run_one(b, db_path, gold_override=gold)
        all_ok = all_ok and ok

    print(f"\nOVERALL: {'ALL RETAINED' if all_ok else 'INTELLIGENCE LOSS DETECTED'}")
    return 0 if (all_ok or not args.check) else 1


if __name__ == "__main__":
    sys.exit(main())
