#!/usr/bin/env python3
"""Cross-benchmark retention gate — enforces "fixes only make the benchmark go up".

Checks that a benchmark's ground-truth facts are still present in a project's ACTIVE
events, and reports the active-event count. Run before/after any de-noising fix
(R1/R3/R4/R5) on BOTH benchmarks — the fact count must stay full while active count
drops. Losing a fact = intelligence loss = FAIL. Stored-events only (no model).

Benchmarks:
  relay  — Relay recall-heavy gateway (v2-broad), 12 facts incl. 3 supersession chains.
  mobc   — MOB_C / Taskflow (legacy), 10 facts incl. the bcrypt->argon2id chain + Celery graveyard.

Usage:
    python scripts/eval_retention.py --benchmark relay --project <path> [--check]
    python scripts/eval_retention.py --db <project.db>            # auto-detects by project_id
    python scripts/eval_retention.py                              # both, auto from known ids
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

# Each fact: name -> list of (all-of) keyword groups; present if ANY group matches.
BENCHMARKS = {
    "relay": {
        "project_id": "83c6f1dcec7c2318",
        "min_retain": 12,  # v2-broad baseline (verified)
        "facts": {
            "D4 chain latest=opus":        [["opus"]],
            "D2 chain 3 attempts":         [["3", "attempt"], ["3 total"]],
            "D2 chain full jitter":        [["jitter"]],
            "D8 chain semantic cosine":    [["cosine"], ["semantic", "cache"]],
            "D11 timeout 120s":            [["120", "timeout"]],
            "D12 redact/never-log":        [["redact"], ["never", "log"]],
            "D13 config fail-fast":        [["fail-fast"], ["startup"], ["startuperror"]],
            "D14 hashed virtual keys":     [["sha-256"], ["hash", "key"]],
            "D7 spend units (token)":      [["token"]],
            "D15 graveyard: LangChain":    [["langchain"]],
            "D16 graveyard: in-process":   [["in-process"], ["process-local"], ["module-level dict"]],
            "T1 router/retry thread":      [["router"], ["retry", "policy"]],
        },
    },
    "mobc": {
        "project_id": "9d61801554d730b8",
        "min_retain": 9,  # legacy baseline; D3 Alembic is a genuine legacy recall gap (v2 may recover it)
        "facts": {
            "D1 Postgres (not SQLite)":    [["postgres"]],
            "D2 UUID primary keys":        [["uuid"]],
            "D3 Alembic migrations":       [["alembic"]],
            "D4 async SQLAlchemy":         [["async", "sqlalchemy"], ["async", "sqla"]],
            "D5 JWT secret env var":       [["taskflow_jwt_secret"], ["jwt", "secret"]],
            "D6 snake/camel case":         [["snake_case"], ["camelcase"], ["camel case"]],
            "D7 shadcn/headless ui":       [["shadcn"], ["headless"], ["material ui"], ["chakra"], ["component library"]],
            "D8 /api/v1 prefix":           [["/api/v1"], ["api/v1"]],
            "D9 chain latest=argon2id":    [["argon2id"], ["argon2"]],
            "D10 graveyard: no Celery":    [["celery"]],
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


def _db_for(project: str | None, db: str | None, bench: str) -> Path | None:
    if db:
        return Path(db)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from memlora.config import Config
    from memlora.storage.connection import get_db_path
    if project:
        from memlora.storage.connection import hash_project_path
        return get_db_path(Config.load(), hash_project_path(project))
    return get_db_path(Config.load(), BENCHMARKS[bench]["project_id"])


def run_one(bench: str, db_path: Path) -> bool:
    spec = BENCHMARKS[bench]
    facts = spec["facts"]
    floor = spec.get("min_retain", len(facts))
    if not db_path.exists():
        print(f"[{bench}] no db at {db_path} — skip"); return True
    descs = _active_descriptions(db_path)
    kept = sum(_present(descs, g) for g in facts.values())
    ok = kept >= floor
    print(f"\n[{bench}] active events: {len(descs)}   retained {kept}/{len(facts)}  "
          f"(baseline >= {floor}) {'OK' if ok else 'REGRESSION'}")
    for name, groups in facts.items():
        if not _present(descs, groups):
            print(f"   MISS {name}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=list(BENCHMARKS), help="default: both")
    ap.add_argument("--db", help="explicit project .db (implies single benchmark)")
    ap.add_argument("--project", help="project path")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    benches = [args.benchmark] if args.benchmark else list(BENCHMARKS)
    all_ok = True
    for b in benches:
        ok = run_one(b, _db_for(args.project, args.db, b))
        all_ok = all_ok and ok
    print(f"\nOVERALL: {'ALL RETAINED' if all_ok else 'INTELLIGENCE LOSS DETECTED'}")
    return 0 if (all_ok or not args.check) else 1


if __name__ == "__main__":
    sys.exit(main())
