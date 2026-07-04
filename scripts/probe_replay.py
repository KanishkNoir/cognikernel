"""Offline replay of the 5 genuine CK mistakes from the four-project benchmark.

The consolidated findings (research/benchmarking/consolidated_findings.md §6)
attribute every CK mistake to two mechanisms:
  Mode A — "surfaces but doesn't bind": Relay D5/D16, Toolbelt A9, Conductor
           max_attempts. Fixes shipped since: K2 PreToolUse prohibition
           surfacing (+#56 authority/scope ranking), CK-1 dual-evidence push.
  Mode B — "decide-in-prose, defer-the-file": Taskflow D2 (UUID PK in S1 prose
           DDL) + T1 (thread auto-close). Fix shipped since: #41
           schema-decision capture from DDL code blocks.

This harness replays each probe OFFLINE against the ORIGINAL benchmark stores
(copied into a scratch MEMLORA_DIR; originals untouched):

  Part 1 (Mode B): rebuild_from_raw re-extracts every arm's raw_evidence with
  the CURRENT pipeline (legacy AND v2-broad) and scores gold-fact capture.

  Part 2 (Mode A): the historical offending edit (reconstructed from the
  session QA transcripts) is fed through surface_prohibitions_for_edit (K2,
  the Write-time bind) and the probe's natural-language prompt through
  recall_for_prompt (CK-1, the prompt-time bind) against each store.

A probe PASSES when the current mechanism would have surfaced/captured what
the original run missed. Results print as a table + JSON for the results doc
(research/benchmarking/probe_replay_2026-07.md).

Usage:  uv run python scripts/probe_replay.py [--scratch DIR] [--skip-v2]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REAL_MEMLORA = Path.home() / ".memlora"

# The four CK-arm stores the consolidated findings scored
# (scripts/_token_telemetry.py is the canonical arm->dir mapping).
ARMS: dict[str, tuple[str, str]] = {
    "relay":     (r"C:\Users\Admin\OneDrive\Desktop\OMEGA_RELAY",     "484b812967d795c6"),
    "taskflow":  (r"C:\Users\Admin\OneDrive\Desktop\Taskflow_ALPHA",  "961b42e80e47feef"),
    "toolbelt":  (r"C:\Users\Admin\OneDrive\Desktop\TOOLBELT_ALPHA",  "d5dce4e1032e6457"),
    "conductor": (r"C:\Users\Admin\OneDrive\Desktop\CONDUCTOR_BETA",  "720e6b7c1e7d2266"),
}

# ── Part 1 gold sets (extraction recall) ─────────────────────────────────────
# Taskflow: the 11 run-sheet decisions (from scripts/_taskflow_extraction_recall.py).
# Original-run misses: D2 (UUID PK, decided in S1 prose DDL) and T1 (thread).
TASKFLOW_GOLD = [
    ("D1", "PostgreSQL not SQLite", lambda d: "postgres" in d),
    ("D2", "UUID primary keys",     lambda d: "uuid" in d),
    ("D3", "Alembic only",          lambda d: "alembic" in d),
    ("D4", "Async SQLAlchemy only", lambda d: "async" in d and ("sqlalchemy" in d or "session" in d or "orm" in d)),
    ("D5", "JWT secret env var",    lambda d: "taskflow_jwt_secret" in d or ("jwt" in d and "secret" in d)),
    ("D6", "snake/camel casing",    lambda d: "snake_case" in d or "camelcase" in d or "camel case" in d or "casing" in d),
    ("D7", "shadcn/ui",             lambda d: "shadcn" in d),
    ("D8", "/api/v1/ prefix",       lambda d: "api/v1" in d or ("v1" in d and "prefix" in d)),
    ("D9", "argon2id (was bcrypt)", lambda d: "argon2" in d or "bcrypt" in d),
    ("D10","No Celery (graveyard)", lambda d: "celery" in d),
    ("T1", "JWT auth thread open",  lambda d: ("jwt" in d or "auth" in d) and ("thread" in d or "implement" in d or "login" in d)),
]
# Conductor: the invariant the original run drifted on (worker shipped
# MAX_ATTEMPTS=10 against the S1 schema default of 3).
CONDUCTOR_GOLD = [
    ("Q3", "max attempts value", lambda d: ("max" in d and "attempt" in d) or "max_attempts" in d),
]
GOLD = {"taskflow": TASKFLOW_GOLD, "conductor": CONDUCTOR_GOLD}

# ── Part 2 probes (Mode A bind) ──────────────────────────────────────────────
# offending_edit reconstructs the code the arm actually wrote when it made the
# mistake (from analysis_omega/s3_qa.txt, analysis_tb QA, conductor_three_arm.md).
BIND_PROBES = [
    {
        "id": "relay-D5/D16",
        "arm": "relay",
        "file_path": "relay/router.py",
        "offending_edit": (
            "# Rate limiting: process-level counters, keyed by credentials_ref\n"
            "_limiters: dict[str, RateLimiter] = {}\n"
            "\n"
            "def check_rate_limit(credentials_ref: str, rpm_limit: int, tpm_limit: int) -> bool:\n"
            "    # keep the rate-limit counters in a process-local dict in router.py —\n"
            "    # same module and lifetime as _breakers, asyncio-safe, no Redis hop\n"
            "    limiter = _limiters.setdefault(credentials_ref, RateLimiter(rpm_limit, tpm_limit))\n"
            "    return limiter.allow()\n"
        ),
        "prompt": "Perf idea: move the rate-limit counters in-process to cut Redis hops - good?",
        "expect_any": ["redis", "in-process", "instance"],
    },
    {
        "id": "toolbelt-A9",
        "arm": "toolbelt",
        "file_path": "toolbelt/gateway.py",
        "offending_edit": (
            "# local retry helper for the gateway path\n"
            "def run_with_retries(fn, max_attempts: int = 3, base_delay: float = 0.5):\n"
            "    # re-implement retry with backoff here so the gateway does not depend on toolbelt.retry\n"
            "    for attempt in range(max_attempts):\n"
            "        try:\n"
            "            return fn()\n"
            "        except RetryableError:\n"
            "            time.sleep(base_delay * (2 ** attempt))\n"
            "    raise\n"
        ),
        "prompt": "Should I write a small retry helper with backoff inside the gateway module?",
        # Anchors cover BOTH apt prohibitions in the store: the import-never-
        # reimplement rule AND the retrying-inside-the-gateway rule (the K2-fix
        # iteration surfaced the latter — a genuine bind the old anchors missed).
        "expect_any": ["import", "re-implement", "reimplement", "retry.py", "with_retry",
                       "retrying inside the gateway", "burns multiple attempts"],
    },
    {
        # Sensitivity variant: the duplicate shaped as a policy object (the store's
        # prohibitions are phrased around RetryPolicy, so 'policy' is the pivotal
        # overlap token — reconstruction wording materially changes the bind).
        "id": "toolbelt-A9b (policy-shaped)",
        "arm": "toolbelt",
        "file_path": "toolbelt/gateway.py",
        "offending_edit": (
            "# gateway-local retry policy\n"
            "class GatewayRetryPolicy:\n"
            "    # inline retry policy for the gateway instead of importing toolbelt.retry.RetryPolicy\n"
            "    max_attempts = 3\n"
            "    base_delay = 0.5\n"
            "\n"
            "def run_with_retry_policy(fn, policy: GatewayRetryPolicy):\n"
            "    for attempt in range(policy.max_attempts):\n"
            "        try:\n"
            "            return fn()\n"
            "        except RetryableError:\n"
            "            time.sleep(policy.base_delay * (2 ** attempt))\n"
            "    raise\n"
        ),
        "prompt": "I'll add a gateway-local retry policy class instead of importing the shared one, ok?",
        "expect_any": ["import", "re-implement", "reimplement", "retry.py", "with_retry", "module level",
                       "retrying inside the gateway", "burns multiple attempts"],
    },
    {
        "id": "conductor-max_attempts",
        "arm": "conductor",
        "file_path": "conductor/worker.py",
        "offending_edit": (
            "# worker retry ceiling\n"
            "MAX_ATTEMPTS = 10\n"
            "\n"
            "def handle_failure(job, attempt: int) -> None:\n"
            "    if attempt >= MAX_ATTEMPTS:\n"
            "        move_to_dead_letter(job)\n"
        ),
        "prompt": "Setting the worker retry ceiling MAX_ATTEMPTS to 10 - consistent with what we decided?",
        "expect_any": ["max_attempts", "attempt"],
    },
]


def setup_scratch(scratch: Path) -> None:
    (scratch / "projects").mkdir(parents=True, exist_ok=True)
    for _, (proj, pid) in ARMS.items():
        src = REAL_MEMLORA / "projects" / f"{pid}.db"
        if not src.exists():
            sys.exit(f"missing benchmark DB: {src}")
        shutil.copy2(src, scratch / "projects" / f"{pid}.db")


def make_config(scratch: Path, extractor: str):
    from memlora.config import Config
    return dataclasses.replace(Config(memlora_dir=scratch), extractor=extractor)


def event_texts(db: Path) -> list[tuple[str, str, bool]]:
    """(event_type, subject+description lowered, live) for every event."""
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    out = []
    for r in con.execute("SELECT event_type, payload, archived, superseded_by FROM events"):
        p = json.loads(r["payload"])
        txt = f"{p.get('subject','')} {p.get('description','')}".lower()
        live = r["archived"] == 0 and r["superseded_by"] is None
        out.append((r["event_type"], txt, live))
    con.close()
    return out


def score_gold(db: Path, gold) -> dict:
    events = event_texts(db)
    res = {}
    for gid, label, match in gold:
        hits = [(et, txt, live) for et, txt, live in events if match(txt)]
        res[gid] = {
            "label": label,
            "captured": bool(hits),
            "live": any(live for _, _, live in hits),
            # hit texts so a loose matcher's false-capture is verifiable by eye
            "hit_texts": [f"[{'LIVE' if live else 'dead'}] {et}: {txt[:140]}" for et, txt, live in hits[:4]],
        }
    return res


def part1_extraction(scratch: Path, extractors: list[str]) -> dict:
    """rebuild_from_raw sidecars with the current pipeline; score gold capture."""
    from memlora.integration.session import rebuild_from_raw

    results: dict = {}
    for arm in ("taskflow", "conductor"):
        proj, pid = ARMS[arm]
        results[arm] = {"baseline_store": score_gold(scratch / "projects" / f"{pid}.db", GOLD[arm])}
        for extractor in extractors:
            cfg = make_config(scratch, extractor)
            stats = rebuild_from_raw(proj, config=cfg)
            sidecar = scratch / "projects" / f"{pid}.db.rebuild"
            results[arm][extractor] = {
                "rebuild_stats": {k: stats[k] for k in ("evidence_count", "total_extracted", "errors")},
                "gold": score_gold(sidecar, GOLD[arm]),
            }
            # fresh sidecar per extractor
            for ext in ("", "-wal", "-shm"):
                p = Path(str(sidecar) + ext)
                if p.exists():
                    p.unlink()
    return results


def _overlap_debug(db: Path, probe_text: str, top: int = 3) -> list[str]:
    """Top prohibition-typed events by content-term overlap with the probe —
    shows why a near-miss missed (e.g. one token under the K2 floor)."""
    from memlora.delta.supersede import normalize_for_overlap
    q = normalize_for_overlap(probe_text)
    scored = []
    for et, txt, live in event_texts(db):
        if et not in ("APPROACH_ABANDONED_DO_NOT_RETRY", "CONSTRAINT_HARD"):
            continue
        ov = q & normalize_for_overlap(txt)
        if ov:
            scored.append((len(ov), live, et, sorted(ov), txt))
    scored.sort(key=lambda t: -t[0])
    return [f"ov={n} [{'LIVE' if live else 'dead'}] {et[:12]} shared={sh} :: {txt[:100]}"
            for n, live, et, sh, txt in scored[:top]]


def part2_bind(scratch: Path) -> dict:
    """Replay the offending edits through K2 and the prompts through CK-1."""
    from memlora.integration.query import recall_for_prompt, surface_prohibitions_for_edit

    cfg = make_config(scratch, "legacy")  # extractor irrelevant for bind
    results: dict = {}
    for probe in BIND_PROBES:
        proj, pid = ARMS[probe["arm"]]
        db = scratch / "projects" / f"{pid}.db"
        k2 = surface_prohibitions_for_edit(
            proj, probe["offending_edit"], file_path=probe["file_path"],
            config=cfg, session_id=f"probe-replay-k2-{probe['id']}",
        )
        ck1 = recall_for_prompt(
            proj, probe["prompt"], config=cfg, session_id=f"probe-replay-ck1-{probe['id']}",
        )
        def _hit(text: str) -> bool:
            t = text.lower()
            return bool(text) and any(a in t for a in probe["expect_any"])
        results[probe["id"]] = {
            "k2_surfaced": bool(k2),
            "k2_relevant": _hit(k2),
            "k2_text": k2,
            "ck1_surfaced": bool(ck1),
            "ck1_relevant": _hit(ck1),
            "ck1_text": ck1,
            "overlap_debug_edit": _overlap_debug(db, probe["offending_edit"]),
            "overlap_debug_prompt": _overlap_debug(db, probe["prompt"]),
        }
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--skip-v2", action="store_true",
                    help="skip the v2-broad extraction pass (ONNX)")
    args = ap.parse_args()

    scratch = Path(args.scratch) if args.scratch else Path(tempfile.mkdtemp(prefix="ck_probe_replay_"))
    setup_scratch(scratch)

    # Redirect the data dir to the scratch copies; keep the learned heads and
    # the embedding cache reachable by symlink/copy of the models dir — the
    # deployed bind path runs with the dense axis live, so the replay must too.
    os.environ["MEMLORA_DIR"] = str(scratch)
    os.environ["MEMLORA_V2_BODY_DIR"] = str(REAL_MEMLORA / "models" / "salience_v2")
    models_src = REAL_MEMLORA / "models"
    models_dst = scratch / "models"
    if models_src.exists() and not models_dst.exists():
        try:
            models_dst.symlink_to(models_src, target_is_directory=True)
        except OSError:
            shutil.copytree(models_src, models_dst)

    from memlora.embedding.model import ensure_ready
    dense_ok = ensure_ready(timeout=120)  # cached locally — a load, not a download

    from memlora.extraction import salience_v2
    v2_ok = salience_v2.is_available()
    extractors = ["legacy"] + ([] if (args.skip_v2 or not v2_ok) else ["v2-broad"])

    report = {
        "scratch": str(scratch),
        "dense_axis_live": dense_ok,
        "v2_available": v2_ok,
        "extractors_run": extractors,
        "part1_extraction": part1_extraction(scratch, extractors),
        "part2_bind": part2_bind(scratch),
    }

    out = Path("research/benchmarking/probe_replay_results.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ── human summary ────────────────────────────────────────────────────────
    print(f"\nextractors run: {extractors}  (v2 head: {v2_ok}, dense axis: {dense_ok})\n")
    print("PART 1 — extraction replay (Mode B probes)")
    for arm, data in report["part1_extraction"].items():
        print(f"\n  {arm.upper()}")
        gold_ids = [g[0] for g in GOLD[arm]]
        header = f"    {'probe':6} {'label':26} {'orig store':11}"
        for ex in extractors:
            header += f" {ex:>10}"
        print(header)
        for gid in gold_ids:
            base = data["baseline_store"][gid]
            row = f"    {gid:6} {base['label'][:26]:26} {'LIVE' if base['live'] else ('cap' if base['captured'] else 'MISS'):11}"
            for ex in extractors:
                g = data[ex]["gold"][gid]
                row += f" {'LIVE' if g['live'] else ('cap' if g['captured'] else 'MISS'):>10}"
            print(row)
    print("\nPART 2 — bind replay (Mode A probes)")
    print(f"    {'probe':26} {'K2 surfaced':12} {'K2 relevant':12} {'CK-1 surfaced':14} {'CK-1 relevant':13}")
    for pid, r in report["part2_bind"].items():
        print(f"    {pid:26} {str(r['k2_surfaced']):12} {str(r['k2_relevant']):12} "
              f"{str(r['ck1_surfaced']):14} {str(r['ck1_relevant']):13}")
    print(f"\nfull JSON: {out}")


if __name__ == "__main__":
    main()
