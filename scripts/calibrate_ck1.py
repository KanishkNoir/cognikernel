"""J4.3 — CK-1 gate calibration: replay real benchmark prompts through the gate.

Replays every user prompt from the gamma transcripts against a sandboxed copy
of the gamma DB, with the session-start block exposure recorded in the render
ledger first (so the redundancy filter behaves as in production).

Acceptance: injection rate 10-30%; zero off-topic injections (manual review of
the listing below); >=80% of PROBE prompts whose gold fact exists-and-not-in-
block get an injection containing it. Named regression fixture: the S3-P2
rate-limiter prompt (the measured D5/D16 failure) MUST inject the Redis fact.

Usage: python scripts/calibrate_ck1.py [--cold]

Caveat: the store contains all five sessions' events, so early-session prompts
see "future" facts — fine for gate precision/rate calibration, which is what
this measures.
"""
from __future__ import annotations

import argparse
import glob
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
sys.path.insert(0, str(_REPO / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

PID = "79e384f12a00c402"
SRC_DB = _REPO / ".bench_dbs" / "gamma_cogni.db"
TDIR = Path(r"C:\Users\Admin\.claude\projects\C--Users-Admin-OneDrive-Desktop-GAMMA-COGNI")
PROJECT = r"C:\Users\Admin\OneDrive\Desktop\GAMMA_COGNI"

# (prompt fragment, gold regex) for probe prompts — from replay_recall.
from replay_recall import PROBES  # noqa: E402  (same scripts/ dir)

NAMED_FIXTURE = ("Implement the rpm/tpm limiter", r"Redis|in-process counters fail")


def user_prompts(path: Path) -> list[str]:
    out = []
    for line in open(path, encoding="utf-8"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") != "user":
            continue
        c = (o.get("message") or {}).get("content")
        text = c if isinstance(c, str) else ""
        if isinstance(c, list):
            if all(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                continue
            text = "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")
        text = text.strip()
        if text and not text.startswith("<") and not text.startswith("Caveat:"):
            out.append(text)
    return out


def sweep(collected, config_base) -> None:
    """Offline gate sweep over the cached temporal retrievals."""
    import dataclasses

    from cognikernel.integration.query import select_ck1_hits

    grid = [
        (d, b, anchor, cap)
        for d in (3, 5)
        for b in (3, 5)
        for anchor in (0, 2, 3)
        for cap in (2, 3)
    ]
    print("\n d  b anchor cap |  rate  gold(eligible)  fixture")
    print("-" * 52)
    for d, b, anchor, cap in grid:
        cfg = dataclasses.replace(
            config_base, ck1_dense_rank_max=d, ck1_bm25_rank_max=b,
            ck1_dual_anchor_terms=anchor, ck1_max_events=cap,
        )
        n_inj = gold = fixture = 0
        eligible = 0
        for prompt, hits, seen, gold_rx, is_fixture, in_store, in_block in collected:
            passed = select_ck1_hits(hits, prompt, cfg, seen)
            snippet = " ".join(h["description"] for h in passed)
            if passed:
                n_inj += 1
            # Eligible = gold exists in the temporal store AND is not already
            # fully covered by the block (block coverage → silence is correct).
            if gold_rx is not None and in_store and not in_block:
                eligible += 1
                if passed and gold_rx.search(snippet):
                    gold += 1
            if is_fixture and passed and re.search(NAMED_FIXTURE[1], snippet, re.I):
                fixture = 1
        rate = 100 * n_inj / max(len(collected), 1)
        pct = 100 * gold / max(eligible, 1)
        print(f" {d}  {b}   {anchor}    {cap}  |  {rate:4.0f}%   {gold:>2}/{eligible}"
              f" ({pct:3.0f}%)   {'PASS' if fixture else 'fail'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cold", action="store_true", help="force BM25-only mode")
    ap.add_argument("--sweep", action="store_true", help="grid-sweep gate params")
    args = ap.parse_args()

    home = Path(tempfile.mkdtemp(prefix="cognikernel_ck1_"))
    (home / "projects").mkdir()
    # The embedding-model cache lives under COGNIKERNEL_DIR — carry the real one
    # into the sandbox so the warm path doesn't re-download 130MB per run.
    real_models = Path.home() / ".cognikernel" / "models"
    if real_models.exists():
        shutil.copytree(real_models, home / "models")
    os.environ["COGNIKERNEL_DIR"] = str(home)
    os.environ["COGNIKERNEL_DISABLE_AUTO_WARM"] = "1"

    if args.cold:
        import cognikernel.embedding.model as m
        m.is_ready = lambda: False  # type: ignore[assignment]
        m.warm = lambda: None  # type: ignore[assignment]
        print("mode: forced cold (BM25 only)")
    else:
        from cognikernel.embedding.model import ensure_ready
        print("dense axis:", "warm" if ensure_ready(timeout=90) else "COLD")

    from cognikernel.config import Config
    from cognikernel.integration.query import recall_for_prompt
    from cognikernel.integration.session import render_state
    from cognikernel.storage.projections import invalidate_projection

    config = Config.load(project_path=PROJECT)

    sessions = sorted(
        (p for p in TDIR.glob("*.jsonl") if p.stat().st_size > 100_000),
        key=lambda p: p.stat().st_mtime,
    )
    stems = [p.stem for p in sessions]
    probe_rx = [(frag, re.compile(gold, re.IGNORECASE | re.DOTALL))
                for _, frag, gold in [(i, p[1][:60], p[2]) for i, p in enumerate(PROBES)]]

    from cognikernel.retrieval.hybrid import hybrid_recall
    from cognikernel.storage.render_ledger import rendered_event_ids

    # Phase 1 — collect temporal retrievals once: (prompt, hits, block-seen,
    # gold regex | None, is_fixture). Ledger growth from ck1 injections within
    # a session is approximated as block-only (second-order for calibration).
    collected = []
    db_path = home / "projects" / f"{PID}.db"
    for si, spath in enumerate(sessions, 1):
        session_id = f"calib-s{si}"
        # TEMPORAL replay: session si sees only PRIOR sessions' events — a
        # faithful simulation of what the store held when these prompts fired
        # (the anachronistic full-store replay let S3's own wrong-value events
        # outrank the S1 facts the real S3 prompt would have hit).
        shutil.copy(SRC_DB, db_path)
        prior = stems[: si - 1]
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        ph = ",".join("?" * len(prior)) or "''"
        conn.execute(f"DELETE FROM events WHERE session_id NOT IN ({ph})", prior)
        conn.commit()
        invalidate_projection(conn, PID)
        conn.close()
        # Simulate session start: block exposure lands in the ledger.
        render_state(PROJECT, config, session_id=session_id)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        seen = rendered_event_ids(conn, PID, session_id)
        actives = conn.execute(
            "SELECT id, json_extract(payload,'$.description') d FROM events "
            "WHERE project_id=? AND archived=0 AND superseded_by IS NULL",
            (PID,),
        ).fetchall()
        for prompt in user_prompts(spath):
            hits = hybrid_recall(conn, PID, prompt, k=8, n_per_axis=10)
            gold_rx = None
            for frag, rx in probe_rx:
                if prompt[:60].startswith(frag[:40]):
                    gold_rx = rx
                    break
            # Acceptance metric needs: does the gold fact exist in THIS
            # temporal store, and is it already in the block (→ silence is
            # the CORRECT behavior, not a miss)?
            gold_ids: set[int] = set()
            if gold_rx is not None:
                gold_ids = {r["id"] for r in actives if gold_rx.search(r["d"] or "")}
            collected.append((
                prompt, hits, seen, gold_rx,
                prompt.startswith(NAMED_FIXTURE[0]),
                bool(gold_ids), bool(gold_ids) and gold_ids <= seen,
            ))
        conn.close()

    if args.sweep:
        sweep(collected, config)
        return

    # Phase 2 — evaluate the shipped defaults.
    from cognikernel.integration.query import select_ck1_hits

    n_inject = probe_hits = eligible = 0
    fixture_ok = False
    injections: list[tuple[str, str]] = []
    for prompt, hits, seen, gold_rx, is_fixture, in_store, in_block in collected:
        passed = select_ck1_hits(hits, prompt, config, seen)
        snippet = " | ".join(h["description"][:90] for h in passed)
        if passed:
            n_inject += 1
            injections.append((prompt[:70], snippet[:170]))
        if gold_rx is not None and in_store and not in_block:
            eligible += 1
            if passed and gold_rx.search(snippet):
                probe_hits += 1
        if is_fixture:
            print(f"\nFIXTURE hits for: {prompt[:60]}")
            for h in hits:
                mark = "*" if re.search(NAMED_FIXTURE[1], h["description"], re.I) else " "
                print(f"  {mark} d={h['dense_rank']} b={h['bm25_rank']} "
                      f"seen={h['id'] in seen}  {h['description'][:85]}")
            if passed and re.search(NAMED_FIXTURE[1], snippet, re.I):
                fixture_ok = True

    rate = 100.0 * n_inject / max(len(collected), 1)
    print(f"\nprompts: {len(collected)}  injections: {n_inject}  rate: {rate:.0f}%  "
          f"(target 10-30%)")
    print(f"gold injected for eligible probes: {probe_hits}/{eligible} "
          f"(target >=80%)")
    print(f"NAMED FIXTURE (S3-P2 rpm/tpm limiter -> Redis): "
          f"{'PASS' if fixture_ok else 'FAIL'}")
    print("\nall injections (off-topic review):")
    for p, s in injections:
        print(f"  Q: {p}")
        print(f"     -> {s}")


if __name__ == "__main__":
    main()
