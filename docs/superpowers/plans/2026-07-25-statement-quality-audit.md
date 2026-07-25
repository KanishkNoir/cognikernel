# Statement Quality Audit (Phase A0 + Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how often CogniKernel's stored memory statements are defective, with human labels, so the go/no-go threshold for building a generator is decided by evidence rather than by a hunch from 11 hand-read examples.

**Architecture:** Two standalone research scripts following the `scripts/model_eval.py` pattern — one sweeps all project stores and emits a blinded, stratified labeling pool; one reads completed human labels and computes rates, confidence intervals, and inter-labeler agreement. Pure helper functions live at module top and are covered by a pytest gate in `tests/eval/`. No `src/cognikernel/` changes: this is research tooling, and the shipped package is guarded by import-linter layer contracts.

**Tech Stack:** Python 3.11/3.12, stdlib only (`sqlite3`, `json`, `re`, `random`, `math`, `hashlib`, `collections`), pytest. No new dependencies, no network, no API keys.

## Global Constraints

Copied from `docs/superpowers/specs/2026-07-25-memory-statement-generation-design.md`:

- **Phase B (eval + generator) is out of scope.** This plan stops at a measured number and a documented decision. Do not build a generator, do not call a teacher API, do not touch `.env`.
- **Pre-registered Phase A gate:** Phase B proceeds only if the human-labeled defect rate on `DECISION` + `APPROACH_ABANDONED_DO_NOT_RETRY` is **≥ 20%**.
- **Pre-registered Phase A0 gate** (set here, before any number is seen): run full Phase A if the clean-bucket defect rate is **≥ 25%**; abandon the branch if **< 15%**; if between, extend the pilot to n=150 before deciding. Record the decision in writing before proceeding either way.
- **Heuristics are a proxy, never the result.** Every reported defect rate must come from human labels. The heuristic sweep is reported only alongside its measured miss rate against those labels.
- **Labelers must not see heuristic verdicts or store/project identity.** Pool files carry only `id`, `event_type`, `text`. Metadata lives in a sidecar keyed by `id`.
- **Read-only against stores.** Open every project DB with `mode=ro`. This tooling must never write to `~/.cognikernel/`.
- Windows is a supported dev platform: every script starts with `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` (stores contain em-dashes and box-drawing characters that crash cp1252 stdout).
- Outputs go to `research/statement_audit/`, timestamped, following `scripts/model_eval.py`.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/audit_statement_quality.py` (create) | Sweep stores → classify by heuristic → stratified sample → write blinded pool + sidecar meta + heuristic summary |
| `scripts/audit_report.py` (create) | Read completed labels → defect rates, Wilson CIs, Cohen's kappa, heuristic miss rate → results JSON + printed report |
| `tests/eval/test_statement_audit.py` (create) | pytest gate over the pure helpers in both scripts |
| `research/statement_audit/` (create) | pool + meta (gitignored — raw project memory), heuristic summary and results JSON (committed) |
| `docs/superpowers/specs/2026-07-25-memory-statement-generation-design.md` (modify, Task 6) | record the A0 outcome and the decision |

Scripts are not a package (nothing in `tests/` imports `scripts/` today), so the test loads them by path with `importlib.util`. This avoids adding `scripts/__init__.py` and changing how 40+ existing scripts resolve.

---

## The labeling codebook

This is the instrument. It must be in the plan verbatim because label quality *is* the result.

A labeler marks **all** codes that apply. `CLEAN` is exclusive — if `CLEAN` is marked, nothing else may be.

| Code | Definition | Example from the real store |
|---|---|---|
| `DANGLING_REFERENCE` | Opens with or turns on a pronoun / deictic / definite reference whose referent is not in the statement itself. | *"With it, you see three sibling spans with relay.attempt=0,1,2…"* |
| `NOT_A_STATEMENT` | A fragment, table row, heading, or code debris rather than a sentence asserting a fact. | *"Timestamp column type (virtual_keys): TIMESTAMPTZ"* |
| `WRONG_TYPE` | Content does not match its event type — e.g. a definition or rationale stored as `CONSTRAINT_HARD`, an explanation stored as `DECISION`. | `CONSTRAINT_HARD`: *"Non-zero temperature means the caller explicitly wants non-deterministic output."* |
| `NOT_DURABLE` | Narration, status chatter, a question, or a prompt fragment. Should never have been stored at all. **These become abstain-gold in Phase B.** | `DECISION`: *"Now update the cache lookup in router.py to branch on streaming."* |
| `META_TALK` | About CogniKernel / the agent tooling itself rather than the host project. | *"The overridden constraints will be updated by the Stop hook when the session closes."* |
| `COMPOUND` | Carries 2+ independent facts that belong in separate events. | *"Also deferred: learned canonicalizer, cross-encoder reranking, decay tuning, and anything Hopfield-flavored."* |
| `MISSING_SUBJECT` | No grammatical subject, so the reader cannot tell what the statement is about. | *"Fails the build if any file other than app/auth/jwt.py calls .get_secret_value()."* |
| `OTHER` | Defective for a reason not listed. **Requires a note.** | — |
| `CLEAN` | None of the above. Usable as-is. | *"All upstream timeouts surface to the client as 504 Gateway Timeout, never as 500."* |

**Judge the statement standalone.** The question is whether *this text alone*, injected into a future session as authoritative memory, is correct and comprehensible. Do not open the source transcript to resolve a reference — needing to is exactly what `DANGLING_REFERENCE` measures.

---

### Task 1: Defect heuristics and store sweep

**Files:**
- Create: `scripts/audit_statement_quality.py`
- Create: `tests/eval/test_statement_audit.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `DEFECT_CODES: tuple[str, ...]` — the codebook codes excluding `CLEAN`
  - `classify_heuristic(text: str) -> set[str]` — surface-detectable codes only; empty set means "no flag raised"
  - `iter_statements(store_dir: Path) -> Iterator[dict]` — yields `{"store": str, "event_type": str, "text": str}` for active typed events

- [ ] **Step 1: Write the failing test**

Create `tests/eval/test_statement_audit.py`:

```python
"""Phase A statement-quality audit — pytest gate over the audit helpers.

The heuristics here are a deliberately crude PROXY. Their job is to split the
corpus into strata for sampling, not to produce the reported defect rate — that
comes from human labels only (see the spec's Phase A section). These tests pin
the heuristics' stated behaviour so a regex typo cannot silently reshape the
strata.

LIMITATION: passing these tests says the heuristics do what they claim, NOT that
they detect defects well. Their real miss rate is measured against human labels
in scripts/audit_report.py and must be reported alongside any heuristic number.
"""
from __future__ import annotations

import collections
import importlib.util
import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = _ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


audit = _load("audit_statement_quality")


@pytest.mark.parametrize("text,expected", [
    ("With it, you see three sibling spans.", "DANGLING_REFERENCE"),
    ("But the corpus also carries a clean negative result.", "DANGLING_REFERENCE"),
    ("Now update the cache lookup in router.py.", "NOT_DURABLE"),
    ("Let me check the router config first.", "NOT_DURABLE"),
    ("The Stop hook will persist this at session close.", "META_TALK"),
    ("Network errors are transient blips. |.", "NOT_A_STATEMENT"),
    ("TIMESTAMPTZ", "NOT_A_STATEMENT"),
])
def test_classify_heuristic_flags_known_defects(text, expected):
    assert expected in audit.classify_heuristic(text)


def test_classify_heuristic_passes_clean_statements():
    clean = "All upstream timeouts surface to the client as 504 Gateway Timeout, never as 500."
    assert audit.classify_heuristic(clean) == set()


def test_classify_heuristic_returns_only_codebook_codes():
    got = audit.classify_heuristic("Now update it. |.")
    assert got <= set(audit.DEFECT_CODES)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_statement_audit.py -v`
Expected: FAIL — `FileNotFoundError` / `ModuleNotFoundError` for `scripts/audit_statement_quality.py`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/audit_statement_quality.py`:

```python
"""Phase A statement-quality audit — sweep stores, emit a blinded labeling pool.

Reads every CogniKernel project store read-only, buckets stored memory
statements by a crude surface heuristic, and writes a stratified, blinded sample
for human labeling plus a sidecar of the metadata the labeler must not see.

The heuristic is a PROXY used to stratify sampling. The reported defect rate
comes from human labels only; scripts/audit_report.py measures how badly the
heuristic misses.

Writes:
  research/statement_audit/pool_<stamp>.jsonl        (labeler opens this)
  research/statement_audit/pool_meta_<stamp>.jsonl   (blinded metadata, keyed by id)
  research/statement_audit/heuristic_<stamp>.json    (corpus-wide sweep summary)

Usage: uv run python scripts/audit_statement_quality.py [--n 60] [--stratum clean]
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import random
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Iterator

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("research/statement_audit")
STORE_DIR = Path.home() / ".cognikernel" / "projects"
TYPES = ("DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
         "APPROACH_ABANDONED_DO_NOT_RETRY")

DEFECT_CODES = (
    "DANGLING_REFERENCE", "NOT_A_STATEMENT", "WRONG_TYPE", "NOT_DURABLE",
    "META_TALK", "COMPOUND", "MISSING_SUBJECT", "OTHER",
)

# Surface heuristics only. WRONG_TYPE, COMPOUND, MISSING_SUBJECT and OTHER are
# deliberately absent — they are semantic and no regex detects them, which is
# precisely why human labeling is required.
_RX = (
    ("DANGLING_REFERENCE", re.compile(
        r"^(with (it|this|that)\b|it\s|this\s|that\s|these\s|those\s|they\s"
        r"|but\s|however\b|the (above|latter|former|old|new)\b|old\s|new\s)", re.I)),
    ("NOT_DURABLE", re.compile(
        r"^(now\s|let me\b|let's\b|i'll\b|i will\b|i need to\b|next[,\s]|then\s"
        r"|first[,\s]|going to\b|update\s|add\s|read\s|check\s)", re.I)),
    ("META_TALK", re.compile(
        r"\b(stop hook|session context|cognikernel|claude\.md|memory block"
        r"|the recall|injected block)\b", re.I)),
    ("NOT_A_STATEMENT", re.compile(
        r"(\|\s*\.?\s*$|^[-+|=\s]+$|^\W{0,3}$|\.\.\.$|[─-╿]|—\s*$)")),
)


def classify_heuristic(text: str) -> set[str]:
    """Surface-detectable defect codes for one statement. Empty = no flag."""
    hits = {code for code, rx in _RX if rx.search(text)}
    if len(text.split()) < 5:
        hits.add("NOT_A_STATEMENT")
    return hits


def iter_statements(store_dir: Path) -> Iterator[dict]:
    """Yield active typed statements from every store, read-only."""
    placeholders = ",".join("?" * len(TYPES))
    for db in sorted(glob.glob(str(store_dir / "*.db"))):
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        except sqlite3.Error:
            continue  # a store that won't open is skipped, not fatal
        try:
            rows = conn.execute(
                f"SELECT event_type, payload FROM events "
                f"WHERE event_type IN ({placeholders}) AND archived=0", TYPES
            ).fetchall()
        except sqlite3.Error:
            continue  # older schema without this table/column
        finally:
            conn.close()
        for event_type, payload in rows:
            try:
                text = (json.loads(payload).get("description") or "").strip()
            except (json.JSONDecodeError, AttributeError):
                continue
            if text:
                yield {"store": Path(db).stem, "event_type": event_type, "text": text}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_statement_audit.py -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_statement_quality.py tests/eval/test_statement_audit.py
git commit -m "feat(audit): defect heuristics and read-only store sweep"
```

---

### Task 2: Stratified blinded sampling and pool emission

**Files:**
- Modify: `scripts/audit_statement_quality.py` (append to Task 1's module)
- Modify: `tests/eval/test_statement_audit.py` (append)

**Interfaces:**
- Consumes: `classify_heuristic`, `iter_statements`, `DEFECT_CODES` from Task 1
- Produces:
  - `statement_id(store: str, text: str) -> str` — stable 12-hex id
  - `stratified_sample(rows: list[dict], n: int, stratum: str, seed: int) -> list[dict]` — `stratum` is `"clean"` (heuristic found nothing), `"flagged"`, or `"all"`; balanced across `event_type`, deterministic under `seed`
  - `main()` — CLI writing pool, meta, and heuristic-summary files

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_statement_audit.py`:

```python
def _rows(n_per_type=25):
    out = []
    for t in ("DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT",
              "APPROACH_ABANDONED_DO_NOT_RETRY"):
        for i in range(n_per_type):
            out.append({"store": f"s{i % 3}", "event_type": t,
                        "text": f"The {t.lower()} number {i} is recorded plainly here."})
    return out


def test_statement_id_is_stable_and_store_scoped():
    a = audit.statement_id("store1", "same text")
    assert a == audit.statement_id("store1", "same text")
    assert a != audit.statement_id("store2", "same text")
    assert len(a) == 12


def test_stratified_sample_is_deterministic_under_seed():
    rows = _rows()
    first = audit.stratified_sample(rows, n=20, stratum="all", seed=7)
    second = audit.stratified_sample(rows, n=20, stratum="all", seed=7)
    assert [r["text"] for r in first] == [r["text"] for r in second]
    assert audit.stratified_sample(rows, n=20, stratum="all", seed=8) != first


def test_stratified_sample_balances_event_types():
    got = audit.stratified_sample(_rows(), n=20, stratum="all", seed=1)
    counts = collections.Counter(r["event_type"] for r in got)
    assert len(got) == 20
    assert max(counts.values()) - min(counts.values()) <= 1


def test_clean_stratum_excludes_heuristically_flagged():
    rows = _rows() + [{"store": "s9", "event_type": "DECISION",
                       "text": "Now update the router config."}]
    got = audit.stratified_sample(rows, n=30, stratum="clean", seed=3)
    assert all(audit.classify_heuristic(r["text"]) == set() for r in got)


def test_stratified_sample_caps_at_available_rows():
    assert len(audit.stratified_sample(_rows(2), n=500, stratum="all", seed=1)) == 8
```

Add `import collections` to the test file's imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_statement_audit.py -v`
Expected: FAIL — `AttributeError: module has no attribute 'statement_id'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/audit_statement_quality.py`:

```python
def statement_id(store: str, text: str) -> str:
    """Stable id for one statement. Store-scoped so identical text in two
    projects stays two rows — cross-project repetition is itself a finding."""
    return hashlib.sha256(f"{store}\x00{text}".encode("utf-8")).hexdigest()[:12]


def stratified_sample(rows: list[dict], n: int, stratum: str, seed: int) -> list[dict]:
    """Deterministic sample balanced across event_type.

    stratum: "clean" (heuristic found nothing) | "flagged" | "all".
    Returns fewer than n when the pool is smaller. Round-robins across types so
    a rare type (APPROACH_ABANDONED) is not swamped by DECISION.
    """
    if stratum == "clean":
        pool = [r for r in rows if not classify_heuristic(r["text"])]
    elif stratum == "flagged":
        pool = [r for r in rows if classify_heuristic(r["text"])]
    else:
        pool = list(rows)

    by_type: dict[str, list[dict]] = collections.defaultdict(list)
    for r in pool:
        by_type[r["event_type"]].append(r)

    rng = random.Random(seed)
    for bucket in by_type.values():
        bucket.sort(key=lambda r: (r["store"], r["text"]))  # stable pre-shuffle order
        rng.shuffle(bucket)

    out: list[dict] = []
    types = sorted(by_type)
    while len(out) < n and any(by_type[t] for t in types):
        for t in types:
            if by_type[t] and len(out) < n:
                out.append(by_type[t].pop())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="statements to sample")
    ap.add_argument("--stratum", choices=("clean", "flagged", "all"), default="clean")
    ap.add_argument("--seed", type=int, default=1106)
    ap.add_argument("--store-dir", type=Path, default=STORE_DIR)
    args = ap.parse_args()

    rows = list(iter_statements(args.store_dir))
    if not rows:
        sys.exit(f"no statements found under {args.store_dir}")

    counts = collections.Counter()
    per_type = collections.defaultdict(collections.Counter)
    for r in rows:
        hits = classify_heuristic(r["text"])
        for h in hits:
            counts[h] += 1
            per_type[r["event_type"]][h] += 1
        counts["_any" if hits else "_none"] += 1
        per_type[r["event_type"]]["_any" if hits else "_none"] += 1

    sample = stratified_sample(rows, args.n, args.stratum, args.seed)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pool_path = OUT_DIR / f"pool_{stamp}.jsonl"
    meta_path = OUT_DIR / f"pool_meta_{stamp}.jsonl"
    with pool_path.open("w", encoding="utf-8") as pf, \
         meta_path.open("w", encoding="utf-8") as mf:
        for r in sample:
            sid = statement_id(r["store"], r["text"])
            # BLINDED: labeler sees only id, type, text. No store, no heuristic.
            pf.write(json.dumps({"id": sid, "event_type": r["event_type"],
                                 "text": r["text"], "labels": [], "notes": ""},
                                ensure_ascii=False) + "\n")
            mf.write(json.dumps({"id": sid, "store": r["store"],
                                 "heuristic": sorted(classify_heuristic(r["text"])),
                                 "stratum": args.stratum, "seed": args.seed},
                                ensure_ascii=False) + "\n")

    summary = {
        "stamp": stamp, "stores_scanned": len(set(r["store"] for r in rows)),
        "statements_total": len(rows), "sampled": len(sample),
        "stratum": args.stratum, "seed": args.seed,
        "heuristic_counts": dict(counts),
        "heuristic_by_type": {k: dict(v) for k, v in per_type.items()},
    }
    (OUT_DIR / f"heuristic_{stamp}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"statements: {len(rows)} across {summary['stores_scanned']} stores")
    print(f"heuristic flagged: {counts['_any']} "
          f"({100 * counts['_any'] / len(rows):.1f}%)")
    print(f"\nwrote {pool_path}  ({len(sample)} to label)")
    print(f"wrote {meta_path}  (do NOT open before labeling)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_statement_audit.py -v`
Expected: PASS (14 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_statement_quality.py tests/eval/test_statement_audit.py
git commit -m "feat(audit): deterministic stratified sampling and blinded pool output"
```

---

### Task 3: Statistics — Wilson CI and Cohen's kappa

**Files:**
- Create: `scripts/audit_report.py`
- Modify: `tests/eval/test_statement_audit.py` (append)

**Interfaces:**
- Consumes: nothing from Tasks 1–2 (pure stats, kept separate so it is testable without stores)
- Produces:
  - `wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]`
  - `cohens_kappa(a: list[set[str]], b: list[set[str]], codes: tuple[str, ...]) -> dict[str, float]` — per-code kappa plus `"_mean"`

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_statement_audit.py`:

```python
report = _load("audit_report")


def test_wilson_ci_known_value():
    # 15/60 = 0.25; Wilson 95% CI is approx (0.158, 0.372)
    lo, hi = report.wilson_ci(15, 60)
    assert lo == pytest.approx(0.158, abs=0.002)
    assert hi == pytest.approx(0.372, abs=0.002)


def test_wilson_ci_handles_zero_and_full():
    # At p=0 the centre and half-width are mathematically equal, so the bound is
    # 0 up to float error — assert with a tolerance, not ==.
    assert report.wilson_ci(0, 30)[0] == pytest.approx(0.0, abs=1e-9)
    assert report.wilson_ci(30, 30)[1] == pytest.approx(1.0, abs=1e-9)


def test_wilson_ci_empty_sample_is_full_interval():
    assert report.wilson_ci(0, 0) == (0.0, 1.0)


def test_cohens_kappa_perfect_agreement():
    a = [{"WRONG_TYPE"}, set(), {"COMPOUND"}, set()]
    k = report.cohens_kappa(a, list(a), ("WRONG_TYPE", "COMPOUND"))
    assert k["WRONG_TYPE"] == pytest.approx(1.0)
    assert k["_mean"] == pytest.approx(1.0)


def test_cohens_kappa_chance_agreement_is_zero():
    # A says code on first half, B says code on alternating items -> ~chance
    a = [{"X"}, {"X"}, set(), set()]
    b = [{"X"}, set(), {"X"}, set()]
    assert report.cohens_kappa(a, b, ("X",))["X"] == pytest.approx(0.0, abs=1e-9)


def test_cohens_kappa_degenerate_column_is_none_not_crash():
    # Neither labeler ever used the code — kappa undefined, must not divide by zero
    k = report.cohens_kappa([set(), set()], [set(), set()], ("NEVER",))
    assert k["NEVER"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_statement_audit.py -v`
Expected: FAIL — `FileNotFoundError` for `scripts/audit_report.py`.

- [ ] **Step 3: Write minimal implementation**

Create `scripts/audit_report.py`:

```python
"""Phase A statement-quality audit — compute rates from completed human labels.

Reads a labeled pool (pool_<stamp>.jsonl with `labels` filled in) plus its
sidecar meta, and reports the human-labeled defect rate with Wilson confidence
intervals, per-code and per-event-type breakdowns, inter-labeler agreement when
a second labeler file is supplied, and the heuristic's measured miss rate.

The human number is the result. The heuristic number is reported only as a
proxy, with its miss rate attached.

Usage: uv run python scripts/audit_report.py research/statement_audit/pool_<stamp>.jsonl \
           [--labeler-b path.jsonl] [--gate-types DECISION,APPROACH_ABANDONED_DO_NOT_RETRY]
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT_DIR = Path("research/statement_audit")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. Correct at the boundaries, unlike normal approx."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def cohens_kappa(a: list[set[str]], b: list[set[str]],
                 codes: tuple[str, ...]) -> dict:
    """Per-code binary Cohen's kappa plus the mean over defined codes.

    Returns None for a code neither labeler ever applied (kappa undefined
    rather than zero — reporting 0.0 there would understate agreement).
    """
    out: dict = {}
    defined = []
    for code in codes:
        n = len(a)
        both = sum(1 for x, y in zip(a, b) if code in x and code in y)
        neither = sum(1 for x, y in zip(a, b) if code not in x and code not in y)
        pa = (both + neither) / n if n else 0.0
        pa_yes = sum(1 for x in a if code in x) / n if n else 0.0
        pb_yes = sum(1 for x in b if code in x) / n if n else 0.0
        pe = pa_yes * pb_yes + (1 - pa_yes) * (1 - pb_yes)
        if pe >= 1.0:
            out[code] = None
            continue
        out[code] = (pa - pe) / (1 - pe)
        defined.append(out[code])
    out["_mean"] = sum(defined) / len(defined) if defined else None
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_statement_audit.py -v`
Expected: PASS (20 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_report.py tests/eval/test_statement_audit.py
git commit -m "feat(audit): Wilson CI and per-code Cohen's kappa"
```

---

### Task 4: Report assembly and the pre-registered gate

**Files:**
- Modify: `scripts/audit_report.py` (append)
- Modify: `tests/eval/test_statement_audit.py` (append)

**Interfaces:**
- Consumes: `wilson_ci`, `cohens_kappa` from Task 3
- Produces:
  - `load_labeled(path: Path) -> list[dict]` — raises `ValueError` naming unlabeled ids
  - `defect_rate(rows: list[dict]) -> tuple[int, int]` — `(defective, total)`; defective = any code other than `CLEAN`
  - `heuristic_miss_rate(rows, meta) -> dict` — human-defective items the heuristic missed
  - `main()` — prints the report, writes `results_<stamp>.json`, and prints the gate verdict

- [ ] **Step 1: Write the failing test**

Append to `tests/eval/test_statement_audit.py`:

```python
def _labeled(tmp_path, rows):
    p = tmp_path / "pool.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_load_labeled_rejects_unlabeled_rows(tmp_path):
    p = _labeled(tmp_path, [
        {"id": "a1", "event_type": "DECISION", "text": "x", "labels": ["CLEAN"]},
        {"id": "b2", "event_type": "DECISION", "text": "y", "labels": []},
    ])
    with pytest.raises(ValueError, match="b2"):
        report.load_labeled(p)


def test_defect_rate_counts_any_non_clean_code():
    rows = [{"labels": ["CLEAN"]}, {"labels": ["WRONG_TYPE"]},
            {"labels": ["COMPOUND", "MISSING_SUBJECT"]}, {"labels": ["CLEAN"]}]
    assert report.defect_rate(rows) == (2, 4)


def test_clean_is_exclusive_and_rejected_when_mixed(tmp_path):
    p = _labeled(tmp_path, [{"id": "c3", "event_type": "DECISION", "text": "z",
                             "labels": ["CLEAN", "WRONG_TYPE"]}])
    with pytest.raises(ValueError, match="c3"):
        report.load_labeled(p)


def test_other_requires_a_note(tmp_path):
    p = _labeled(tmp_path, [{"id": "d4", "event_type": "DECISION", "text": "z",
                             "labels": ["OTHER"], "notes": ""}])
    with pytest.raises(ValueError, match="d4"):
        report.load_labeled(p)


def test_heuristic_miss_rate_counts_human_defects_heuristic_missed():
    rows = [{"id": "a", "labels": ["WRONG_TYPE"]}, {"id": "b", "labels": ["CLEAN"]},
            {"id": "c", "labels": ["COMPOUND"]}]
    meta = {"a": {"heuristic": []}, "b": {"heuristic": []},
            "c": {"heuristic": ["NOT_DURABLE"]}}
    got = report.heuristic_miss_rate(rows, meta)
    assert got["human_defective"] == 2
    assert got["missed_by_heuristic"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/eval/test_statement_audit.py -v`
Expected: FAIL — `AttributeError: module has no attribute 'load_labeled'`.

- [ ] **Step 3: Write minimal implementation**

Append to `scripts/audit_report.py`:

```python
CODES = ("DANGLING_REFERENCE", "NOT_A_STATEMENT", "WRONG_TYPE", "NOT_DURABLE",
         "META_TALK", "COMPOUND", "MISSING_SUBJECT", "OTHER")


def load_labeled(path: Path) -> list[dict]:
    """Load a labeled pool, rejecting incomplete or contradictory rows loudly.

    A silently-skipped unlabeled row would bias the rate toward whatever the
    labeler happened to finish first, so this raises instead.
    """
    rows = [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    unlabeled, mixed, unnoted, unknown = [], [], [], []
    for r in rows:
        labels = r.get("labels") or []
        if not labels:
            unlabeled.append(r["id"])
        if "CLEAN" in labels and len(labels) > 1:
            mixed.append(r["id"])
        if "OTHER" in labels and not (r.get("notes") or "").strip():
            unnoted.append(r["id"])
        bad = [c for c in labels if c != "CLEAN" and c not in CODES]
        if bad:
            unknown.append(f"{r['id']}:{','.join(bad)}")
    problems = []
    if unlabeled:
        problems.append(f"unlabeled: {', '.join(unlabeled)}")
    if mixed:
        problems.append(f"CLEAN mixed with other codes: {', '.join(mixed)}")
    if unnoted:
        problems.append(f"OTHER without a note: {', '.join(unnoted)}")
    if unknown:
        problems.append(f"unknown codes: {', '.join(unknown)}")
    if problems:
        raise ValueError("; ".join(problems))
    return rows


def defect_rate(rows: list[dict]) -> tuple[int, int]:
    """(defective, total). Defective = carries any code other than CLEAN."""
    bad = sum(1 for r in rows
              if any(c != "CLEAN" for c in (r.get("labels") or [])))
    return bad, len(rows)


def heuristic_miss_rate(rows: list[dict], meta: dict) -> dict:
    """How often the surface heuristic missed a human-labeled defect."""
    human_bad = [r for r in rows
                 if any(c != "CLEAN" for c in (r.get("labels") or []))]
    missed = [r for r in human_bad if not meta.get(r["id"], {}).get("heuristic")]
    return {
        "human_defective": len(human_bad),
        "missed_by_heuristic": len(missed),
        "miss_rate": len(missed) / len(human_bad) if human_bad else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", type=Path)
    ap.add_argument("--labeler-b", type=Path, default=None)
    ap.add_argument("--gate-types", default="DECISION,APPROACH_ABANDONED_DO_NOT_RETRY")
    ap.add_argument("--gate-threshold", type=float, default=0.20)
    args = ap.parse_args()

    rows = load_labeled(args.pool)
    meta_path = args.pool.with_name(args.pool.name.replace("pool_", "pool_meta_"))
    meta = {}
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                meta[rec["id"]] = rec

    bad, n = defect_rate(rows)
    lo, hi = wilson_ci(bad, n)
    print(f"labeled            : {n}")
    print(f"defective (human)  : {bad}  = {100 * bad / n:.1f}%  "
          f"[95% CI {100 * lo:.1f}-{100 * hi:.1f}%]")

    print("\n-- per code --")
    code_counts = collections.Counter(
        c for r in rows for c in (r.get("labels") or []) if c != "CLEAN")
    for c, v in code_counts.most_common():
        print(f"  {c:20s} {v:4d}  {100 * v / n:5.1f}%")

    print("\n-- per event type --")
    by_type = collections.defaultdict(list)
    for r in rows:
        by_type[r.get("event_type", "?")].append(r)
    for t, trows in sorted(by_type.items()):
        tb, tn = defect_rate(trows)
        tlo, thi = wilson_ci(tb, tn)
        print(f"  {t:32s} n={tn:4d}  {100 * tb / tn:5.1f}%  "
              f"[{100 * tlo:.1f}-{100 * thi:.1f}%]")

    miss = heuristic_miss_rate(rows, meta) if meta else None
    if miss:
        rate = miss["miss_rate"]
        print(f"\nheuristic miss rate: {miss['missed_by_heuristic']}/"
              f"{miss['human_defective']}"
              + (f" = {100 * rate:.1f}%" if rate is not None else " (n/a)"))

    kappa = None
    if args.labeler_b:
        b_rows = load_labeled(args.labeler_b)
        b_by_id = {r["id"]: set(r.get("labels") or []) for r in b_rows}
        shared = [r for r in rows if r["id"] in b_by_id]
        if shared:
            kappa = cohens_kappa([set(r["labels"]) for r in shared],
                                 [b_by_id[r["id"]] for r in shared], CODES)
            mean = kappa["_mean"]
            shown = f"{mean:.2f}" if mean is not None else "undefined"
            print(f"\ninter-labeler kappa (n={len(shared)}): mean {shown}")

    gate_types = [t.strip() for t in args.gate_types.split(",") if t.strip()]
    gate_rows = [r for r in rows if r.get("event_type") in gate_types]
    verdict = None
    if gate_rows:
        gb, gn = defect_rate(gate_rows)
        glo, ghi = wilson_ci(gb, gn)
        rate = gb / gn
        verdict = "PROCEED" if rate >= args.gate_threshold else "STOP"
        print(f"\n-- pre-registered gate ({'+'.join(gate_types)}) --")
        print(f"  rate {100 * rate:.1f}% [{100 * glo:.1f}-{100 * ghi:.1f}%] "
              f"vs threshold {100 * args.gate_threshold:.0f}%  ->  {verdict}")
        print("  NOTE: the point estimate decides the gate as pre-registered; "
              "the CI is reported for honesty, not to move the line after the fact.")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"results_{stamp}.json").write_text(json.dumps({
        "stamp": stamp, "pool": str(args.pool), "n": n, "defective": bad,
        "rate": bad / n, "ci95": [lo, hi], "per_code": dict(code_counts),
        "heuristic_miss": miss, "kappa": kappa,
        "gate": {"types": gate_types, "threshold": args.gate_threshold,
                 "verdict": verdict},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_DIR / f'results_{stamp}.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/eval/test_statement_audit.py -v`
Expected: PASS (25 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/audit_report.py tests/eval/test_statement_audit.py
git commit -m "feat(audit): report assembly with pre-registered gate verdict"
```

---

### Task 5: Run the A0 pilot

**Files:**
- Create: `research/statement_audit/*` (generated, gitignored)
- Modify: `.gitignore` (add `research/statement_audit/pool*.jsonl` — raw statements contain private project content)

**Interfaces:**
- Consumes: both CLIs from Tasks 1–4
- Produces: a labeled pool and a `results_<stamp>.json` carrying the A0 verdict

**This task is mostly human work.** The labeling cannot be delegated to a model: the whole point is a human number to measure heuristics (and later, a generator) against.

- [ ] **Step 1: Add the gitignore entry**

```bash
printf '\n# statement-audit pools contain raw project memory\nresearch/statement_audit/pool*.jsonl\n' >> .gitignore
git add .gitignore && git commit -m "chore: gitignore statement-audit pools"
```

- [ ] **Step 2: Generate the clean-bucket pilot pool**

Run: `uv run python scripts/audit_statement_quality.py --n 60 --stratum clean`
Expected: prints corpus totals (6,434 statements across 45 contributing stores; 163 store files scanned, 118 holding no typed statements) and writes three files under `research/statement_audit/`.

- [ ] **Step 3 — AMENDED during execution (human decision): LLM pre-labels, human verifies a subset**

The original step had the human label all 60. The human elected instead to have
hosted models pre-label, with human verification of a subset. **What this costs
is stated plainly, because it changes what the number means:** the reported rate
becomes a model's notion of "defective", anchored to human judgment only on the
verified subset and only as strongly as the measured agreement. The
pre-registered gate is decided on that basis, and the writeup must say so.

Why it is nonetheless defensible: the same pattern (teacher-written, human
spot-checked) is already the spec's design for Phase B gold data, and
`scripts/build_cot_sft.py` established the precedent in this repo.

**This does not touch the no-LLM promise.** That is a *runtime* constraint —
nothing leaves the machine during a session, no key needed to use CogniKernel.
This is offline eval construction. Verified: the wheel packages only
`src/cognikernel` (`pyproject.toml:50`), so nothing under `scripts/` ships.

**Sub-step 3a — three models label all 60 independently.** See Task 5A below.

**Sub-step 3b — human verifies a stratified 20.** The verification file is
generated from the pool, blinded to the models' answers, and labeled by hand
using the codebook. Agreement is then reported as Cohen's kappa per model
against the human subset. **If model-human kappa is poor, the model labels do
not stand** and the branch falls back to full human labeling.

- [ ] **Step 3 (original, retained for reference): Label the pool**

Open `research/statement_audit/pool_<stamp>.jsonl`. For each row, fill `labels` using the codebook at the top of this plan. Judge each statement standalone — do not open the source transcript.

```jsonl
{"id":"a1b2c3d4e5f6","event_type":"CONSTRAINT_HARD","text":"Non-zero temperature means the caller explicitly wants non-deterministic output.","labels":["WRONG_TYPE"],"notes":""}
{"id":"f6e5d4c3b2a1","event_type":"DECISION","text":"The cap constant lives next to the slot registry so it's easy to find and tune.","labels":["CLEAN"],"notes":""}
```

Do **not** open `pool_meta_<stamp>.jsonl` while labeling — it carries the heuristic verdicts, and seeing them contaminates the miss-rate measurement.

- [ ] **Step 4: Run the report**

Run: `uv run python scripts/audit_report.py research/statement_audit/pool_<stamp>.jsonl`
Expected: per-code and per-type tables, the heuristic miss rate, and a results JSON.

- [ ] **Step 5: Do NOT commit the audit outputs — transcribe the numbers instead**

**Amended during execution (human ruling).** This step originally said to commit
`results_*.json` and `heuristic_*.json`. That contradicted an established repo
convention discovered mid-execution: `.gitignore:4` ignores **all** of
`research/`, and zero files under it are tracked anywhere — including
`research/model_eval/salience_eval.jsonl`, the frozen 416-row eval. Research data
stays out of git here.

So: audit outputs remain on disk only. The authoritative numbers are transcribed
into the **tracked spec** in Task 6, which is what a later reader cites. Record
the source filenames (`heuristic_<stamp>.json`, `results_<stamp>.json`) in the
spec so the on-disk artifact behind each number is identifiable.

Known cost, stated rather than hidden: the raw results are not reproducible from
a fresh clone. Re-running `scripts/audit_statement_quality.py` reproduces the
*sweep* deterministically (fixed seed), but not the human labels — those exist
only in the local pool file. **Back up the labeled pool outside the repo before
anything runs `git clean -fdx`.**

The `research/statement_audit/pool*.jsonl` rule added in Step 1 is redundant
under the blanket `research/` rule. Keep it anyway: it is defence-in-depth for a
file containing raw project memory, and it stays load-bearing if `research/` is
ever narrowed to track eval sets.

---

### Task 6: Record the A0 decision

**Files:**
- Modify: `docs/superpowers/specs/2026-07-25-memory-statement-generation-design.md` (the "Phase A0" section)

- [ ] **Step 1: Write the outcome into the spec**

Add an "A0 outcome" subsection under Phase A0 recording: n labeled, defect rate with CI, per-code counts, heuristic miss rate, and the decision against the pre-registered rule (≥25% → run full Phase A; <15% → abandon; between → extend to n=150).

Record the number **as measured**, including if it kills the branch. The spec already states a low rate is an acceptable outcome; a result that ends the work is still a result, and this branch exists to measure where CogniKernel underperforms — not to confirm that it does.

- [ ] **Step 2: Note the single-labeler limitation**

A0 has one labeler, so no kappa is available and the rate carries unmeasured labeler bias. State this explicitly, and note that full Phase A requires a second labeler on an overlapping subset — **who that is, is an open question for the user**, since the codebook's author labeling alone would measure the codebook rather than the corpus.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/specs/2026-07-25-memory-statement-generation-design.md
git commit -m "docs(spec): record A0 pilot outcome and decision"
```

---

## Phase A (full audit) — scope note

Not tasked here. It reuses every script above with `--n 400 --stratum all` plus a second labeler file via `--labeler-b`, and is authorized only if Task 6's decision says PROCEED. The additional work is: recruiting the second labeler, an overlapping subset design, and reporting kappa alongside the rate. Spec it as its own plan once A0 returns a number.

## Out of scope (per spec)

- Any generator, teacher API call, or `.env` change — Phase B, gated on the ≥20% threshold.
- Fixing skeleton path truncation (`xtraction/`, `rc/memlora/`) or `.uv-cache` entries appearing in the skeleton. These are real bugs found during the audit but they are string-slicing and scope defects, not generation gaps. File separately.
- Any change to `salience_v2`, `supersession_xenc`, the taxonomy, or `src/cognikernel/`.

---

### Task 5A: LLM pre-labeling (added during execution)

**Files:**
- Create: `scripts/audit_label_llm.py`
- Modify: `tests/eval/test_statement_audit.py` (append)

**Interfaces produced:**
- `build_prompt(row: dict) -> list[dict]` — chat messages carrying the codebook + one statement
- `parse_labels(raw: str) -> tuple[list[str], str]` — `(labels, notes)`; raises `ValueError` on unparseable or off-codebook output
- `main()` — CLI writing one labeler file per model

**Transport facts, verified live against the API — do not re-derive:**
- Together is OpenAI-compatible: use the `openai` SDK with
  `base_url="https://api.together.xyz/v1"`, key from `TOGETHER_API_KEY`.
  Do **not** use `urllib` — Cloudflare fingerprint-blocks it with a bare
  `403 error code: 1010` that looks like an auth failure but is not.
  (`openai` is not a project dependency; run with `uv run --with openai`.)
- **`max_tokens` must be generous — 8192, NOT 1024.** *(Corrected after the
  first run. The original 1024 figure below was calibrated on a trivial smoke
  prompt and proved badly too low for the real codebook prompt: 167 of 341
  cached responses returned `finish_reason=length`, 162 of them with empty
  content. Every model's largest successful completion sat at the ceiling —
  DeepSeek 1021, Kimi 1016, GLM-5.2 974 — against medians of 600–800, the
  signature of a binding cap severing the tail. Worse, the loss was
  **non-random**: items needing more reasoning truncated, so surviving labels
  skewed toward CLEAN and would have biased the defect rate downward, pushing
  the pre-registered gate toward a false "abandon". Kimi lost 82% of its rows.
  The cache key must also include `max_tokens`, or raising the cap silently
  replays the stale truncated responses.)*
- Original (superseded) rationale for a generous cap: DeepSeek-V4-Pro and Kimi-K2.6 are
  reasoning models that spend hidden thinking tokens before any visible output.
  Measured: at `max_tokens=50` DeepSeek returned `finish_reason="length"` with
  *truncated* JSON, and cognitrace (`src/cognitrace/harness/reader.py:50-58`)
  documents the worse case — empty content that reads as a confident answer
  rather than an error. A trivial `{"labels":["CLEAN"]}` reply cost 39 output
  tokens on DeepSeek, 74 on Kimi, 85 on gpt-oss.
- `temperature=0`. Do **not** pass `seed` — Together ignores it (see
  cognitrace `reader.py:247`).
- **The `/v1/models` catalog lists non-serverless models, and version suffixes
  matter.** `zai-org/GLM-5`, `zai-org/GLM-5.1`, `zai-org/GLM-4.7`, and
  `Qwen/Qwen3-235B-A22B-Instruct-2507-FP8` all return `400 model_not_available`
  and need a dedicated endpoint — but **`zai-org/GLM-5.2` is serverless and
  works** (confirmed against a working call in the sibling CogniTrace project).
  An initial sweep missed it purely because the listing was truncated before it
  sorted in; do not conclude a family is unavailable from one id.
  Verified serverless and used as labelers: `deepseek-ai/DeepSeek-V4-Pro`,
  `moonshotai/Kimi-K2.6`, `openai/gpt-oss-120b`, `zai-org/GLM-5.2`.
- Measured reasoning overhead on a trivial `{"labels":["CLEAN"]}` reply:
  DeepSeek 39 output tokens, Kimi 74, gpt-oss 85, **GLM-5.2 197**. GLM-5.2 alone
  would truncate under any cap below ~200, which is why the 1024 floor is not
  merely cautious.

**Requirements:**
1. Read the blinded pool (`id`, `event_type`, `text`) — never the meta sidecar.
2. For each of the three models, emit
   `research/statement_audit/labels_<model-slug>_<stamp>.jsonl` in **exactly the
   pool schema** (`id`, `event_type`, `text`, `labels`, `notes`), so
   `scripts/audit_report.py --labeler-b` consumes it unchanged.
3. The codebook goes in the prompt **verbatim** from this plan's codebook table,
   including the "judge the statement standalone" instruction. Record the
   prompt's SHA-256 in a manifest (as cognitrace's `prompt_fingerprints()` does)
   so a prompt edit cannot silently change what the number means.
4. On-disk response cache keyed by `sha256(model + prompt)` so a re-run or an
   added model never re-spends on work already done.
5. Retry with full-jitter backoff honouring any `retry-after` header.
6. `parse_labels` enforces the codebook: `CLEAN` exclusive, `OTHER` requires a
   note, unknown codes rejected. One corrective retry on a violation, then
   record the item as a parse failure. **Report the parse-failure rate** — a
   model that cannot follow the codebook is not a usable labeler.
7. Write a manifest recording model ids, prompt SHA, pool filename, counts, and
   parse-failure rate per model.

**Constraints:** no new *project* dependency (`openai` is invoked ad-hoc via
`uv run --with`); never write to `~/.cognikernel/`; do not modify
`scripts/audit_statement_quality.py`, `scripts/audit_report.py`, or
`src/cognikernel/`; outputs are gitignored under `research/`.
