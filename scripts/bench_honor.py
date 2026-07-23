"""Counterfactual memory-to-action benchmark — the "honor" stage.

The recall-QA benchmarks saturate (Relay scored 92 twice across a real model
change) and cannot attribute a correct answer to memory vs. repo re-derivation
(research/benchmarking/benchmark_redesign_2026-07.md §1-2). This harness
measures what memory actually *causes*: for each target fact, the real rendered
injection block is surgically toggled and the SAME task prompt is executed
under each condition by a headless `claude -p` agent in an EMPTY directory
(no repo — the only variable is the block):

  PRESENT    block contains the fact (as rendered, or inserted verbatim from
             the store's own wording when budget eviction dropped it)
  ABSENT     the fact's lines are deleted
  CORRUPTED  the fact is replaced with its superseded/stale value (chains give
             the natural plausible-wrong for free)

Per-probe metrics:
  block_present   was the fact in the real rendered block at all (render-
                  presence — budget eviction is itself a finding)
  honor(P/A)      did the produced artifact satisfy the objective graders
  lift            honor(PRESENT) - honor(ABSENT) — the only honest evidence
                  memory caused the behavior
  corrupt_follow  did the artifact follow the stale value — the T1 failure
                  mode (memory noise corrupting action) as a metric

Probe bank: research/benchmarking/honor_probes.jsonl (see fields there).
Results:    research/benchmarking/honor_results.json (incremental — partial
            runs are usable; --resume skips already-recorded cells).

Usage:
  uv run python scripts/bench_honor.py --dry-run            # show surgery only
  uv run python scripts/bench_honor.py --probes relay-retry3 --model haiku
  uv run python scripts/bench_honor.py --model sonnet --repeats 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PROBES_PATH = ROOT / "research" / "benchmarking" / "honor_probes.jsonl"
RESULTS_PATH = ROOT / "research" / "benchmarking" / "honor_results.json"

CONDITIONS = ("present", "absent", "corrupted")

# The agent preamble mirrors deployment: the block arrives as authoritative
# session context, and the task follows. No tools exist in the empty cwd.
PROMPT_TEMPLATE = """{block}

---
You are the coding agent for this project. The session context above is
authoritative — it supersedes your own defaults and general best practices.

Task: {task}
{repo_excerpt}
Do not use any tools. Reply with only the requested output."""


def load_probes() -> list[dict]:
    probes = []
    for line in PROBES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            probes.append(json.loads(line))
    return probes


# Sections that carry MEMORY (surgery allowed). The skeleton and hot-files
# sections are repo structure — mutating them across conditions would confound
# the counterfactual (the repo must stay frozen; only memory toggles).
_MEMORY_SECTIONS = (
    "### Hard constraints — never violate",
    "### Active thread",
    "### Do not retry — confirmed failures",
    "### Component state",
    "### Key decisions",
)

# The structural / retrieval channel. Toggled OFF (whole sections removed) for
# the skeleton-ablation: comparing honor with vs without these decomposes total
# memory lift into event-lift (rationale/chains, from _MEMORY_SECTIONS) and
# structural-lift (contracts/signatures, from the skeleton). A "zero event-lift"
# fact that DROPS when the skeleton is removed was carried by CK's structural
# channel, not by the model's prior.
_STRUCTURAL_HEADERS = ("### Codebase skeleton", "### Most active files")


def strip_skeleton(block: str) -> str:
    """Remove the structural-memory sections (skeleton + most-active-files)
    entirely — from each matching header to the next '### '/'## ' or EOF."""
    out: list[str] = []
    dropping = False
    for line in block.splitlines():
        if line.startswith("### ") or line.startswith("## "):
            dropping = any(line.strip().startswith(h) for h in _STRUCTURAL_HEADERS)
        if not dropping:
            out.append(line)
    return "\n".join(out)


def delete_fact_lines(block: str, patterns: list[str],
                      all_memory: bool = False) -> tuple[str, int]:
    """Remove every MEMORY-section line matching any pattern, plus its indented
    continuation lines (the renderer's '  — previously: …' annotations).
    Lines in structural sections (skeleton, hot files) are never touched.
    `all_memory=True` treats the whole block as memory — used for flat-arm
    artifacts (CLAUDE.md / auto-memory notes), which have no CK sections and
    contain no repo structure that must stay frozen."""
    pats = [re.compile(p, re.I) for p in patterns]
    out: list[str] = []
    deleted = 0
    skip_continuation = False
    in_memory_section = all_memory
    for line in block.splitlines():
        if not all_memory and (line.startswith("### ") or line.startswith("## ")):
            in_memory_section = line.strip() in _MEMORY_SECTIONS
            skip_continuation = False
            out.append(line)
            continue
        if skip_continuation and (line.startswith("  ") and not line.startswith("  -")):
            deleted += 1
            continue
        skip_continuation = False
        if in_memory_section and any(p.search(line) for p in pats):
            deleted += 1
            skip_continuation = True
            continue
        out.append(line)
    return "\n".join(out), deleted


def insert_line(block: str, section: str, line_to_add: str) -> str:
    """Insert *line_to_add* right after the section header (first line of the
    section = highest primacy). Falls back to the hard-constraints section,
    then to prepending, so a probe never silently loses its PRESENT arm."""
    lines = block.splitlines()
    for target in (section, "### Hard constraints — never violate"):
        for i, ln in enumerate(lines):
            if ln.strip() == target:
                return "\n".join(lines[: i + 1] + [line_to_add] + lines[i + 1 :])
    return line_to_add + "\n" + block


def build_condition_block(block: str, probe: dict, condition: str) -> tuple[str, dict]:
    all_memory = bool(probe.get("block_files"))
    stripped, n_deleted = delete_fact_lines(block, probe["block_patterns"], all_memory)
    meta = {"block_present": n_deleted > 0, "lines_deleted": n_deleted}
    if condition == "absent":
        return stripped, meta
    if condition == "present":
        # Keep the real rendering when the fact survived the budget; insert the
        # store-worded line only when eviction dropped it. Exception: when the
        # artifact's matching lines carry a STALE value (flat-arm notes that
        # were never updated), as-is ≠ fact-present — `present_requires_insert`
        # forces delete-stale + insert-correct so PRESENT means what it says.
        if probe.get("present_requires_insert"):
            return insert_line(stripped, probe["inject_section"], probe["inject_line"]), meta
        if n_deleted > 0:
            return block, meta
        return insert_line(block, probe["inject_section"], probe["inject_line"]), meta
    if condition == "corrupted":
        return insert_line(stripped, probe["inject_section"], probe["corrupt_line"]), meta
    raise ValueError(condition)


# Executor failure banners — NOT agent answers. A cell must never be recorded
# with one of these as its output (the 2026-07-12 run recorded ~40 cells of
# "You've hit your session limit" as honored=False, poisoning the aggregates).
_INVALID_OUTPUT = re.compile(
    r"hit your (session|usage) limit|Error: Reached max turns|"
    r"credit balance is too low|overloaded_error", re.I)


def run_agent(prompt: str, model: str, cwd: Path, timeout: int = 240) -> str | None:
    """Returns the agent's answer, or None when the executor failed (limit hit,
    max-turns error, timeout) — callers must skip recording on None."""
    # Prompt goes via STDIN: on Windows a multi-line argv through the cmd shim
    # gets mangled (the task tail after the block was silently lost).
    import shutil as _shutil
    exe = _shutil.which("claude") or "claude"
    env = {k: v for k, v in os.environ.items()
           if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
    for attempt in range(2):
        try:
            proc = subprocess.run(
                [exe, "-p", "--model", model, "--max-turns", "1"],
                input=prompt,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(cwd), env=env, timeout=timeout,
            )
            out = (proc.stdout or "").strip()
            if out and _INVALID_OUTPUT.search(out):
                return None
            if out:
                return out
        except subprocess.TimeoutExpired:
            pass
        time.sleep(2)
    return None


def grade(output: str, must: list[str], must_not: list[str]) -> bool:
    if not output:
        return False
    ok = all(re.search(p, output, re.I | re.S) for p in must)
    bad = any(re.search(p, output, re.I | re.S) for p in must_not)
    return ok and not bad


def followed_corruption(output: str, follow: list[str]) -> bool:
    return bool(output) and any(re.search(p, output, re.I | re.S) for p in follow)


# ── Tier-2 grading: cross-family LLM judge for the audit queue ───────────────
# Regex (tier 1) stays the primary grader — deterministic, drift-free. The
# judge runs ONLY on disputed cells (PRESENT-unhonored, CORRUPTED-neither) and
# answers a narrow classification with the decision text as ground truth —
# never a quality score. Cross-family (OpenAI) to avoid same-family
# self-preference when Claude output is being judged. Verdicts are cached in
# the results JSON; a probe whose cells the judge overturns >30% has broken
# regexes — fix the probe, don't lean on the judge.

def _openai_key() -> str:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENAI_API_KEY"):
                _, _, v = line.partition("=")
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    return os.environ.get("OPENAI_API_KEY", "")


_JUDGE_SYSTEM = """You are a strict evaluation judge for a coding-agent memory benchmark.
A project made a specific recorded DECISION. An agent produced an OUTPUT for a task.
Classify the OUTPUT against the DECISION:
- COMPLIES: the output satisfies the decision's substance in ANY reasonable surface form
  (different function names, decorators vs calls, wording changes are all fine). It does
  NOT comply merely by mentioning the decision.
- FOLLOWS_STALE: the output implements, asserts, or builds on the stale/superseded value
  or approach — even partially, and regardless of how it is phrased or named.
- NEITHER: the output does something else entirely (off-task, refusal, a third approach
  that matches neither the decision nor the stale value).
Reply with ONLY a JSON object:
{"verdict": "COMPLIES" | "FOLLOWS_STALE" | "NEITHER", "rationale": "<one sentence>"}"""

_JUDGE_USER = """DECISION (authoritative, in the project's own words):
{decision}

STALE/SUPERSEDED value (the wrong thing an agent might fall back to):
{stale}

TASK given to the agent:
{task}

AGENT OUTPUT:
{output}"""


def judge_cell(probe: dict, cell: dict, model: str) -> dict:
    import urllib.request
    body = json.dumps({
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": _JUDGE_USER.format(
                decision=probe["inject_line"], stale=probe["corrupt_line"],
                task=probe["task_prompt"], output=cell["output"] or "[EMPTY]")},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {_openai_key()}",
                 "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            verdict = json.loads(data["choices"][0]["message"]["content"])
            return {"verdict": verdict.get("verdict", "PARSE_ERROR"),
                    "rationale": verdict.get("rationale", "")}
        except Exception as exc:  # noqa: BLE001 — retry then surface
            err = str(exc)
            time.sleep(2 * (attempt + 1))
    return {"verdict": "JUDGE_ERROR", "rationale": err}


def run_judge_pass(results: dict, probes_by_id: dict, out_path: Path, model: str,
                   rejudge: bool = False) -> None:
    audit = [c for c in results["cells"].values()
             if (c["condition"] == "present" and not c["honored"])
             or (c["condition"] == "corrupted" and not c["honored"]
                 and not c["followed_corruption"])]
    todo = [c for c in audit if rejudge or "judge_verdict" not in c]
    print(f"judge pass: {len(audit)} audit cells, {len(todo)} to judge (model={model})")
    for c in todo:
        probe = probes_by_id.get(c["probe"])
        if probe is None:
            continue
        v = judge_cell(probe, c, model)
        c["judge_verdict"] = v["verdict"]
        c["judge_rationale"] = v["rationale"]
        c["judge_model"] = model
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  {c['probe']}::{c['condition']}::{c['rep']:<3} regex=unhonored "
              f"judge={v['verdict']:14} {v['rationale'][:90]}")
    # overturn stats per probe: judge says COMPLIES where regex said not honored
    print("\njudge-vs-regex (PRESENT cells): overturns flag broken graders (>30% -> fix probe)")
    for pid, probe in probes_by_id.items():
        cells = [c for c in results["cells"].values()
                 if c["probe"] == pid and c["condition"] == "present" and not c["honored"]]
        if not cells:
            continue
        overturned = sum(1 for c in cells if c.get("judge_verdict") == "COMPLIES")
        print(f"  {pid:26} disputed={len(cells)} judge-overturned={overturned}")


def print_scorecard(results: dict, probes_by_id: dict) -> None:
    """Aggregate scorecard: honor rate / lift / load-bearing share / corruption
    resistance, broken down by arm, by project, and by fact kind. Judge verdicts
    override tier-1 regex where present. honor(A) anchors every probe against
    what the model gets free, so these numbers cannot saturate silently."""
    from collections import defaultdict
    by: dict = defaultdict(lambda: defaultdict(list))
    for c in results["cells"].values():
        if c["condition"] not in CONDITIONS:
            continue
        hon = c["honored"] or (c.get("judge_verdict") == "COMPLIES")
        fol = c["followed_corruption"] or (c.get("judge_verdict") == "FOLLOWS_STALE")
        by[c["probe"]][c["condition"]].append((hon, fol))

    def dims(pid: str) -> dict:
        p = probes_by_id.get(pid, {})
        return {
            "arm": p.get("arm", "CK"),
            "project": Path(p.get("project_path", "?")).name,
            "kind": p.get("fact_kind", "?"),
        }

    for dim in ("arm", "project", "kind"):
        agg: dict = defaultdict(lambda: {"n": 0, "P": [], "A": [], "flip": [], "load": 0, "resist": 0})
        for pid, conds in by.items():
            if not conds.get("present") or not conds.get("absent"):
                continue
            P = sum(h for h, _ in conds["present"]) / len(conds["present"])
            A = sum(h for h, _ in conds["absent"]) / len(conds["absent"])
            F = (sum(f for _, f in conds["corrupted"]) / len(conds["corrupted"])) if conds.get("corrupted") else 0.0
            s = agg[dims(pid)[dim]]
            s["n"] += 1; s["P"].append(P); s["A"].append(A); s["flip"].append(F)
            if P - A >= 0.5:
                s["load"] += 1
            if F <= 0.25:
                s["resist"] += 1
        print(f"\nSCORECARD by {dim}")
        print(f"  {'':22}{'probes':7}{'honor(P)':9}{'honor(A)':9}{'lift':7}{'load-bearing':13}{'flip':6}{'resist':7}")
        for k in sorted(agg):
            s = agg[k]
            m = lambda x: sum(x) / len(x) if x else 0.0
            print(f"  {k:22}{s['n']:<7}{m(s['P']):<9.2f}{m(s['A']):<9.2f}"
                  f"{m(s['P']) - m(s['A']):<+7.2f}{s['load']}/{s['n']:<11}{m(s['flip']):<6.2f}{s['resist']}/{s['n']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="*", default=None, help="probe ids to run (default: all)")
    ap.add_argument("--conditions", nargs="*", default=list(CONDITIONS), choices=CONDITIONS)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--out", default=str(RESULTS_PATH))
    ap.add_argument("--dry-run", action="store_true", help="show block surgery, no agent calls")
    ap.add_argument("--resume", action="store_true", help="skip cells already in the results file")
    ap.add_argument("--no-skeleton", action="store_true",
                    help="skeleton-ablation: strip the structural/retrieval sections from every "
                         "block, isolating the event channel (compare vs a normal run to get structural-lift)")
    ap.add_argument("--judge-pass", action="store_true",
                    help="tier-2: judge the audit queue in an existing results file (no agent calls)")
    ap.add_argument("--judge-model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    ap.add_argument("--rejudge", action="store_true", help="re-judge cells that already have a verdict")
    ap.add_argument("--scorecard", action="store_true",
                    help="print the aggregate scorecard from an existing results file, then exit")
    ap.add_argument("--purge-invalid", action="store_true",
                    help="drop cells whose stored output is an executor failure banner, then exit")
    ap.add_argument("--regrade", action="store_true",
                    help="re-run tier-1 grading on stored outputs with current probe regexes, then exit")
    args = ap.parse_args()

    if args.scorecard:
        out_path = Path(args.out)
        results = json.loads(out_path.read_text(encoding="utf-8"))
        print_scorecard(results, {p["id"]: p for p in load_probes()})
        return

    if args.purge_invalid:
        out_path = Path(args.out)
        results = json.loads(out_path.read_text(encoding="utf-8"))
        before = len(results["cells"])
        results["cells"] = {k: c for k, c in results["cells"].items()
                            if not _INVALID_OUTPUT.search(c.get("output", ""))}
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"purged {before - len(results['cells'])} invalid cells ({len(results['cells'])} remain)")
        return

    if args.regrade:
        # Grader iteration without re-executing agents: recompute tier-1 verdicts
        # from stored outputs using the CURRENT probe regexes. Judge verdicts are
        # preserved (they grade the same immutable output).
        out_path = Path(args.out)
        results = json.loads(out_path.read_text(encoding="utf-8"))
        probes_by_id = {p["id"]: p for p in load_probes()}
        changed = 0
        for c in results["cells"].values():
            probe = probes_by_id.get(c["probe"])
            if probe is None:
                continue
            new_honored = grade(c["output"], probe["grade"]["must"], probe["grade"]["must_not"])
            new_followed = followed_corruption(c["output"], probe.get("corrupt_follow", []))
            if new_honored != c["honored"] or new_followed != c["followed_corruption"]:
                changed += 1
                print(f"  regraded {c['probe']}::{c['condition']}::{c['rep']}: "
                      f"honored {c['honored']}->{new_honored}, followed {c['followed_corruption']}->{new_followed}")
            c["honored"], c["followed_corruption"] = new_honored, new_followed
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"regrade complete: {changed} cells changed")
        return

    if args.judge_pass:
        out_path = Path(args.out)
        results = json.loads(out_path.read_text(encoding="utf-8"))
        probes_by_id = {p["id"]: p for p in load_probes()}
        run_judge_pass(results, probes_by_id, out_path, args.judge_model, args.rejudge)
        return

    from cognikernel.integration.session import render_state

    probes = load_probes()
    if args.probes:
        probes = [p for p in probes if p["id"] in set(args.probes)]
        missing = set(args.probes) - {p["id"] for p in probes}
        if missing:
            sys.exit(f"unknown probe ids: {sorted(missing)}")

    out_path = Path(args.out)
    results: dict = {"model": args.model, "repeats": args.repeats, "cells": {}}
    if args.resume and out_path.exists():
        results = json.loads(out_path.read_text(encoding="utf-8"))
        results.setdefault("cells", {})

    exec_dir = Path(tempfile.mkdtemp(prefix="ck_honor_exec_"))
    blocks: dict[str, str] = {}

    for probe in probes:
        # Block source: CK arm renders the real injection block; flat arms
        # (auto-memory notes / self-written CLAUDE.md) concatenate their files.
        if probe.get("block_files"):
            key = "|".join(probe["block_files"])
            if key not in blocks:
                parts = []
                for fp in probe["block_files"]:
                    p = Path(fp)
                    parts.append(f"## Project memory: {p.name}\n{p.read_text(encoding='utf-8')}")
                blocks[key] = "\n\n".join(parts)
            block = blocks[key]
        else:
            path = probe["project_path"]
            if path not in blocks:
                blocks[path] = render_state(path)
            block = blocks[path]

        if args.no_skeleton:
            block = strip_skeleton(block)

        for condition in args.conditions:
            cond_block, meta = build_condition_block(block, probe, condition)
            if args.dry_run:
                stripped, n = delete_fact_lines(block, probe["block_patterns"])
                print(f"\n=== {probe['id']} [{condition}] block_present={meta['block_present']} "
                      f"deleted={meta['lines_deleted']} block_chars={len(cond_block)}")
                if condition == "present" and not meta["block_present"]:
                    print(f"    would insert: {probe['inject_line'][:110]}")
                if condition == "corrupted":
                    print(f"    corrupt line: {probe['corrupt_line'][:110]}")
                continue

            for rep in range(args.repeats):
                cell_id = f"{probe['id']}::{condition}::{rep}"
                if args.resume and cell_id in results["cells"]:
                    continue
                prompt = PROMPT_TEMPLATE.format(
                    block=cond_block,
                    task=probe["task_prompt"],
                    repo_excerpt=(probe.get("repo_excerpt") or ""),
                )
                t0 = time.time()
                output = run_agent(prompt, args.model, exec_dir)
                if output is None:
                    print(f"  {cell_id:46} EXECUTOR FAILED (limit/max-turns/timeout) — not recorded")
                    continue
                cell = {
                    "probe": probe["id"],
                    "condition": condition,
                    "rep": rep,
                    "block_present": meta["block_present"],
                    "honored": grade(output, probe["grade"]["must"], probe["grade"]["must_not"]),
                    "followed_corruption": followed_corruption(output, probe.get("corrupt_follow", [])),
                    "seconds": round(time.time() - t0, 1),
                    "output": output[:2000],
                }
                results["cells"][cell_id] = cell
                out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
                print(f"  {cell_id:46} honored={cell['honored']} "
                      f"corrupt_follow={cell['followed_corruption']} ({cell['seconds']}s)")

    if args.dry_run:
        return

    # ── summary table ────────────────────────────────────────────────────────
    print(f"\n{'probe':26} {'in-block':8} {'honor(P)':9} {'honor(A)':9} {'lift':6} {'corrupt-flip':12}")
    for probe in probes:
        cells = [c for c in results["cells"].values() if c["probe"] == probe["id"]]
        if not cells:
            continue
        def rate(cond, key="honored"):
            sub = [c[key] for c in cells if c["condition"] == cond]
            return sum(sub) / len(sub) if sub else None
        p, a = rate("present"), rate("absent")
        flip = rate("corrupted", "followed_corruption")
        in_block = any(c["block_present"] for c in cells)
        fmt = lambda v: "  —  " if v is None else f"{v:.2f}"
        lift = "  —  " if (p is None or a is None) else f"{p - a:+.2f}"
        print(f"{probe['id']:26} {str(in_block):8} {fmt(p):9} {fmt(a):9} {lift:6} {fmt(flip):12}")

    # ── audit queue ──────────────────────────────────────────────────────────
    # A PRESENT cell graded un-honored is NOT reportable as a violation until a
    # human reads the output: it may be a legitimate alternative implementation
    # that satisfies the decision in a surface form the regex grader missed
    # (re-baselining discipline — same rule as the QA rubric's useful-redecision
    # category). Same for CORRUPTED cells that neither honored nor followed:
    # the model may have done a third reasonable thing.
    audit = [c for c in results["cells"].values()
             if (c["condition"] == "present" and not c["honored"])
             or (c["condition"] == "corrupted" and not c["honored"] and not c["followed_corruption"])]
    if audit:
        print(f"\nAUDIT before reporting ({len(audit)} cells — grader may have missed a legitimate variant):")
        for c in audit:
            print(f"  {c['probe']}::{c['condition']}::{c['rep']}  -> {c['output'][:120].replace(chr(10), ' | ')}")
    print_scorecard(results, {p["id"]: p for p in load_probes()})
    print(f"\nfull JSON: {out_path}")


if __name__ == "__main__":
    main()
