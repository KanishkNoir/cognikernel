"""Mine honor-probe CANDIDATES from real stores — never writes the bank directly.

Sources (per benchmark_redesign_2026-07.md §3.5: probes are minted from the
store's own decisions, so re-baselining is by construction):

  chains     superseded_by pairs where both sides are value-shaped decisions/
             constraints — the NEW text is the fact, the OLD text is the
             natural plausible-wrong corruption. Echo/dedup/status-churn links
             are filtered out (most links are these; the 2026-07 survey found
             "23 tests pass"->"43 tests pass" style pairs dominate).
  graveyard  live APPROACH_ABANDONED events with rejection morphology naming a
             concrete approach — corruption is the drafted "approved" inversion.

A cross-family LLM (OpenAI, same convention as the campaign scripts) drafts the
task_prompt / corrupt_line / grade regexes per candidate and self-rates quality.
Output: research/benchmarking/honor_probes_mined.jsonl with status="draft" —
a human reviews and promotes good rows into honor_probes.jsonl. The rejects are
themselves signal: a store whose mine yields mostly junk has an extraction-
precision problem (Task #16 evidence).

Usage:
  uv run python scripts/mine_honor_probes.py                  # all stores
  uv run python scripts/mine_honor_probes.py --stores relay_ultra --no-llm
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = ROOT / "research" / "benchmarking" / "honor_probes_mined.jsonl"

STORES: dict[str, tuple[str, str]] = {
    # name -> (project_path, project_id)
    "relay_ultra":     (r"C:\Users\Admin\OneDrive\Desktop\Relay_Ultra",       "6ffdc35c727bf5d2"),
    "conductor_ultra": (r"C:\Users\Admin\OneDrive\Desktop\Conductor_Ultra",   "082cccb2b4d1e6cf"),
    "toolbelt_ultra":  (r"C:\Users\Admin\OneDrive\Desktop\Toolbelt_Ultra",    "af4c70958ec60b9e"),
    "taskflow_ultra":  (r"C:\Users\Admin\OneDrive\Desktop\Taskflow_Ultra_CK", "b55954a8cd813d1a"),
    "relay_orig":      (r"C:\Users\Admin\OneDrive\Desktop\OMEGA_RELAY",       "484b812967d795c6"),
    "taskflow_orig":   (r"C:\Users\Admin\OneDrive\Desktop\Taskflow_ALPHA",    "961b42e80e47feef"),
    "toolbelt_orig":   (r"C:\Users\Admin\OneDrive\Desktop\TOOLBELT_ALPHA",    "d5dce4e1032e6457"),
    "conductor_orig":  (r"C:\Users\Admin\OneDrive\Desktop\CONDUCTOR_BETA",    "720e6b7c1e7d2266"),
}

VALUE_TYPES = ("DECISION", "CONSTRAINT_HARD", "CONSTRAINT_SOFT")

# narration/status-churn shapes that dominate raw supersession links
_NARRATION = re.compile(
    r"^(now |let'?s |let me |running |all \d+ tests|clean —|verified:|good[,.]|confirmed)",
    re.I)
_REJECTION = re.compile(
    r"reject|never|abandon|do not|don'?t|ruled out|not (going|use|adopt)|instead of|"
    r"rather than|no longer|explicitly (rejected|abandoned|ruled)", re.I)
_VALUE_SHAPE = re.compile(r"\d|[A-Z_]{3,}|->|→|=>|=")


def _txt(payload: str) -> str:
    try:
        d = json.loads(payload)
        return (d.get("description") or d.get("text") or "").strip()
    except Exception:
        return ""


def _jaccard(a: str, b: str) -> float:
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    return len(ta & tb) / max(1, len(ta | tb))


def mine_store(name: str, pid: str, cap: int) -> list[dict]:
    db = Path.home() / ".memlora" / "projects" / f"{pid}.db"
    con = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cands: list[dict] = []

    # ── value-evolution chains ───────────────────────────────────────────────
    rows = con.execute("""
        SELECT o.payload op, o.event_type oet, n.payload np, n.event_type net
        FROM events o JOIN events n ON o.superseded_by = n.id""").fetchall()
    chains = []
    for r in rows:
        old, new = _txt(r["op"]), _txt(r["np"])
        if r["oet"] not in VALUE_TYPES or r["net"] not in VALUE_TYPES:
            continue
        if len(old) < 30 or len(new) < 30:
            continue
        if _NARRATION.search(old) or _NARRATION.search(new):
            continue
        if not (_VALUE_SHAPE.search(old) and _VALUE_SHAPE.search(new)):
            continue
        j = _jaccard(old, new)
        if j >= 0.6:  # echo/restatement, not evolution
            continue
        chains.append((j, old, new))
    chains.sort(key=lambda t: t[0])  # least-overlapping = most evolution-shaped
    for j, old, new in chains[:cap]:
        cands.append({"store": name, "provenance": "mined-chain", "jaccard": round(j, 2),
                      "fact_text": new, "stale_text": old})

    # ── graveyard rejections ─────────────────────────────────────────────────
    rows = con.execute("""
        SELECT payload FROM events
        WHERE event_type='APPROACH_ABANDONED_DO_NOT_RETRY'
          AND superseded_by IS NULL AND (archived IS NULL OR archived=0)""").fetchall()
    grave = []
    for r in rows:
        t = _txt(r["payload"])
        if len(t) < 40 or t.endswith("?"):
            continue
        if not _REJECTION.search(t):
            continue
        if _NARRATION.search(t):
            continue
        grave.append(t)
    for t in grave[:cap]:
        cands.append({"store": name, "provenance": "mined-graveyard",
                      "fact_text": t, "stale_text": ""})
    con.close()
    return cands


# ── LLM drafting (cross-family, campaign-script conventions) ─────────────────

def _openai_key() -> str:
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("OPENAI_API_KEY"):
                _, _, v = line.partition("=")
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    import os
    return os.environ.get("OPENAI_API_KEY", "")


_DRAFT_SYSTEM = """You draft evaluation probes for a coding-agent memory benchmark.
Given a project decision (FACT) and optionally its superseded old value (STALE), produce a
JSON probe that tests whether an agent's produced ARTIFACT honors the fact:
- task_prompt: a realistic coding task that TEMPTS violating the fact without mentioning it,
  ending with "Output only code." (or a one-sentence-answer form for pure judgment calls).
- inject_line: the fact as one "- " markdown line in the project's own words.
- corrupt_line: one "- " line asserting the stale value (use STALE if given; for a rejected
  approach, assert it was approved/adopted instead).
- grade: {"must": [regex...], "must_not": [regex...]} on the artifact. Regexes must accept ANY
  reasonable implementation form of the fact (use alternations) and must_not must name ONLY
  the specific stale/wrong value, never "anything different". Keep regexes simple and bounded
  (no unbounded .* spans across values that could self-match, e.g. beware '= 3 (1 original +
  2 retries)' matching a must_not for 2).
- corrupt_follow: [regex...] that fire when the artifact implements the stale value.
- quality: "high"|"medium"|"low" — high only if the FACT is a real, project-specific,
  action-testable decision (not narration, not a status update, not generic best practice).
Reply with ONLY the JSON object."""


def draft_with_llm(cand: dict, model: str) -> dict:
    import urllib.request
    user = f"FACT (authoritative): {cand['fact_text']}\n\nSTALE (superseded old value, may be empty): {cand['stale_text']}"
    body = json.dumps({
        "model": model, "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": _DRAFT_SYSTEM},
                     {"role": "user", "content": user}],
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {_openai_key()}",
                 "Content-Type": "application/json"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                data = json.loads(r.read().decode("utf-8"))
            return json.loads(data["choices"][0]["message"]["content"])
        except Exception:
            time.sleep(2 * (attempt + 1))
    return {}


def main() -> None:
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", nargs="*", default=list(STORES))
    ap.add_argument("--cap", type=int, default=6, help="max candidates per source per store")
    ap.add_argument("--no-llm", action="store_true", help="mine + filter only, no drafting")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    args = ap.parse_args()

    all_rows: list[dict] = []
    for name in args.stores:
        path, pid = STORES[name]
        cands = mine_store(name, pid, args.cap)
        print(f"{name}: {len(cands)} candidates "
              f"({sum(1 for c in cands if c['provenance']=='mined-chain')} chain, "
              f"{sum(1 for c in cands if c['provenance']=='mined-graveyard')} graveyard)")
        for i, cand in enumerate(cands):
            row = {
                "id": f"{name}-{cand['provenance'].split('-')[1]}-{i}",
                "status": "draft",
                "project_path": path,
                **cand,
            }
            if not args.no_llm:
                draft = draft_with_llm(cand, args.model)
                if draft:
                    row.update({
                        "fact_kind": "chain-latest" if cand["provenance"] == "mined-chain" else "graveyard",
                        "task_prompt": draft.get("task_prompt", ""),
                        "inject_line": draft.get("inject_line", ""),
                        "corrupt_line": draft.get("corrupt_line", ""),
                        "grade": draft.get("grade", {"must": [], "must_not": []}),
                        "corrupt_follow": draft.get("corrupt_follow", []),
                        "llm_quality": draft.get("quality", "low"),
                        # heuristic defaults for the harness fields
                        "block_patterns": [re.escape(cand["fact_text"][:60])],
                        "inject_section": "### Hard constraints — never violate"
                            if cand["provenance"] == "mined-chain"
                            else "### Do not retry — confirmed failures",
                    })
            all_rows.append(row)

    OUT_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + "\n",
        encoding="utf-8")
    hi = sum(1 for r in all_rows if r.get("llm_quality") == "high")
    print(f"\n{len(all_rows)} candidates -> {OUT_PATH}  (llm_quality=high: {hi})")
    print("Review drafts, fix regexes/block_patterns, then move promoted rows into honor_probes.jsonl.")


if __name__ == "__main__":
    main()
