"""Capture a reproducible snapshot of what CogniKernel actually injects.

Research tooling — NOT part of the shipped package.

This is the before/after instrument for the injection-correctness branch. Run it
once before the fixes land and once after, with the SAME script, so the
measurement does not change underneath the comparison:

    python scripts/injection_snapshot.py --label before
    #  ... land the fixes ...
    python scripts/injection_snapshot.py --label after
    python scripts/injection_snapshot.py --diff before after

What it records, per project:
  - the full rendered injection block (verbatim, so claims are auditable)
  - total tokens and per-section token counts — where the budget actually goes
  - symbol-graph composition: first-party vs vendored/cache nodes
  - event counts by type

Read-only with respect to memory: it renders and measures, never writes events.
(`render_state` does run migrations on open, which is the normal read path.)
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path

# Two outputs, deliberately separated by sensitivity:
#   FULL    — includes the verbatim rendered block, i.e. raw project memory.
#             Lands under docs/research/, which .gitignore excludes. Local only.
#   METRICS — counts and token totals with no memory content. Safe to commit,
#             and it is what the paper's impact table cites.
SNAPSHOT_DIR = Path("docs/research/snapshots")
METRICS_DIR = Path("docs/metrics")

# Paths that are not the user's own source. Kept in sync with the D3 skip list
# in symbols/extractor.py, but defined here independently so the measurement
# does not silently track a change in the thing being measured.
VENDORED_RE = re.compile(
    r"(^|/)(\.uv-cache|\.pytest_tmp|\.pytest_cache|\.claude|site-packages|"
    r"\.venv|venv|node_modules|\.tox|\.mypy_cache|\.ruff_cache|dist|build|"
    r"__pycache__|\.eggs|vendor|target)(/|$)"
)

SECTION_RE = re.compile(r"^###\s+(.*)$")


def _count_tokens(text: str) -> int:
    from cognikernel.compression.token_count import count_tokens

    return count_tokens(text)


def _section_tokens(block: str) -> dict[str, int]:
    """Token count per '### Section' heading, in render order."""
    sections: dict[str, list[str]] = {}
    current = "«header»"
    sections[current] = []
    for line in block.split("\n"):
        m = SECTION_RE.match(line)
        if m:
            current = m.group(1).strip()
            sections.setdefault(current, [])
            continue
        sections[current].append(line)
    return {name: _count_tokens("\n".join(body)) for name, body in sections.items()}


def _symbol_composition(db_path: Path) -> dict:
    """First-party vs vendored node counts in the symbol graph."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute("SELECT path FROM symbol_nodes").fetchall()
        conn.close()
    except Exception:
        return {"total": 0, "vendored": 0, "first_party": 0, "vendored_pct": 0.0,
                "examples": []}
    total = len(rows)
    vendored = [p for (p,) in rows if p and VENDORED_RE.search(p)]
    return {
        "total": total,
        "vendored": len(vendored),
        "first_party": total - len(vendored),
        "vendored_pct": round(100.0 * len(vendored) / total, 2) if total else 0.0,
        "examples": sorted(vendored)[:5],
    }


def _event_counts(db_path: Path) -> dict[str, int]:
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        rows = conn.execute(
            "SELECT event_type, COUNT(*) FROM events WHERE archived=0 GROUP BY event_type"
        ).fetchall()
        conn.close()
    except Exception:
        return {}
    return {t: c for t, c in sorted(rows)}


def snapshot_project(project_path: Path) -> dict:
    """Render and measure one project. Returns a JSON-serializable record."""
    from cognikernel.config import Config
    from cognikernel.integration.session import render_state
    from cognikernel.utils.paths import canonicalize_path  # noqa: F401  (import check)
    from cognikernel.integration.session import resolve_project_id
    from cognikernel.utils.paths import canonicalize_path  # noqa: F811

    config = Config.load(project_path=project_path)
    project_id = resolve_project_id(project_path, config)
    db_path = config.projects_dir / f"{project_id}.db"

    block = render_state(str(project_path))
    return {
        "project_path": str(project_path),
        "project_id": project_id,
        "block": block,
        "total_tokens": _count_tokens(block),
        "section_tokens": _section_tokens(block),
        "symbol_composition": _symbol_composition(db_path),
        "event_counts": _event_counts(db_path),
    }


def cmd_capture(label: str, project_paths: list[Path]) -> int:
    records = []
    for p in project_paths:
        try:
            records.append(snapshot_project(p))
        except Exception as exc:
            print(f"  !! {p}: {exc}")
    captured_at = int(time.time())
    payload = {"label": label, "captured_at": captured_at, "projects": records}

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    out = SNAPSHOT_DIR / f"{label}.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    # Metrics-only twin: every field except the verbatim block, so the impact
    # table can be published without shipping anyone's project memory.
    metrics = {
        "label": label,
        "captured_at": captured_at,
        "projects": [
            {k: v for k, v in r.items() if k != "block"} | {
                "block_chars": len(r["block"]),
            }
            for r in records
        ],
    }
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_out = METRICS_DIR / f"injection_{label}.json"
    metrics_out.write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    for r in records:
        sc = r["symbol_composition"]
        print(f"\n{r['project_path']}  (project {r['project_id']})")
        print(f"  block            : {r['total_tokens']} tok")
        print(f"  symbol nodes     : {sc['total']} total, {sc['vendored']} vendored "
              f"({sc['vendored_pct']}%)")
        if sc["examples"]:
            print(f"  vendored e.g.    : {sc['examples'][0]}")
        print("  section tokens   :")
        for name, tok in sorted(r["section_tokens"].items(), key=lambda kv: -kv[1]):
            if tok:
                print(f"      {tok:>5}  {name}")
    print(f"\nwrote {out} (local only — contains verbatim memory)")
    print(f"wrote {metrics_out} (committable)")
    return 0


def cmd_diff(before_label: str, after_label: str) -> int:
    """Compare two snapshots. Reads the metrics twins, so it works from a
    fresh clone that never had the local full snapshots."""
    a = json.loads(
        (METRICS_DIR / f"injection_{before_label}.json").read_text(encoding="utf-8")
    )
    b = json.loads(
        (METRICS_DIR / f"injection_{after_label}.json").read_text(encoding="utf-8")
    )
    by_id = {r["project_id"]: r for r in b["projects"]}

    print(f"{'project':18} {'metric':22} {'before':>10} {'after':>10} {'delta':>10}")
    print("-" * 74)
    for rec_a in a["projects"]:
        rec_b = by_id.get(rec_a["project_id"])
        if rec_b is None:
            continue
        pid = rec_a["project_id"][:16]
        pairs = [
            ("total tokens", rec_a["total_tokens"], rec_b["total_tokens"]),
            ("symbol nodes", rec_a["symbol_composition"]["total"],
             rec_b["symbol_composition"]["total"]),
            ("vendored nodes", rec_a["symbol_composition"]["vendored"],
             rec_b["symbol_composition"]["vendored"]),
            ("vendored %", rec_a["symbol_composition"]["vendored_pct"],
             rec_b["symbol_composition"]["vendored_pct"]),
        ]
        for name, x, y in pairs:
            print(f"{pid:18} {name:22} {x:>10} {y:>10} {y - x:>+10}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--label", help="capture a snapshot under this label")
    ap.add_argument("--project", action="append", default=[],
                    help="project path to snapshot (repeatable; default: cwd)")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"),
                    help="print a before/after comparison table")
    args = ap.parse_args()

    if args.diff:
        return cmd_diff(*args.diff)
    if not args.label:
        ap.error("one of --label or --diff is required")
    paths = [Path(p).resolve() for p in args.project] or [Path.cwd()]
    return cmd_capture(args.label, paths)


if __name__ == "__main__":
    raise SystemExit(main())
