#!/usr/bin/env python3
"""H7 — automate the run sheet's manual `bypass_to_update` count.

Usage:
    python scripts/h7_bypass.py <session.jsonl> [<session2.jsonl> ...]
    python scripts/h7_bypass.py <session.jsonl> --events   # list each bypass

A bypass-to-update = an Edit/Write on a file with no prior Read of it in the same
session (research/benchmarking/methodology.md, H7). This prints the headline
count + diagnostic splits per transcript.

NOTE: this is the bypass side only. The H7 metric is bypass-WITH-CORRECTNESS —
pair these counts with whether the resulting edits pass tests (a high bypass rate
with low correctness is overconfident editing, not a win). And bucket the
PreToolUse `symbol_files_mtime_stale` hook reason before drawing conclusions: an
mtime-stale false-allow lets a Read through and depresses the bypass rate for a
reason unrelated to skeleton quality.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memlora.telemetry.bypass import analyze_bypass  # noqa: E402


def _report(path: Path, show_events: bool) -> None:
    text = path.read_text(encoding="utf-8", errors="replace")
    r = analyze_bypass(text)
    print(f"\n{path.name}")
    print(f"  modifications (Edit/Write)      : {r.total_modifications}")
    print(f"  bypass_to_update (headline)     : {r.bypass_to_update}")
    print(f"  bypass_rate                     : {r.bypass_rate:.1%}")
    print(f"  - edit_without_read (strong)    : {r.edit_without_read}")
    print(f"  - write_first_touch (ambiguous) : {r.write_first_touch}")
    print(f"  read calls                      : {r.read_calls}")
    print(f"  distinct files modified         : {r.distinct_files_modified}")
    if show_events and r.events:
        print("  bypass events:")
        for e in r.events:
            tag = " (self-created earlier)" if e.prior_write else ""
            print(f"    #{e.ordinal:<3} {e.tool:<12} {e.path}{tag}")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_events = "--events" in sys.argv[1:]
    if not args:
        print(f"Usage: {sys.argv[0]} <session.jsonl> [...] [--events]", file=sys.stderr)
        sys.exit(1)

    missing = [a for a in args if not Path(a).exists()]
    if missing:
        print(f"File(s) not found: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    for arg in args:
        _report(Path(arg), show_events)


if __name__ == "__main__":
    main()
