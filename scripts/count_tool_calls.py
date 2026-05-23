#!/usr/bin/env python3
"""Count tool calls in a Claude Code session transcript.

Usage:
    python scripts/count_tool_calls.py <session.jsonl>
    python scripts/count_tool_calls.py <session.jsonl> --before-code

With --before-code: only counts tool calls before the first assistant message
containing a fenced code block (```) — useful for measuring cold-start read overhead.

Session transcripts are at:
    ~/.claude/projects/<encoded-path>/sessions/<session-id>.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def count_tool_calls(
    transcript_path: Path,
    before_code_only: bool = False,
) -> tuple[Counter, int]:
    """Returns (tool_name_counter, total_records)."""
    counts: Counter = Counter()
    total = 0
    done = False

    with transcript_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or done:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            msg_type = record.get("type", "")

            if before_code_only and msg_type == "assistant":
                content = record.get("message", {}).get("content", "")
                if isinstance(content, str) and "```" in content:
                    done = True
                    continue
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            if "```" in block.get("text", ""):
                                done = True
                                break

            if msg_type == "tool_use":
                name = record.get("name") or record.get("tool_name", "unknown")
                counts[name] += 1

    return counts, total


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(f"Usage: {sys.argv[0]} <session.jsonl> [--before-code]", file=sys.stderr)
        sys.exit(1)

    transcript_path = Path(args[0])
    before_code = "--before-code" in args

    if not transcript_path.exists():
        print(f"File not found: {transcript_path}", file=sys.stderr)
        sys.exit(1)

    counts, total = count_tool_calls(transcript_path, before_code_only=before_code)

    label = "(before first code block)" if before_code else "(all)"
    print(f"Tool calls {label} in: {transcript_path.name}\n")

    if not counts:
        print("  No tool calls found.")
        return

    max_count = max(counts.values())
    for name, count in counts.most_common():
        bar = "#" * int(count / max_count * 20)
        print(f"  {name:<30} {count:>4}  {bar}")

    print(f"\n  Total tool calls: {sum(counts.values())}")
    if before_code:
        read_like = sum(counts[n] for n in ("Read", "Glob", "Grep"))
        print(f"  Read/Glob/Grep (cold-start overhead): {read_like}")


if __name__ == "__main__":
    main()
