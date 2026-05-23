#!/usr/bin/env python3
"""Parse Claude Code usage.jsonl and report token counts per session.

Usage:
    python scripts/collect_telemetry.py <project_path>

Reads from ~/.claude/projects/<encoded-path>/usage.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def encode_project_path(project_path: str) -> str:
    """Match Claude Code's path encoding: replace : \\ / with -"""
    return str(Path(project_path).resolve()).replace(":", "-").replace("\\", "-").replace("/", "-")


def find_usage_file(project_path: str) -> Path | None:
    encoded = encode_project_path(project_path)
    base = Path.home() / ".claude" / "projects" / encoded
    for candidate in [base / "usage.jsonl", base / "stats.jsonl"]:
        if candidate.exists():
            return candidate
    return None


def summarize_usage(usage_file: Path) -> None:
    sessions: dict[str, dict] = defaultdict(lambda: {
        "input_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "output_tokens": 0,
        "records": 0,
    })
    with usage_file.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            session_id = record.get("session_id", "unknown")
            s = sessions[session_id]
            s["input_tokens"] += record.get("input_tokens", 0)
            s["cache_read_input_tokens"] += record.get("cache_read_input_tokens", 0)
            s["cache_creation_input_tokens"] += record.get("cache_creation_input_tokens", 0)
            s["output_tokens"] += record.get("output_tokens", 0)
            s["records"] += 1

    header = f"{'Session':<16} {'Input':>9} {'CacheRead':>10} {'CacheWrite':>11} {'Output':>9} {'Records':>8}  CacheHit%"
    print(header)
    print("-" * len(header))

    total_input = 0
    total_cache_read = 0
    total_output = 0

    for sid, s in sorted(sessions.items()):
        total_input += s["input_tokens"]
        total_cache_read += s["cache_read_input_tokens"]
        total_output += s["output_tokens"]
        hit_rate = s["cache_read_input_tokens"] / max(s["input_tokens"], 1) * 100
        print(
            f"{sid[:15]:<16} {s['input_tokens']:>9,} {s['cache_read_input_tokens']:>10,} "
            f"{s['cache_creation_input_tokens']:>11,} {s['output_tokens']:>9,} {s['records']:>8}  "
            f"{hit_rate:.1f}%"
        )

    print("-" * len(header))
    overall_hit = total_cache_read / max(total_input, 1) * 100
    print(f"  Total input tokens:     {total_input:>12,}")
    print(f"  Total cache reads:      {total_cache_read:>12,}  ({overall_hit:.1f}% hit rate)")
    print(f"  Total output tokens:    {total_output:>12,}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <project_path>", file=sys.stderr)
        sys.exit(1)

    project_path = sys.argv[1]
    usage_file = find_usage_file(project_path)
    if usage_file is None:
        print(f"No usage.jsonl found for: {Path(project_path).resolve()}", file=sys.stderr)
        print("Claude Code writes usage data to ~/.claude/projects/<encoded-path>/", file=sys.stderr)
        sys.exit(1)

    print(f"Reading: {usage_file}\n")
    summarize_usage(usage_file)


if __name__ == "__main__":
    main()
