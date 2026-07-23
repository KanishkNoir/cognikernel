"""H7 — bypass-to-update detection over a Claude Code session transcript.

A *bypass-to-update* event is an Edit/Write tool call on a file path with **no
prior Read of that file earlier in the same session**: the model changed a file
relying on the injected skeleton/decisions instead of reading it first. This is
the H7 hypothesis (`research/benchmarking/methodology.md`). It automates the
count the run sheet (`research/benchmarking/run_sheet.md`) currently records by
hand ("Edit/Write with no prior Read of that file").

Headline = the literal documented definition (no prior Read), so this tool
reproduces-then-replaces the manual count and stays comparable across arms/runs.
Finer splits are exposed as *diagnostic* fields, not folded into the headline:
  - edit_without_read  — Edit-family on a never-read file. The strong signal:
                         an Edit must match existing content, so editing a file
                         never read this session is direct skeleton-trust.
  - write_first_touch  — a Write with no prior Read and no prior Write. Ambiguous
                         (creating a new file vs. blind-overwriting an existing
                         one); a fresh file isn't in the skeleton, so this is not
                         clean evidence either way.

Two things this module deliberately does NOT do:
  1. The correctness gate. methodology.md: a high bypass rate with low
     correctness is a *warning* (overconfident editing), not a win. Pair these
     counts with test outcomes before drawing any conclusion.
  2. Account for the PreToolUse hook. An mtime-stale false-allow (see
     `integration.lookup`) lets a Read through that skeleton-trust would have
     blocked, depressing the bypass rate for a non-skeleton reason. Bucket that
     hook reason before interpreting the number.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field

from cognikernel.utils.paths import canonicalize_path

_READ_TOOLS = frozenset({"Read"})
_EDIT_TOOLS = frozenset({"Edit", "MultiEdit", "NotebookEdit"})
_WRITE_TOOLS = frozenset({"Write"})
# Most file tools name the target `file_path`; NotebookEdit uses `notebook_path`.
_PATH_KEY_OVERRIDES = {"NotebookEdit": "notebook_path"}


@dataclass(frozen=True)
class ToolCall:
    name: str
    path: str       # canonical key for matching ('' if no/unresolved path)
    raw_path: str   # the original string from the transcript (for display)


@dataclass(frozen=True)
class BypassEvent:
    ordinal: int    # 1-based index among modification (Edit/Write) calls
    tool: str
    path: str
    prior_write: bool  # was this path written earlier this session? (likely self-created)


@dataclass
class BypassReport:
    total_modifications: int = 0   # all Edit/Write-family calls — the denominator
    bypass_to_update: int = 0      # HEADLINE: modifications with no prior Read
    edit_without_read: int = 0     # diagnostic: Edit-family subset (strong signal)
    write_first_touch: int = 0     # diagnostic: Write, no prior Read AND no prior Write
    read_calls: int = 0
    distinct_files_modified: int = 0
    events: list[BypassEvent] = field(default_factory=list)

    @property
    def bypass_rate(self) -> float:
        """bypass_to_update / total_modifications, or 0.0 when no modifications."""
        if self.total_modifications == 0:
            return 0.0
        return self.bypass_to_update / self.total_modifications


def _path_key(name: str, tool_input: dict | None, cwd: str) -> tuple[str, str]:
    """Return (canonical_key, raw_path) for a tool call, or ('', '') if no path.

    Both Read and Edit/Write paths are funneled through canonicalize_path with the
    record's cwd so an absolute path on one side and a relative path on the other
    map to the same key (otherwise a naive string compare false-positives a
    bypass). Falls back to a lowercase forward-slash form if canonicalization
    can't resolve (e.g., path outside cwd) so a modification is never silently
    dropped.
    """
    key = _PATH_KEY_OVERRIDES.get(name, "file_path")
    raw = ((tool_input or {}).get(key) or "").strip()
    if not raw:
        return "", ""
    canon = canonicalize_path(raw, cwd or None)
    if not canon:
        canon = raw.replace("\\", "/").lower()
    return canon, raw


def iter_tool_calls(jsonl_text: str) -> Iterator[ToolCall]:
    """Yield Read/Edit/Write-family tool calls, in order, from a session JSONL.

    Tool calls are `tool_use` blocks nested inside assistant `message.content`
    (mirrors telemetry.ingest) — NOT top-level records. Malformed lines are
    skipped.
    """
    for line in jsonl_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("type") != "assistant":
            continue
        content = rec.get("message", {}).get("content")
        if not isinstance(content, list):
            continue
        cwd = rec.get("cwd") or ""
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name", "")
            if name not in _READ_TOOLS and name not in _EDIT_TOOLS and name not in _WRITE_TOOLS:
                continue
            canon, raw = _path_key(name, block.get("input"), cwd)
            yield ToolCall(name=name, path=canon, raw_path=raw)


def analyze_bypass(jsonl_text: str) -> BypassReport:
    """Compute the H7 bypass report for one session transcript.

    Single forward pass: a modification on a path not yet Read this session is a
    bypass (headline). State (reads/writes seen) is updated *after* each call so
    the check always reflects what was known before it.
    """
    report = BypassReport()
    read_paths: set[str] = set()
    written_paths: set[str] = set()
    modified: set[str] = set()

    for call in iter_tool_calls(jsonl_text):
        if call.name in _READ_TOOLS:
            if call.path:
                read_paths.add(call.path)
                report.read_calls += 1
            continue

        # Edit/Write-family modification.
        if not call.path:
            continue
        report.total_modifications += 1
        modified.add(call.path)
        prior_read = call.path in read_paths
        prior_write = call.path in written_paths

        if not prior_read:
            report.bypass_to_update += 1
            report.events.append(
                BypassEvent(
                    ordinal=report.total_modifications,
                    tool=call.name,
                    path=call.path,
                    prior_write=prior_write,
                )
            )
            if call.name in _EDIT_TOOLS:
                report.edit_without_read += 1
            elif call.name in _WRITE_TOOLS and not prior_write:
                report.write_first_touch += 1

        written_paths.add(call.path)

    report.distinct_files_modified = len(modified)
    return report
