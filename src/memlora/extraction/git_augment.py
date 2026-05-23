"""Git diff integration for COMPONENT_STATUS event extraction.

Produces events for every file touched in a session, weighted by churn size.
Cross-references transcript events against git evidence to boost corroborated
abandoned-approach events.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from typing import Any

from memlora.extraction.hashing import compute_content_hash
from memlora.storage.events import Event

# Matches "M\tpath/to/file" or "A\tfile" or "R100\told\tnew" etc.
_NAME_STATUS = re.compile(r"^([AMDRC]\d*)\t(.+)$", re.MULTILINE)
# Matches "path/to/file | 42 +++---" lines from --stat
_STAT_LINE = re.compile(r"^(.+?)\s*\|\s*(\d+)\s*([\+\-]*)$", re.MULTILINE)

_ABANDONED_TYPES = frozenset(
    {"APPROACH_ABANDONED", "APPROACH_ABANDONED_DO_NOT_RETRY"}
)

# Weight formula: 0.5 base + up to 0.4 proportional to churn (cap at 200 lines).
_WEIGHT_BASE = 0.5
_WEIGHT_CHURN_SCALE = 200.0
_WEIGHT_CHURN_MAX = 0.4


@dataclass
class FileChange:
    path: str
    change_type: str   # "added" | "modified" | "deleted" | "renamed"
    lines_changed: int = 0
    old_path: str = ""  # populated for renames (the path before the rename)


def run_git_diff(repo_path: str) -> str | None:
    """Run git diff and return output, or None on error."""
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD~1..HEAD", "--stat", "--name-status"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=10,
        )
        return result.stdout if result.returncode == 0 else None
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def parse_diff(diff_output: str) -> list[FileChange]:
    """Parse name-status lines from git diff output into FileChange objects."""
    stat_counts = _parse_stat_counts(diff_output)
    changes: list[FileChange] = []

    for m in _NAME_STATUS.finditer(diff_output):
        status = m.group(1).upper()
        paths = m.group(2).split("\t")
        path = paths[-1].strip()  # for renames, take the new path

        if status.startswith("R"):
            change_type = "renamed"
        elif status == "A":
            change_type = "added"
        elif status == "D":
            change_type = "deleted"
        else:
            change_type = "modified"

        old_path = paths[0].strip() if change_type == "renamed" and len(paths) > 1 else ""
        changes.append(
            FileChange(
                path=path,
                change_type=change_type,
                lines_changed=stat_counts.get(path, 0),
                old_path=old_path,
            )
        )

    return changes


def extract_git_events(
    diff_output: str,
    project_id: str,
    session_id: str,
) -> list[Event]:
    """Produce COMPONENT_STATUS events from a git diff string."""
    events: list[Event] = []

    for change in parse_diff(diff_output):
        weight = _WEIGHT_BASE + min(
            _WEIGHT_CHURN_MAX, change.lines_changed / _WEIGHT_CHURN_SCALE
        )
        description = (
            f"{change.path} {change.change_type} ({change.lines_changed} lines)"
        )
        payload: dict[str, Any] = {
            "path": change.path,
            "status": "in_flux",
            "lines_changed": change.lines_changed,
            "change_type": change.change_type,
            "intent": infer_intent_from_path(change.path),
            "description": description,
            "rationale": "",
        }
        events.append(
            Event(
                project_id=project_id,
                session_id=session_id,
                event_type="COMPONENT_STATUS",
                payload=payload,
                content_hash=compute_content_hash("COMPONENT_STATUS", description),
                weight=weight,
            )
        )

    return events


def infer_intent_from_path(path: str) -> str:
    """Guess file intent from path conventions — fallback is the bare path."""
    parts = path.lower().replace("\\", "/").split("/")
    stem = parts[-1].rsplit(".", 1)[0] if "." in parts[-1] else parts[-1]

    # Directory hints take priority.
    dir_hints = {
        frozenset({"auth", "authentication", "login"}): "authentication",
        frozenset({"routes", "router", "api", "endpoints"}): "API routes",
        frozenset({"models", "model", "schema", "schemas"}): "data model",
        frozenset({"utils", "util", "helpers", "common", "shared"}): "utilities",
        frozenset({"tests", "test", "spec", "__tests__"}): "tests",
        frozenset({"migrations", "migration"}): "database migration",
        frozenset({"middleware", "interceptors"}): "middleware",
        frozenset({"config", "settings", "configuration"}): "configuration",
        frozenset({"components", "views", "pages"}): "UI component",
        frozenset({"services", "service"}): "service layer",
    }
    for keywords, intent in dir_hints.items():
        if any(p in keywords for p in parts[:-1]):
            return intent

    # File stem hints.
    for keyword, intent in [
        ("middleware", "middleware"),
        ("router", "routing"),
        ("handler", "request handler"),
        ("controller", "controller"),
        ("service", "service layer"),
        ("repository", "data repository"),
        ("validator", "validation"),
        ("serializer", "serialization"),
        ("migration", "database migration"),
        ("test", "tests"),
    ]:
        if keyword in stem:
            return intent

    return path


def cross_reference_signals(
    transcript_events: list[Event],
    git_events: list[Event],
) -> list[Event]:
    """Boost abandoned events corroborated by matching file changes in the git diff."""
    git_paths = [e.payload.get("path", "").lower() for e in git_events]

    for event in transcript_events:
        if event.event_type not in _ABANDONED_TYPES:
            continue
        hints = _extract_hints(event.payload.get("description", ""))
        if any(any(h in gp for h in hints) for gp in git_paths):
            event.weight = min(2.0, event.weight + 0.2)
            event.payload["git_corroborated"] = True

    return transcript_events


# ── internals ────────────────────────────────────────────────────────────────

def _parse_stat_counts(diff_output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in _STAT_LINE.finditer(diff_output):
        path = m.group(1).strip()
        marker = m.group(3) or ""
        counts[path] = marker.count("+") + marker.count("-")
    return counts


def _extract_hints(text: str) -> list[str]:
    """Extract 3+ character lowercase tokens that could be library/path fragments."""
    return [w for w in re.findall(r"[a-z][a-z0-9\-]{2,}", text.lower()) if len(w) >= 3]
