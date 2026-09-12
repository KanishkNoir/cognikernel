"""`cognikernel show <project> --as-of <when>` (S5 T-503, gate G3).

Replays what the store believed at a past time, and states its own horizon.
Transition times exist only from migration 021 on: measured across 179 real
stores, every one of 483 superseded and 58 archived claims had none. Those
claims are listed under "Timing unknown" rather than guessed live or ended.

Ranking is rebuilt from the claims live at that time, so session recency is as of
then. File centrality still comes from today's import graph — the store keeps no
history of the code — and the output says so.
"""
from __future__ import annotations

import datetime
import logging
import re
import sqlite3
import subprocess
from pathlib import Path
from typing import Any

from cognikernel.integration.why import _session, _session_label, _when
from cognikernel.storage.projections import build_projection
from cognikernel.storage.provenance import claims_as_of, session_order
from cognikernel.utils.decision_key import derive_decision_key

_log = logging.getLogger("cognikernel.as_of")

ACCEPTED_FORMATS = "YYYY-MM-DD (end of that day), 'YYYY-MM-DD HH:MM', or a git commit sha"

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}$")
_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")

_MAX_DESCRIPTION = 110
_MAX_COMPONENTS = 10


def _commit_time(ref: str, project_path: str | Path) -> int | None:
    """Commit time of `ref` in epoch ms, or None when it is not a commit there."""
    try:
        result = subprocess.run(
            ["git", "-C", str(project_path), "show", "-s", "--format=%ct", f"{ref}^{{commit}}"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as exc:
        _log.warning("show --as-of: git could not resolve %s: %s", ref, exc)
        return None
    stamp = result.stdout.strip()
    if result.returncode != 0 or not stamp.isdigit():
        return None
    return int(stamp) * 1000


def parse_when(value: str, project_path: str | Path) -> tuple[int, str]:
    """Turn `--as-of` into (epoch ms, label). Raises ValueError naming the formats."""
    value = value.strip()
    try:
        if _DATE.match(value):
            day = datetime.datetime.strptime(value, "%Y-%m-%d")
            end = day.replace(hour=23, minute=59, second=59, microsecond=999000)
            return int(end.timestamp() * 1000), f"{value} (end of day)"
        if _DATETIME.match(value):
            moment = datetime.datetime.strptime(value.replace("T", " "), "%Y-%m-%d %H:%M")
            return int(moment.timestamp() * 1000), moment.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        raise ValueError(f"{value!r} is not a valid date; expected {ACCEPTED_FORMATS}") from None
    if _SHA.match(value):
        at = _commit_time(value, project_path)
        if at is not None:
            return at, f"commit {value[:7]} ({_when(at)})"
        raise ValueError(f"{value!r} is not a commit in {Path(project_path).resolve()}; "
                         f"expected {ACCEPTED_FORMATS}")
    raise ValueError(f"cannot read {value!r} as a time; expected {ACCEPTED_FORMATS}")


def _short(text: str) -> str:
    return text if len(text) <= _MAX_DESCRIPTION else text[:_MAX_DESCRIPTION - 3] + "..."


def explain_as_of(conn: sqlite3.Connection, project_id: str, at: int, label: str) -> dict[str, Any]:
    claims = claims_as_of(conn, project_id, at)
    order = session_order(conn, project_id, at_ms=at)
    # rebuild_projection backfills missing decision keys in the store and then
    # consolidates same-topic choices. The as-of view must consolidate the same
    # way without writing, so keys are derived on these in-memory copies only.
    for event in claims.live:
        if event.decision_key is None:
            event.decision_key = derive_decision_key(event.payload, event.event_type)
    projection = build_projection(conn, project_id, claims.live, built_at=at)

    def entry(rec: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": rec["id"],
            "event_type": rec["event_type"],
            "description": rec["payload"].get("description") or "",
            "weight": rec["weight"],
            "created_at": rec["created_at"],
            "session": _session(order, rec["session_id"]),
        }

    return {
        "as_of": at,
        "label": label,
        "counts": {"live": len(claims.live), "ended": len(claims.ended), "unknown": len(claims.unknown)},
        "earliest_claim": claims.earliest_claim,
        "first_recorded_end": claims.first_recorded_end,
        "hard_constraints": [entry(r) for r in projection.hard_constraints],
        "decisions": [entry(r) for r in projection.ranked_decisions],
        "graveyard": [entry(r) for r in projection.graveyard],
        "active_threads": [entry(r) for r in projection.active_threads],
        "components": [
            {"id": r["id"], "path": path, "status": r["payload"].get("status", "unknown")}
            for path, r in sorted(projection.component_map.items())
        ],
        "unknown": [
            {
                "id": e.id,
                "event_type": e.event_type,
                "description": e.payload.get("description") or "",
                "created_at": e.created_at,
                "session": _session(order, e.session_id),
                "superseded_by": e.superseded_by,
                "archived": e.archived,
            }
            for e in claims.unknown
        ],
    }


def _day(ms: int) -> str:
    return datetime.datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d")


def _horizon(data: dict[str, Any]) -> list[str]:
    """Counts, end-time horizon and centrality caveat — stated on every view,
    including an empty one."""
    counts = data["counts"]
    lines = [f"  {counts['live']} live · {counts['ended']} ended by then · "
             f"{counts['unknown']} with no recorded end time"]
    if data["first_recorded_end"] and counts["unknown"]:
        lines.append(f"  end times are recorded from {_day(data['first_recorded_end'])}; "
                     "a claim replaced or archived before then has none")
    elif data["first_recorded_end"]:
        lines.append(f"  end times are recorded from {_day(data['first_recorded_end'])}; "
                     "every claim replaced or archived up to then has one")
    else:
        lines.append("  no end times are recorded in this store; "
                     "every replaced or archived claim is listed under Timing unknown")
    lines.append("  ranking uses the sessions up to then; file centrality uses today's import graph")
    return lines


def render_as_of(data: dict[str, Any]) -> str:
    counts = data["counts"]
    if not any(counts.values()):
        first = data["earliest_claim"]
        tail = f"; the first claim was recorded {_when(first)}" if first else ""
        return "\n".join([f"No claims existed yet as of {data['label']}{tail}.", *_horizon(data)])

    lines = [f"As of {data['label']}", *_horizon(data)]

    for title, key in (("Hard constraints", "hard_constraints"), ("Decisions", "decisions"),
                       ("Do not retry", "graveyard"), ("Open threads", "active_threads")):
        if not data[key]:
            continue
        lines += ["", title]
        lines += [f"  #{item['id']}  {_short(item['description'])}  ·  {_session_label(item['session'])}"
                  for item in data[key]]

    components = data["components"]
    if components:
        lines += ["", f"Components ({len(components)} tracked)"]
        lines += [f"  {c['path']}  {c['status']}" for c in components[:_MAX_COMPONENTS]]
        if len(components) > _MAX_COMPONENTS:
            lines.append(f"  ... {len(components) - _MAX_COMPONENTS} more (see --json)")

    if data["unknown"]:
        lines += ["", "Timing unknown — replaced or archived at a time the store did not record"]
        for item in data["unknown"]:
            how = (f"superseded by #{item['superseded_by']}" if item["superseded_by"] is not None
                   else "archived")
            lines.append(f"  #{item['id']}  {item['event_type']}  {_short(item['description'])}  ·  "
                         f"{_session_label(item['session'])} · created {_when(item['created_at'])} · {how}")
    return "\n".join(lines)
