"""Read-only provenance queries behind `cognikernel why` (S5 T-502).

Four questions a user asks of a claim, answered from what the store already
holds, with no schema change and no writes:

  find_claims         which claim is this — including claims since replaced
  session_order       which session did it come from, as a position in order
  supersession_chain  what it replaced, and what replaced it
  claim_provenance    which captured evidence it was extracted from

Session order is derived, never stored: a session's position is the rank of its
first sighting — its earliest raw_evidence capture, or its earliest event for a
session that stored no evidence. Session ids alone cannot be ordered, which is
how recalled history ended up attributed to the wrong session.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from cognikernel.storage.events import Event, _row_to_event

_ID_SUBJECT = re.compile(r"^#?(\d+)$")
_TERM = re.compile(r"[a-z0-9_]+")

# Supersession chains are a handful of restatements; this bounds the walk over a
# pathological or corrupt store rather than any realistic history.
_MAX_CHAIN = 10_000


@dataclass(frozen=True)
class SessionPosition:
    session_id: str
    position: int      # 1-based, by first sighting
    total: int         # sessions known to the store for this project
    first_seen: int    # epoch milliseconds


def session_order(conn: sqlite3.Connection, project_id: str) -> dict[str, SessionPosition]:
    """Every session of a project, keyed by id, numbered by first sighting."""
    rows = conn.execute(
        """
        SELECT session_id, MIN(first_seen) AS first_seen
        FROM (
            SELECT session_id, MIN(captured_at) AS first_seen
              FROM raw_evidence WHERE project_id = ? GROUP BY session_id
            UNION ALL
            SELECT session_id, MIN(created_at) AS first_seen
              FROM events WHERE project_id = ? GROUP BY session_id
        )
        GROUP BY session_id
        ORDER BY first_seen, session_id
        """,
        (project_id, project_id),
    ).fetchall()
    total = len(rows)
    return {
        row["session_id"]: SessionPosition(row["session_id"], position, total, row["first_seen"])
        for position, row in enumerate(rows, 1)
    }


def is_live(event: Event) -> bool:
    return event.superseded_by is None and not event.archived


def find_claims(
    conn: sqlite3.Connection,
    project_id: str,
    subject: str,
    limit: int = 5,
) -> list[Event]:
    """Claims matching `subject`: `#123` / `123` by id, otherwise every term.

    Searches superseded and archived claims too — the history is the point — so
    it scans the project's events rather than the FTS index, which only covers
    live claims. Live claims come first, then by weight, newest first.
    """
    subject = subject.strip()
    by_id = _ID_SUBJECT.match(subject)
    if by_id:
        row = conn.execute(
            "SELECT * FROM events WHERE id = ? AND project_id = ?",
            (int(by_id.group(1)), project_id),
        ).fetchone()
        return [_row_to_event(row)] if row else []

    terms = set(_TERM.findall(subject.casefold()))
    if not terms:
        return []
    matches: list[Event] = []
    for row in conn.execute("SELECT * FROM events WHERE project_id = ?", (project_id,)):
        event = _row_to_event(row)
        text = " ".join(str(event.payload.get(key) or "") for key in ("description", "subject", "rationale"))
        if terms <= set(_TERM.findall(text.casefold())):
            matches.append(event)
    matches.sort(key=lambda e: (not is_live(e), -e.weight, -(e.id or 0)))
    return matches[:limit]


def supersession_chain(conn: sqlite3.Connection, event_id: int) -> list[Event]:
    """The claim plus everything it replaced or was replaced by, oldest first.

    Walks superseded_by in both directions with a visited set, so a cycle left by
    a store written before the cycle guard cannot loop.
    """
    seen: dict[int, Event] = {}
    frontier = [event_id]
    while frontier and len(seen) < _MAX_CHAIN:
        current = frontier.pop()
        if current in seen:
            continue
        row = conn.execute("SELECT * FROM events WHERE id = ?", (current,)).fetchone()
        if row is None:
            continue
        event = _row_to_event(row)
        seen[current] = event
        if event.superseded_by is not None:
            frontier.append(event.superseded_by)
        frontier.extend(
            r["id"] for r in conn.execute("SELECT id FROM events WHERE superseded_by = ?", (current,))
        )
    return sorted(seen.values(), key=lambda e: (e.created_at, e.id or 0))


def claim_provenance(conn: sqlite3.Connection, event_id: int) -> list[dict[str, Any]]:
    """The evidence rows a claim was extracted from, earliest capture first.

    Offsets (sentence_index, window_*) and matched_phrase are returned as stored;
    the v2 extractor leaves them NULL, and callers must say so rather than
    implying a precise source location exists.
    """
    rows = conn.execute(
        """
        SELECT p.raw_evidence_id AS evidence_id,
               r.session_id, r.source_type, r.source_path, r.captured_at,
               p.extractor_version, p.matched_phrase, p.sentence_index,
               p.window_start, p.window_end, p.confidence, p.transformation_notes
        FROM event_provenance p
        JOIN raw_evidence r ON r.id = p.raw_evidence_id
        WHERE p.event_id = ?
        ORDER BY r.captured_at, p.raw_evidence_id
        """,
        (event_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def claim_transitions(conn: sqlite3.Connection, event_ids: list[int | None]) -> dict[int, dict[str, Any]]:
    """When and why each claim stopped being live (migration 021 columns).

    NULL means not recorded — every row written before migration 021 — never
    "did not happen": a superseded pre-021 claim has superseded_by set and
    superseded_at NULL.
    """
    ids = sorted({i for i in event_ids if i is not None})
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT id, superseded_at, supersede_reason, archived_at FROM events WHERE id IN ({marks})",
        ids,
    ).fetchall()
    return {
        row["id"]: {
            "superseded_at": row["superseded_at"],
            "supersede_reason": row["supersede_reason"],
            "archived_at": row["archived_at"],
        }
        for row in rows
    }


@dataclass
class AsOfClaims:
    """Every claim created by `at`, sorted into exactly one of three buckets."""

    at: int
    live: list[Event]
    ended: list[Event]
    unknown: list[Event]
    earliest_claim: int | None        # first created_at in the project
    first_recorded_end: int | None    # first superseded_at / archived_at recorded at all


def claims_as_of(conn: sqlite3.Connection, project_id: str, at_ms: int) -> AsOfClaims:
    """What the store believed at `at_ms`, from the 021 transition columns (T-503).

    A claim created at or before `at_ms` is:
      ended    superseded or archived at a RECORDED time at or before `at_ms`
      unknown  superseded or archived with no recorded time (every pre-021 row)
      live     otherwise — including a claim replaced only after `at_ms`

    Unknown is never folded into live or ended: a guess either way would make
    the replay confidently wrong instead of honestly bounded.
    """
    live: list[Event] = []
    ended: list[Event] = []
    unknown: list[Event] = []
    rows = conn.execute(
        "SELECT * FROM events WHERE project_id = ? AND created_at <= ? ORDER BY id",
        (project_id, at_ms),
    ).fetchall()
    for row in rows:
        recorded_ends: list[int] = []
        unrecorded_end = False
        if row["superseded_by"] is not None:
            if row["superseded_at"] is None:
                unrecorded_end = True
            else:
                recorded_ends.append(row["superseded_at"])
        if row["archived"]:
            if row["archived_at"] is None:
                unrecorded_end = True
            else:
                recorded_ends.append(row["archived_at"])

        event = _row_to_event(row)
        if any(end <= at_ms for end in recorded_ends):
            ended.append(event)
        elif unrecorded_end:
            unknown.append(event)
        else:
            live.append(event)

    earliest = conn.execute(
        "SELECT MIN(created_at) FROM events WHERE project_id = ?", (project_id,)
    ).fetchone()[0]
    first_end = conn.execute(
        """
        SELECT MIN(t) FROM (
            SELECT superseded_at AS t FROM events WHERE project_id = ? AND superseded_at IS NOT NULL
            UNION ALL
            SELECT archived_at AS t FROM events WHERE project_id = ? AND archived_at IS NOT NULL
        )
        """,
        (project_id, project_id),
    ).fetchone()[0]
    return AsOfClaims(at_ms, live, ended, unknown, earliest, first_end)
