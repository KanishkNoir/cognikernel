"""Component dependency cascade — one level deep.

When a COMPONENT_STATUS event marks a file as 'blocked' or 'abandoned',
emit 'needs_review' status events for every file that directly depends on it.

The cascade is capped at one level: events whose payload contains
'cascaded_from' are never cascaded further.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memlora.storage.events import Event

_CASCADE_STATUSES: frozenset[str] = frozenset({"blocked", "abandoned"})


def cascade_component_status(
    conn: sqlite3.Connection,
    status_event: Event,
) -> int:
    """Emit needs_review events for direct dependents of a blocked/abandoned file.

    Returns the number of cascade events inserted or deduplicated.
    Guard: events with 'cascaded_from' in payload are never cascaded further.
    """
    payload = status_event.payload
    if payload.get("status") not in _CASCADE_STATUSES:
        return 0
    if "cascaded_from" in payload:
        return 0  # one-level cap

    target_path = payload.get("path", "")
    if not target_path:
        return 0

    project_id = status_event.project_id
    session_id = status_event.session_id

    rows = conn.execute(
        """
        SELECT id, payload FROM events
        WHERE project_id    = ?
          AND event_type    = 'COMPONENT_STATUS'
          AND archived      = 0
          AND superseded_by IS NULL
          AND json_extract(payload, '$.dependencies') LIKE ?
        """,
        (project_id, f'%"{target_path}"%'),
    ).fetchall()

    count = 0
    for row in rows:
        dep_payload = json.loads(row["payload"])
        dep_path = dep_payload.get("path")
        if not dep_path or dep_path == target_path:
            continue

        cascade_payload = {
            "path": dep_path,
            "status": "needs_review",
            "reason": f"depends on {target_path} which is {payload['status']}",
            "cascaded_from": status_event.id,
        }
        from memlora.extraction.hashing import compute_content_hash
        cascade_hash = compute_content_hash(
            "COMPONENT_STATUS",
            f"{dep_path} needs_review cascade from {target_path}",
        )

        try:
            conn.execute(
                """
                INSERT INTO events
                    (project_id, session_id, created_at, event_type,
                     payload, content_hash, weight, mention_count)
                VALUES (?, ?, ?, 'COMPONENT_STATUS', ?, ?, 0.6, 1)
                """,
                (
                    project_id,
                    session_id,
                    int(time.time() * 1000),
                    json.dumps(cascade_payload, sort_keys=True, separators=(",", ":")),
                    cascade_hash,
                ),
            )
        except Exception:
            conn.execute(
                """
                UPDATE events
                SET mention_count = mention_count + 1
                WHERE project_id = ? AND content_hash = ?
                """,
                (project_id, cascade_hash),
            )
        count += 1

    return count
