"""Admission-gate rule counters.

Feeds `cognikernel doctor` and the longitudinal impact claim: how often each
defect rule fires in real use. Never raises — telemetry must not be able to
break extraction, which is the thing it is measuring.
"""
from __future__ import annotations

import logging
import sqlite3
import time

_log = logging.getLogger("cognikernel.quality")


def record_verdict(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    rule_id: str,
) -> None:
    """Increment the counter for one rule firing. Best-effort."""
    if not rule_id:
        return
    try:
        conn.execute(
            """
            INSERT INTO quality_telemetry
                (project_id, session_id, rule_id, count, updated_at)
            VALUES (?, ?, ?, 1, ?)
            ON CONFLICT (project_id, session_id, rule_id) DO UPDATE SET
                count      = count + 1,
                updated_at = excluded.updated_at
            """,
            (project_id, session_id, rule_id, int(time.time() * 1000)),
        )
        conn.commit()
    except Exception as exc:
        _log.warning("quality telemetry write failed: %s", exc)


def get_rule_counts(
    conn: sqlite3.Connection,
    project_id: str,
    limit: int = 10,
) -> list[dict]:
    """Return [{rule_id, total}] for a project, most frequent first."""
    try:
        rows = conn.execute(
            """
            SELECT rule_id, SUM(count) AS total
            FROM quality_telemetry
            WHERE project_id = ?
            GROUP BY rule_id
            ORDER BY total DESC, rule_id ASC
            LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()
        return [{"rule_id": r[0], "total": r[1]} for r in rows]
    except Exception as exc:
        _log.warning("quality telemetry read failed: %s", exc)
        return []
