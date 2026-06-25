"""Render ledger (J4.1) — which events were verbatim exposed to which session.

Channels: 'block' (session-start context block), 'ck1' (per-prompt injection),
'recall' (schema-ready; not written until MCP calls carry a session identity).

Powers CK-1's "not already in context" redundancy filter and exposure
auditing (the access-log pattern). Observability state, never load-bearing:
a missing/empty ledger only risks re-injecting something already shown, so
every function here fails open.
"""
from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterable


def record_rendered(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    event_ids: Iterable[int],
    channel: str,
) -> int:
    """INSERT OR IGNORE (dedup index) the exposed event ids. Returns rows added."""
    ids = [int(i) for i in event_ids if i is not None]
    if not ids or not session_id:
        return 0
    now = int(time.time() * 1000)
    try:
        cur = conn.executemany(
            "INSERT OR IGNORE INTO render_ledger "
            "(project_id, session_id, event_id, channel, rendered_at) "
            "VALUES (?, ?, ?, ?, ?)",
            [(project_id, session_id, i, channel, now) for i in ids],
        )
        conn.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return 0


def rendered_event_ids(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
) -> set[int]:
    """All event ids exposed to `session_id` on any channel. {} on any error."""
    if not session_id:
        return set()
    try:
        return {
            r[0]
            for r in conn.execute(
                "SELECT event_id FROM render_ledger "
                "WHERE project_id = ? AND session_id = ?",
                (project_id, session_id),
            )
        }
    except Exception:
        return set()
