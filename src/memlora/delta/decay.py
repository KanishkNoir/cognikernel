"""Storage-side weight decay and archival.

Distinct from compression-time hyperbolic recency (Stage 3):
  - Compression recency: computed on read, never writes to DB, used for ranking
  - Storage decay: writes weight × 0.92 to DB each session-end, enables archive

CONSTRAINT_HARD and APPROACH_ABANDONED_DO_NOT_RETRY are never archived —
they represent permanent architectural commitments.

Idempotency: decay for a given session_id is only applied once.
The last-applied session is recorded in the meta table so that a retry
or re-run does not compound the decay factor.
"""
from __future__ import annotations

import sqlite3

# Single source of truth for the archive floor is storage.events; re-exported
# here for the existing `from memlora.delta.decay import ARCHIVE_THRESHOLD`
# call sites (delta.merge, delta.__init__).
from memlora.storage.events import ARCHIVE_THRESHOLD  # noqa: F401  (re-export)

DECAY_FACTOR: float = 0.92

_PROTECTED_FROM_ARCHIVE: frozenset[str] = frozenset({
    "CONSTRAINT_HARD",
    "APPROACH_ABANDONED_DO_NOT_RETRY",
})

_META_KEY_PREFIX = "last_decay_session:"


def apply_decay_pass(
    conn: sqlite3.Connection,
    project_id: str,
    current_session_id: str,
) -> int:
    """Decay weights and archive stale events. Returns count of newly archived events.

    Idempotent: calling with the same (project_id, current_session_id) pair twice
    produces no additional decay.
    """
    meta_key = f"{_META_KEY_PREFIX}{project_id}"

    last_row = conn.execute(
        "SELECT value FROM meta WHERE key = ?", (meta_key,)
    ).fetchone()
    if last_row and last_row["value"] == current_session_id:
        return 0  # already applied for this session

    conn.execute(
        """
        UPDATE events
        SET weight = MAX(0.0, weight * ?)
        WHERE project_id = ?
          AND session_id != ?
          AND archived   = 0
        """,
        (DECAY_FACTOR, project_id, current_session_id),
    )

    # Archive only non-protected types that fell below threshold
    protected_placeholders = ",".join("?" * len(_PROTECTED_FROM_ARCHIVE))
    result = conn.execute(
        f"""
        UPDATE events
        SET archived = 1
        WHERE project_id = ?
          AND archived   = 0
          AND weight     < ?
          AND event_type NOT IN ({protected_placeholders})
        """,
        (project_id, ARCHIVE_THRESHOLD, *_PROTECTED_FROM_ARCHIVE),
    )
    archived_count = result.rowcount

    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        (meta_key, current_session_id),
    )

    return archived_count
