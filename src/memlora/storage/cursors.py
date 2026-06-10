"""Ingest-cursor storage for delta extraction (Sprint I / I2).

An ingest cursor tracks the last-processed JSONL line count and an anchor SHA256
(hash of the ANCHOR_LINES lines immediately before the high-water mark). This lets
session_end extract only the new delta on each Stop hook firing instead of
reprocessing the full growing transcript.

Fail-open invariant: any doubt about cursor validity (compaction detected, cursor
missing, anchor mismatch) triggers a full re-extraction. Never skip memory to
save CPU.
"""
from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass

# Number of JSONL lines to include before the high-water mark in each extraction.
# The overlap gives cross-turn attribution context; content_hash dedup in
# execute_merge makes re-extracting overlap idempotent.
OVERLAP_LINES = 20

# Lines hashed to form the compaction anchor.  Must be <= OVERLAP_LINES.
ANCHOR_LINES = 5

_now_ms = lambda: int(time.time() * 1000)


@dataclass
class IngestCursor:
    project_id: str
    session_id: str
    last_line_count: int
    anchor_sha256: str
    updated_at: int
    last_evidence_id: int | None = None  # tail of the evidence chain (I3)


def get_cursor(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
) -> IngestCursor | None:
    row = conn.execute(
        "SELECT * FROM ingest_cursors WHERE project_id=? AND session_id=?",
        (project_id, session_id),
    ).fetchone()
    if row is None:
        return None
    keys = row.keys()
    return IngestCursor(
        project_id=row["project_id"],
        session_id=row["session_id"],
        last_line_count=row["last_line_count"],
        anchor_sha256=row["anchor_sha256"],
        updated_at=row["updated_at"],
        last_evidence_id=row["last_evidence_id"] if "last_evidence_id" in keys else None,
    )


def save_cursor(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    last_line_count: int,
    anchor_sha256: str,
    last_evidence_id: int | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO ingest_cursors
            (project_id, session_id, last_line_count, anchor_sha256, last_evidence_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(project_id, session_id) DO UPDATE SET
            last_line_count  = excluded.last_line_count,
            anchor_sha256    = excluded.anchor_sha256,
            last_evidence_id = excluded.last_evidence_id,
            updated_at       = excluded.updated_at
        """,
        (project_id, session_id, last_line_count, anchor_sha256, last_evidence_id, _now_ms()),
    )


def compute_anchor(lines: list[str], up_to_line: int) -> str:
    """SHA256 of the ANCHOR_LINES lines immediately before `up_to_line`."""
    start = max(0, up_to_line - ANCHOR_LINES)
    anchor_text = "\n".join(lines[start:up_to_line])
    return hashlib.sha256(anchor_text.encode("utf-8", errors="replace")).hexdigest()


def slice_storage_delta(
    jsonl_text: str,
    cursor: IngestCursor | None,
) -> tuple[bytes, bool]:
    """Return the bytes to store in raw_evidence and whether this is a delta.

    Returns (content_bytes, is_delta):
    - is_delta=False: store the full JSONL (first run or compaction fallback).
      Content = all non-empty lines joined with newlines + trailing newline.
    - is_delta=True: store only the new lines since the cursor high-water mark.
      Content = new lines only (no overlap window — that is extraction-only).
      Concatenating root + all deltas byte-exactly reconstructs the full JSONL.

    The anchor check logic mirrors slice_jsonl_for_extraction; they share the
    same compaction detection so both consistently use delta or fall back together.
    """
    lines = [ln for ln in jsonl_text.splitlines() if ln.strip()]
    full_bytes = ("\n".join(lines) + "\n").encode("utf-8")

    if cursor is None or cursor.last_line_count == 0:
        return full_bytes, False

    if len(lines) < cursor.last_line_count:
        return full_bytes, False  # compaction

    expected_anchor = compute_anchor(lines, cursor.last_line_count)
    if expected_anchor != cursor.anchor_sha256:
        return full_bytes, False  # compaction

    new_lines = lines[cursor.last_line_count:]
    if not new_lines:
        # No new content — store an empty delta so the chain is unbroken.
        # (Dedup via content_sha256 will collapse identical empty deltas.)
        return b"", False  # nothing to chain

    delta_bytes = ("\n".join(new_lines) + "\n").encode("utf-8")
    return delta_bytes, True


def slice_jsonl_for_extraction(
    jsonl_text: str,
    cursor: IngestCursor | None,
) -> tuple[str, int, str]:
    """Return the JSONL slice to extract, the new line count, and the new anchor SHA256.

    If `cursor` is None or the anchor doesn't match (compaction detected), returns the
    full JSONL content (fail-open full re-extraction). Otherwise returns only the overlap
    window + new lines after the high-water mark.

    Returns: (jsonl_slice, new_line_count, new_anchor_sha256)
    """
    lines = [ln for ln in jsonl_text.splitlines() if ln.strip()]
    new_line_count = len(lines)
    new_anchor = compute_anchor(lines, new_line_count)

    if cursor is None or cursor.last_line_count == 0:
        return jsonl_text, new_line_count, new_anchor

    if new_line_count < cursor.last_line_count:
        # File shrank — compaction rewrote the transcript. Full re-extraction.
        return jsonl_text, new_line_count, new_anchor

    # Verify anchor: check that the lines just before the cursor haven't changed.
    expected_anchor = compute_anchor(lines, cursor.last_line_count)
    if expected_anchor != cursor.anchor_sha256:
        # Compaction or unexpected rewrite. Full re-extraction.
        return jsonl_text, new_line_count, new_anchor

    # Delta mode: extract overlap window + new lines.
    start = max(0, cursor.last_line_count - OVERLAP_LINES)
    delta_lines = lines[start:]
    return "\n".join(delta_lines), new_line_count, new_anchor
