from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import zlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RawEvidence:
    id: int
    project_id: str
    session_id: str
    source_type: str
    source_path: str
    captured_at: int
    content_sha256: str
    content_encoding: str
    content: bytes
    original_size_bytes: int
    stored_size_bytes: int
    metadata: dict[str, Any]


def store_evidence(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    source_type: str,
    content: bytes | str,
    source_path: str = "",
    metadata: dict[str, Any] | None = None,
    captured_at: int | None = None,
) -> int:
    """Store compressed source evidence and return its stable row id."""
    content_bytes = content.encode("utf-8") if isinstance(content, str) else content
    captured = captured_at if captured_at is not None else int(time.time() * 1000)
    digest = hashlib.sha256(content_bytes).hexdigest()
    compressed = zlib.compress(content_bytes)
    metadata_json = json.dumps(metadata or {}, sort_keys=True, separators=(",", ":"))

    conn.execute(
        """
        INSERT OR IGNORE INTO raw_evidence
            (project_id, session_id, source_type, source_path, captured_at,
             content_sha256, content_encoding, content_blob,
             original_size_bytes, stored_size_bytes, metadata)
        VALUES (?, ?, ?, ?, ?, ?, 'zlib', ?, ?, ?, ?)
        """,
        (
            project_id,
            session_id,
            source_type,
            source_path,
            captured,
            digest,
            compressed,
            len(content_bytes),
            len(compressed),
            metadata_json,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM raw_evidence WHERE project_id=? AND content_sha256=?",
        (project_id, digest),
    ).fetchone()
    return row["id"]


def load_evidence(conn: sqlite3.Connection, evidence_id: int) -> RawEvidence | None:
    row = conn.execute("SELECT * FROM raw_evidence WHERE id=?", (evidence_id,)).fetchone()
    if row is None:
        return None
    return _row_to_evidence(row)


def link_event_provenance(
    conn: sqlite3.Connection,
    event_id: int,
    evidence_id: int,
    extractor_version: str,
    matched_phrase: str | None = None,
    sentence_index: int | None = None,
    window_start: int | None = None,
    window_end: int | None = None,
    confidence: float | None = None,
    transformation_notes: str | None = None,
    created_at: int | None = None,
) -> None:
    """Link a derived event to its source evidence without overwriting detail."""
    conn.execute(
        """
        INSERT INTO event_provenance
            (event_id, raw_evidence_id, extractor_version, matched_phrase,
             sentence_index, window_start, window_end, confidence,
             transformation_notes, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id, raw_evidence_id) DO UPDATE SET
            extractor_version = excluded.extractor_version,
            matched_phrase = COALESCE(event_provenance.matched_phrase, excluded.matched_phrase),
            sentence_index = COALESCE(event_provenance.sentence_index, excluded.sentence_index),
            window_start = COALESCE(event_provenance.window_start, excluded.window_start),
            window_end = COALESCE(event_provenance.window_end, excluded.window_end),
            confidence = COALESCE(event_provenance.confidence, excluded.confidence),
            transformation_notes = COALESCE(
                event_provenance.transformation_notes,
                excluded.transformation_notes
            )
        """,
        (
            event_id,
            evidence_id,
            extractor_version,
            matched_phrase,
            sentence_index,
            window_start,
            window_end,
            confidence,
            transformation_notes,
            created_at if created_at is not None else int(time.time() * 1000),
        ),
    )


def get_evidence_summary(conn: sqlite3.Connection, project_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT COUNT(*) AS count,
               COALESCE(SUM(original_size_bytes), 0) AS original_size_bytes,
               COALESCE(SUM(stored_size_bytes), 0) AS stored_size_bytes
        FROM raw_evidence
        WHERE project_id = ?
        """,
        (project_id,),
    ).fetchone()
    original = int(row["original_size_bytes"])
    stored = int(row["stored_size_bytes"])
    ratio = (original / stored) if stored else 0.0
    return {
        "count": int(row["count"]),
        "original_size_bytes": original,
        "stored_size_bytes": stored,
        "average_compression_ratio": ratio,
    }


def _row_to_evidence(row: sqlite3.Row) -> RawEvidence:
    return RawEvidence(
        id=row["id"],
        project_id=row["project_id"],
        session_id=row["session_id"],
        source_type=row["source_type"],
        source_path=row["source_path"],
        captured_at=row["captured_at"],
        content_sha256=row["content_sha256"],
        content_encoding=row["content_encoding"],
        content=zlib.decompress(row["content_blob"]),
        original_size_bytes=row["original_size_bytes"],
        stored_size_bytes=row["stored_size_bytes"],
        metadata=json.loads(row["metadata"]),
    )
