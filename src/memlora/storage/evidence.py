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
    prev_evidence_id: int | None = None


def store_evidence(
    conn: sqlite3.Connection,
    project_id: str,
    session_id: str,
    source_type: str,
    content: bytes | str,
    source_path: str = "",
    metadata: dict[str, Any] | None = None,
    captured_at: int | None = None,
    prev_evidence_id: int | None = None,
) -> int:
    """Store compressed source evidence and return its stable row id.

    `prev_evidence_id` chains this chunk to the previous one for delta evidence
    (I3). NULL = chain root (full content). Reconstruction: follow the chain
    root→leaf, concatenate all content_blob bytes to recover the full transcript.
    """
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
             original_size_bytes, stored_size_bytes, metadata, prev_evidence_id)
        VALUES (?, ?, ?, ?, ?, ?, 'zlib', ?, ?, ?, ?, ?)
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
            prev_evidence_id,
        ),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM raw_evidence WHERE project_id=? AND content_sha256=?",
        (project_id, digest),
    ).fetchone()
    return row["id"]


def load_full_transcript(conn: sqlite3.Connection, evidence_id: int) -> bytes:
    """Reconstruct the full transcript by following the prev_evidence_id chain.

    For a chain root (prev_evidence_id IS NULL), returns the stored content
    directly. For delta chunks, walks the chain root→leaf and concatenates
    all content_blob bytes in order, producing a byte-exact reconstruction of
    the original growing JSONL transcript.

    Raises ValueError if any chunk in the chain is missing. This preserves the
    audit invariant: same evidence_id + extractor version => same extracted
    events regardless of whether the evidence was stored as a full blob or a
    chain of deltas.
    """
    chunks: list[bytes] = []
    visited: set[int] = set()
    current_id: int | None = evidence_id

    # Collect the chain in reverse (leaf → root), then reverse.
    chain_ids: list[int] = []
    while current_id is not None:
        if current_id in visited:
            raise ValueError(f"Cycle detected in evidence chain at id={current_id}")
        visited.add(current_id)
        row = conn.execute(
            "SELECT id, content_blob, content_encoding, prev_evidence_id FROM raw_evidence WHERE id=?",
            (current_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Evidence chain broken: id={current_id} not found")
        chain_ids.append(current_id)
        current_id = row["prev_evidence_id"]

    chain_ids.reverse()  # root first

    for cid in chain_ids:
        row = conn.execute(
            "SELECT content_blob, content_encoding FROM raw_evidence WHERE id=?",
            (cid,),
        ).fetchone()
        blob = row["content_blob"]
        if row["content_encoding"] == "zlib":
            blob = zlib.decompress(blob)
        # Join guard: a chain root may have been stored raw (no trailing newline,
        # e.g. the original on-disk JSONL). Delta chunks are line-complete, so a
        # missing newline at a chunk boundary would corrupt the boundary line on
        # concatenation. Insert one only when needed — chunks that already end
        # with a newline are unaffected, keeping reconstruction deterministic.
        if chunks and not chunks[-1].endswith(b"\n") and blob:
            chunks.append(b"\n")
        chunks.append(blob)

    return b"".join(chunks)


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
        prev_evidence_id=row["prev_evidence_id"] if "prev_evidence_id" in row.keys() else None,
    )
