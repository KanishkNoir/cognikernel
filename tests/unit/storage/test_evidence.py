from __future__ import annotations

import sqlite3

from memlora.storage.evidence import (
    get_evidence_summary,
    link_event_provenance,
    load_evidence,
    store_evidence,
)
from memlora.storage.events import Event, get_event_by_id, insert_event


def test_store_evidence_compresses_and_round_trips(conn: sqlite3.Connection) -> None:
    payload = ("Assistant: We decided to use SQLite.\n" * 20).encode("utf-8")

    evidence_id = store_evidence(
        conn,
        project_id="proj1",
        session_id="sess1",
        source_type="transcript",
        content=payload,
        source_path="/tmp/sess1.txt",
    )

    row = conn.execute("SELECT * FROM raw_evidence WHERE id=?", (evidence_id,)).fetchone()
    assert row["content_encoding"] == "zlib"
    assert row["stored_size_bytes"] < row["original_size_bytes"]

    loaded = load_evidence(conn, evidence_id)
    assert loaded is not None
    assert loaded.content == payload
    assert loaded.source_path == "/tmp/sess1.txt"


def test_store_evidence_dedupes_by_project_and_hash(conn: sqlite3.Connection) -> None:
    payload = b"same transcript bytes"

    first = store_evidence(conn, "proj1", "sess1", "transcript", payload)
    second = store_evidence(conn, "proj1", "sess1", "transcript", payload)
    other_project = store_evidence(conn, "proj2", "sess1", "transcript", payload)

    assert first == second
    assert other_project != first
    count = conn.execute("SELECT COUNT(*) FROM raw_evidence").fetchone()[0]
    assert count == 2


def test_insert_event_records_evidence_id(conn: sqlite3.Connection) -> None:
    evidence_id = store_evidence(conn, "proj1", "sess1", "transcript", b"content")
    event = Event(
        project_id="proj1",
        session_id="sess1",
        event_type="DECISION",
        payload={"description": "Use SQLite"},
        content_hash="hash-with-evidence",
        evidence_id=evidence_id,
    )

    event_id = insert_event(conn, event)

    stored = get_event_by_id(conn, event_id)
    assert stored is not None
    assert stored.evidence_id == evidence_id


def test_link_event_provenance_is_idempotent(conn: sqlite3.Connection) -> None:
    evidence_id = store_evidence(conn, "proj1", "sess1", "transcript", b"content")
    event_id = insert_event(
        conn,
        Event(
            project_id="proj1",
            session_id="sess1",
            event_type="DECISION",
            payload={"description": "Use SQLite"},
            content_hash="prov-hash",
            evidence_id=evidence_id,
        ),
    )

    link_event_provenance(
        conn,
        event_id=event_id,
        evidence_id=evidence_id,
        extractor_version="test-v1",
        matched_phrase="decided",
        sentence_index=3,
        window_start=2,
        window_end=4,
        confidence=0.9,
        transformation_notes="unit test",
    )
    link_event_provenance(
        conn,
        event_id=event_id,
        evidence_id=evidence_id,
        extractor_version="test-v1",
    )

    rows = conn.execute("SELECT * FROM event_provenance").fetchall()
    assert len(rows) == 1
    assert rows[0]["matched_phrase"] == "decided"


def test_get_evidence_summary_reports_ratio(conn: sqlite3.Connection) -> None:
    store_evidence(conn, "proj1", "sess1", "transcript", b"x" * 200)
    store_evidence(conn, "proj1", "sess2", "git_diff", b"diff" * 40)

    summary = get_evidence_summary(conn, "proj1")

    assert summary["count"] == 2
    assert summary["original_size_bytes"] > summary["stored_size_bytes"]
    assert summary["average_compression_ratio"] > 1.0
