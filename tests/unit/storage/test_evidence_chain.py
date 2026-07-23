"""Tests for chained delta evidence (Sprint I / I3).

AUDIT INVARIANT (from session.py §6.5(c) + I3 spec):
  load_full_transcript(evidence_id) must produce byte-exact reconstruction of
  the original JSONL, regardless of whether evidence was stored as:
    (a) a single full-content blob (legacy / compaction fallback), or
    (b) a chain of delta chunks (root + N deltas).

  Given the same reconstructed bytes + extractor version, the extraction
  pipeline produces the same (event_type, content_hash, payload) set.
"""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from cognikernel.storage.cursors import (
    IngestCursor,
    compute_anchor,
    slice_storage_delta,
)
from cognikernel.storage.evidence import load_full_transcript, store_evidence
from cognikernel.storage.migrations import run_migrations


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


def _make_jsonl(n: int) -> str:
    return "\n".join(f'{{"seq":{i},"type":"user","text":"turn {i}"}}' for i in range(n)) + "\n"


class TestChainReconstruction:
    def test_root_only_roundtrips(self):
        conn = _db()
        content = _make_jsonl(20).encode("utf-8")
        eid = store_evidence(conn, "p", "s", "transcript", content)
        assert load_full_transcript(conn, eid) == content

    def test_two_chunk_chain_reconstructs_full(self):
        conn = _db()
        full = _make_jsonl(40)
        lines = [ln for ln in full.splitlines() if ln.strip()]

        # Simulate two firings: root (0..20) then delta (20..40).
        root_bytes = ("\n".join(lines[:20]) + "\n").encode("utf-8")
        delta_bytes = ("\n".join(lines[20:]) + "\n").encode("utf-8")

        root_id = store_evidence(conn, "p", "s", "transcript", root_bytes)
        delta_id = store_evidence(conn, "p", "s", "transcript", delta_bytes,
                                  prev_evidence_id=root_id)

        reconstructed = load_full_transcript(conn, delta_id)
        expected = full.encode("utf-8")
        assert reconstructed == expected

    def test_three_chunk_chain_reconstructs_full(self):
        conn = _db()
        full = _make_jsonl(60)
        lines = [ln for ln in full.splitlines() if ln.strip()]

        chunk0 = ("\n".join(lines[:20]) + "\n").encode("utf-8")
        chunk1 = ("\n".join(lines[20:40]) + "\n").encode("utf-8")
        chunk2 = ("\n".join(lines[40:]) + "\n").encode("utf-8")

        e0 = store_evidence(conn, "p", "s", "transcript", chunk0)
        e1 = store_evidence(conn, "p", "s", "transcript", chunk1, prev_evidence_id=e0)
        e2 = store_evidence(conn, "p", "s", "transcript", chunk2, prev_evidence_id=e1)

        assert load_full_transcript(conn, e2) == full.encode("utf-8")

    def test_root_load_equals_chain_load(self):
        """A single full-content store must produce the same bytes as a two-chunk chain."""
        conn = _db()
        full = _make_jsonl(30)
        lines = [ln for ln in full.splitlines() if ln.strip()]

        # Full blob.
        full_bytes = full.encode("utf-8")
        e_full = store_evidence(conn, "p", "s1", "transcript", full_bytes)

        # Two-chunk chain producing identical content.
        root = ("\n".join(lines[:15]) + "\n").encode("utf-8")
        delta = ("\n".join(lines[15:]) + "\n").encode("utf-8")
        e_root = store_evidence(conn, "p", "s2", "transcript", root)
        e_delta = store_evidence(conn, "p", "s2", "transcript", delta, prev_evidence_id=e_root)

        assert load_full_transcript(conn, e_full) == load_full_transcript(conn, e_delta)

    def test_cycle_detection_raises(self):
        """A malformed chain with a cycle must raise ValueError."""
        conn = _db()
        # Manually insert a self-referencing row to simulate corruption.
        conn.execute(
            "INSERT INTO raw_evidence (project_id, session_id, source_type, source_path, "
            "captured_at, content_sha256, content_encoding, content_blob, "
            "original_size_bytes, stored_size_bytes, metadata, prev_evidence_id) "
            "VALUES ('p','s','transcript','',0,'deadbeef','zlib',x'789c6300000000ff',1,1,'{}',99999)"
        )
        bad_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("UPDATE raw_evidence SET prev_evidence_id=? WHERE id=?", (bad_id, bad_id))
        conn.commit()
        with pytest.raises(ValueError, match="Cycle"):
            load_full_transcript(conn, bad_id)

    def test_broken_chain_raises(self):
        conn = _db()
        with pytest.raises(ValueError, match="not found"):
            load_full_transcript(conn, 9999)


class TestSliceStorageDelta:
    def test_no_cursor_returns_full_and_not_delta(self):
        jsonl = _make_jsonl(20)
        content, is_delta, has_new = slice_storage_delta(jsonl, cursor=None)
        assert not is_delta
        assert has_new
        assert content == jsonl.encode("utf-8")

    def test_delta_returns_only_new_lines(self):
        full = _make_jsonl(40)
        lines = [ln for ln in full.splitlines() if ln.strip()]
        anchor = compute_anchor(lines, 20)
        cursor = IngestCursor("p", "s", last_line_count=20, anchor_sha256=anchor,
                               updated_at=0, last_evidence_id=1)
        content, is_delta, has_new = slice_storage_delta(full, cursor)
        assert is_delta
        assert has_new
        delta_lines = [ln for ln in content.decode("utf-8").splitlines() if ln.strip()]
        assert len(delta_lines) == 20
        assert all(f'"seq":{i}' in delta_lines[i - 20] for i in range(20, 40))

    def test_compaction_fallback_not_delta(self):
        full = _make_jsonl(40)
        cursor = IngestCursor("p", "s", last_line_count=20, anchor_sha256="bad",
                               updated_at=0, last_evidence_id=1)
        _, is_delta, has_new = slice_storage_delta(full, cursor)
        assert not is_delta
        assert has_new

    def test_no_new_lines_signals_skip(self):
        """Same content as the cursor high-water mark — caller must skip storage."""
        full = _make_jsonl(30)
        lines = [ln for ln in full.splitlines() if ln.strip()]
        anchor = compute_anchor(lines, 30)
        cursor = IngestCursor("p", "s", last_line_count=30, anchor_sha256=anchor,
                               updated_at=0, last_evidence_id=1)
        content, is_delta, has_new = slice_storage_delta(full, cursor)
        assert not has_new
        assert not is_delta
        assert content == b""

    def test_delta_plus_root_concat_equals_full(self):
        """The storage invariant: root_bytes + delta_bytes = full transcript bytes."""
        full = _make_jsonl(50)
        lines = [ln for ln in full.splitlines() if ln.strip()]

        anchor = compute_anchor(lines, 25)
        cursor = IngestCursor("p", "s", last_line_count=25, anchor_sha256=anchor,
                               updated_at=0, last_evidence_id=None)

        root_bytes = ("\n".join(lines[:25]) + "\n").encode("utf-8")
        delta_bytes, is_delta, has_new = slice_storage_delta(full, cursor)
        assert is_delta and has_new
        assert root_bytes + delta_bytes == full.encode("utf-8")

    def test_raw_root_without_trailing_newline_joins_safely(self):
        """Gamma post-mortem F6: a root stored raw (no trailing newline) must not
        corrupt the boundary line when concatenated with a delta chunk."""
        conn = _db()
        full = _make_jsonl(20)
        lines = [ln for ln in full.splitlines() if ln.strip()]

        root_raw = "\n".join(lines[:10])          # NO trailing newline (raw on-disk form)
        delta = ("\n".join(lines[10:]) + "\n")

        e0 = store_evidence(conn, "p", "s", "transcript", root_raw.encode("utf-8"))
        e1 = store_evidence(conn, "p", "s", "transcript", delta.encode("utf-8"),
                            prev_evidence_id=e0)

        reconstructed = load_full_transcript(conn, e1).decode("utf-8")
        rec_lines = [ln for ln in reconstructed.splitlines() if ln.strip()]
        assert len(rec_lines) == 20              # boundary line NOT merged into one
        assert rec_lines[9] == lines[9]
        assert rec_lines[10] == lines[10]
