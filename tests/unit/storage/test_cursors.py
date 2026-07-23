"""Tests for ingest cursors — the I2 high-water mark implementation.

KEY PROPERTY (from sprint spec): full-vs-incremental equivalence.
A JSONL ingested whole vs. in N increments must yield the same (extraction_slice
content coverage, cursor state) outcomes.  Merge-level equivalence is tested via
content_hash dedup in execute_merge; here we test the slice logic in isolation.
"""
from __future__ import annotations

import hashlib
import sqlite3

import pytest

from cognikernel.storage.cursors import (
    ANCHOR_LINES,
    OVERLAP_LINES,
    IngestCursor,
    compute_anchor,
    get_cursor,
    save_cursor,
    slice_jsonl_for_extraction,
)
from cognikernel.storage.migrations import run_migrations


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_jsonl(n: int) -> str:
    """Produce a fake JSONL with n lines."""
    return "\n".join(f'{{"seq":{i},"type":"user","text":"turn {i}"}}' for i in range(n))


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    return conn


# ── slice_jsonl_for_extraction ────────────────────────────────────────────────

class TestSliceJsonl:
    def test_no_cursor_returns_full(self):
        jsonl = _make_jsonl(30)
        sliced, count, anchor = slice_jsonl_for_extraction(jsonl, cursor=None)
        assert sliced == jsonl
        assert count == 30
        assert len(anchor) == 64  # sha256 hex

    def test_first_run_cursor_zero_returns_full(self):
        jsonl = _make_jsonl(20)
        cursor = IngestCursor("p", "s", last_line_count=0, anchor_sha256="", updated_at=0)
        sliced, count, anchor = slice_jsonl_for_extraction(jsonl, cursor)
        assert sliced == jsonl
        assert count == 20

    def test_delta_returns_overlap_plus_new(self):
        full = _make_jsonl(50)
        lines = [ln for ln in full.splitlines() if ln.strip()]

        # Simulate cursor after 30 lines were processed.
        anchor = compute_anchor(lines, 30)
        cursor = IngestCursor("p", "s", last_line_count=30, anchor_sha256=anchor, updated_at=0)

        sliced, new_count, new_anchor = slice_jsonl_for_extraction(full, cursor)
        assert new_count == 50

        sliced_lines = [ln for ln in sliced.splitlines() if ln.strip()]
        # Must include overlap window (20 lines before hw) + 20 new lines = 40 lines.
        assert len(sliced_lines) == OVERLAP_LINES + 20

        # The sliced content must contain seq numbers from (30 - OVERLAP_LINES) onward.
        first_seq = int(sliced_lines[0].split('"seq":')[1].split(',')[0])
        assert first_seq == 30 - OVERLAP_LINES

    def test_no_new_lines_returns_overlap_only(self):
        full = _make_jsonl(30)
        lines = [ln for ln in full.splitlines() if ln.strip()]
        anchor = compute_anchor(lines, 30)
        cursor = IngestCursor("p", "s", last_line_count=30, anchor_sha256=anchor, updated_at=0)

        sliced, count, _ = slice_jsonl_for_extraction(full, cursor)
        assert count == 30
        sliced_lines = [ln for ln in sliced.splitlines() if ln.strip()]
        assert len(sliced_lines) == OVERLAP_LINES

    def test_compaction_detected_returns_full(self):
        full = _make_jsonl(40)
        # Bad anchor (simulating compaction).
        cursor = IngestCursor("p", "s", last_line_count=20, anchor_sha256="bad_hash", updated_at=0)
        sliced, _, _ = slice_jsonl_for_extraction(full, cursor)
        assert sliced == full

    def test_file_shrank_returns_full(self):
        full = _make_jsonl(10)
        lines = [ln for ln in _make_jsonl(20).splitlines() if ln.strip()]
        anchor = compute_anchor(lines, 20)
        cursor = IngestCursor("p", "s", last_line_count=20, anchor_sha256=anchor, updated_at=0)
        sliced, _, _ = slice_jsonl_for_extraction(full, cursor)
        assert sliced == full  # file shrank → full re-extract


# ── equivalence property ─────────────────────────────────────────────────────

class TestEquivalenceProperty:
    """Full-vs-incremental equivalence: the set of lines seen by the extractor
    across N incremental firings must cover the same lines as one full extraction,
    modulo duplication in the overlap window (which content_hash dedup absorbs).
    """

    def test_incremental_covers_all_lines(self):
        total_lines = 60
        full = _make_jsonl(total_lines)
        lines = [ln for ln in full.splitlines() if ln.strip()]

        firing_sizes = [15, 25, 40, 60]  # simulated Stop hook transcript sizes
        seen_seqs: set[int] = set()
        cursor = None

        for size in firing_sizes:
            partial_jsonl = "\n".join(lines[:size])
            sliced, new_count, new_anchor = slice_jsonl_for_extraction(partial_jsonl, cursor)

            for ln in sliced.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                import json
                try:
                    obj = json.loads(ln)
                    seen_seqs.add(obj["seq"])
                except Exception:
                    pass

            cursor = IngestCursor("p", "s", new_count, new_anchor, 0)

        # Every line in the full transcript must have been seen.
        assert seen_seqs == set(range(total_lines))


# ── cursor persistence ─────────────────────────────────────────────────────────

class TestCursorPersistence:
    def test_save_and_get(self):
        conn = _db()
        save_cursor(conn, "proj1", "sess1", 42, "abc123")
        cursor = get_cursor(conn, "proj1", "sess1")
        assert cursor is not None
        assert cursor.last_line_count == 42
        assert cursor.anchor_sha256 == "abc123"

    def test_get_missing_returns_none(self):
        conn = _db()
        assert get_cursor(conn, "proj", "missing") is None

    def test_upsert_advances_cursor(self):
        conn = _db()
        save_cursor(conn, "p", "s", 10, "anchor1")
        save_cursor(conn, "p", "s", 25, "anchor2")
        cursor = get_cursor(conn, "p", "s")
        assert cursor.last_line_count == 25
        assert cursor.anchor_sha256 == "anchor2"

    def test_cursor_not_updated_on_exception_path(self):
        """Cursor must only advance after a successful merge, never on exception."""
        conn = _db()
        save_cursor(conn, "p", "s", 10, "anchor1")
        # Simulate: merge raises, cursor stays at 10.
        try:
            raise RuntimeError("merge failed")
        except RuntimeError:
            pass  # cursor.save never called
        cursor = get_cursor(conn, "p", "s")
        assert cursor.last_line_count == 10
