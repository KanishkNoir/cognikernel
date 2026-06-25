"""Tests for memlora.delta.merge."""
from __future__ import annotations

import json
import sqlite3
from typing import Any
from unittest.mock import patch

import pytest

from memlora.delta.decay import DECAY_FACTOR
from memlora.delta.merge import execute_merge, merge_event
from memlora.storage.evidence import store_evidence
from memlora.storage.events import Event, MAX_EVENT_WEIGHT, WEIGHT_INCREMENT_ON_DEDUP, insert_event


# ── helpers ───────────────────────────────────────────────────────────────────

def make_event(**overrides: Any) -> Event:
    defaults: dict[str, Any] = {
        "project_id": "proj1",
        "session_id": "sess1",
        "event_type": "DECISION",
        "payload": {"description": "Use SQLite for local storage"},
        "content_hash": "hash_a",
        "weight": 1.0,
    }
    defaults.update(overrides)
    return Event(**defaults)


def get_row(conn: sqlite3.Connection, content_hash: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM events WHERE content_hash = ?", (content_hash,)
    ).fetchone()


# ── merge_event ───────────────────────────────────────────────────────────────

class TestMergeEvent:
    def test_insert_returns_inserted_and_id(self, conn: sqlite3.Connection) -> None:
        outcome, row_id = merge_event(conn, make_event())
        assert outcome == "inserted"
        assert isinstance(row_id, int) and row_id > 0

    def test_dedup_returns_updated_and_same_id(self, conn: sqlite3.Connection) -> None:
        e = make_event()
        outcome1, row_id1 = merge_event(conn, e)
        outcome2, row_id2 = merge_event(conn, e)
        assert outcome1 == "inserted"
        assert outcome2 == "updated"
        assert row_id1 == row_id2

    def test_insert_persists_to_db(self, conn: sqlite3.Connection) -> None:
        merge_event(conn, make_event())
        row = get_row(conn, "hash_a")
        assert row is not None
        assert row["event_type"] == "DECISION"

    def test_dedup_cross_session_increments_mention_count(self, conn: sqlite3.Connection) -> None:
        """F9: only cross-session restatements bump — a genuine recurrence signal."""
        merge_event(conn, make_event(session_id="sess1"))
        merge_event(conn, make_event(session_id="sess2"))
        row = get_row(conn, "hash_a")
        assert row["mention_count"] == 2

    def test_dedup_same_session_does_not_bump(self, conn: sqlite3.Connection) -> None:
        """F9: in-session duplicates are overlap-window re-extractions under delta
        ingest (I2) — bumping them inflated boundary events over real decisions."""
        e = make_event(session_id="sess1", weight=1.0)
        merge_event(conn, e)
        merge_event(conn, e)
        merge_event(conn, e)
        row = get_row(conn, "hash_a")
        assert row["mention_count"] == 1          # no in-session inflation
        assert row["weight"] == pytest.approx(1.0)

    def test_dedup_cross_session_increments_weight(self, conn: sqlite3.Connection) -> None:
        merge_event(conn, make_event(session_id="sess1", weight=1.0))
        merge_event(conn, make_event(session_id="sess2", weight=1.0))
        row = get_row(conn, "hash_a")
        assert row["weight"] == pytest.approx(1.0 + WEIGHT_INCREMENT_ON_DEDUP)

    def test_dedup_weight_capped_at_max(self, conn: sqlite3.Connection) -> None:
        merge_event(conn, make_event(session_id="s0", weight=MAX_EVENT_WEIGHT))
        for i in range(10):
            merge_event(conn, make_event(session_id=f"s{i+1}", weight=MAX_EVENT_WEIGHT))
        row = get_row(conn, "hash_a")
        assert row["weight"] <= MAX_EVENT_WEIGHT

    def test_commits_after_insert(self, conn: sqlite3.Connection) -> None:
        merge_event(conn, make_event())
        # If no commit, a fresh connection would see nothing; here we use same conn
        row = get_row(conn, "hash_a")
        assert row is not None

    def test_dedup_resurrects_archived_event(self, conn: sqlite3.Connection) -> None:
        """A fresh mention of an archived event must clear archived (I7c).

        Decay (or a manual archive) marks staleness; re-statement is direct
        evidence of renewed relevance. Without resurrection, the re-stated fact
        lands invisibly on the archived row and never reaches block or recall.
        """
        e = make_event()
        merge_event(conn, e)
        conn.execute("UPDATE events SET archived = 1 WHERE content_hash = 'hash_a'")
        conn.commit()

        outcome, _ = merge_event(conn, e)  # re-mention (same session)
        assert outcome == "updated"
        row = get_row(conn, "hash_a")
        assert row["archived"] == 0   # resurrected even in-session (F9 keeps this)
        assert row["mention_count"] == 1  # but no in-session weight/mention bump

    def test_dedup_resurrects_archived_event_cross_session(self, conn: sqlite3.Connection) -> None:
        merge_event(conn, make_event(session_id="sess1"))
        conn.execute("UPDATE events SET archived = 1 WHERE content_hash = 'hash_a'")
        conn.commit()
        merge_event(conn, make_event(session_id="sess2"))
        row = get_row(conn, "hash_a")
        assert row["archived"] == 0
        assert row["mention_count"] == 2  # cross-session restatement credits


# ── execute_merge — empty ─────────────────────────────────────────────────────

class TestExecuteMergeEmpty:
    def test_empty_candidates_returns_zero_stats(self, conn: sqlite3.Connection) -> None:
        stats = execute_merge(conn, "sess1", [])
        assert stats == {
            "inserted": 0,
            "updated": 0,
            "superseded": 0,
            "cascaded": 0,
            "archived": 0,
        }

    def test_empty_candidates_no_db_writes(self, conn: sqlite3.Connection) -> None:
        execute_merge(conn, "sess1", [])
        row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
        assert row[0] == 0


# ── execute_merge — insert stats ──────────────────────────────────────────────

class TestExecuteMergeStats:
    def test_two_new_events_inserted(self, conn: sqlite3.Connection) -> None:
        candidates = [
            make_event(content_hash="h1"),
            make_event(content_hash="h2", payload={"description": "Use Redis for cache"}),
        ]
        stats = execute_merge(conn, "sess1", candidates)
        assert stats["inserted"] == 2
        assert stats["updated"] == 0

    def test_dedup_counted_as_updated(self, conn: sqlite3.Connection) -> None:
        e = make_event(content_hash="h1")
        insert_event(conn, e)  # pre-seed
        stats = execute_merge(conn, "sess1", [e])
        assert stats["inserted"] == 0
        assert stats["updated"] == 1

    def test_mixed_insert_and_update(self, conn: sqlite3.Connection) -> None:
        e_existing = make_event(content_hash="h1")
        insert_event(conn, e_existing)
        candidates = [
            e_existing,
            make_event(content_hash="h2", payload={"description": "New decision"}),
        ]
        stats = execute_merge(conn, "sess1", candidates)
        assert stats["inserted"] == 1
        assert stats["updated"] == 1

    def test_stats_dict_has_all_keys(self, conn: sqlite3.Connection) -> None:
        stats = execute_merge(conn, "sess1", [make_event()])
        for key in ("inserted", "updated", "superseded", "cascaded", "archived"):
            assert key in stats


# ── execute_merge — supersession ──────────────────────────────────────────────

class TestExecuteMergeSupersession:
    def test_supersession_counted(self, conn: sqlite3.Connection) -> None:
        old = make_event(
            event_type="DECISION",
            content_hash="old_hash",
            payload={"description": "Use SQLite for persistent local storage"},
        )
        insert_event(conn, old)
        new = make_event(
            event_type="DECISION",
            content_hash="new_hash",
            payload={"description": "Use SQLite for persistent local data storage"},
        )
        stats = execute_merge(conn, "sess1", [new])
        assert stats["superseded"] >= 1

    def test_superseded_event_marked_in_db(self, conn: sqlite3.Connection) -> None:
        old = make_event(
            event_type="DECISION",
            content_hash="old_hash",
            payload={"description": "Use SQLite for persistent local storage"},
        )
        old_id = insert_event(conn, old)
        new = make_event(
            event_type="DECISION",
            content_hash="new_hash",
            payload={"description": "Use SQLite for persistent local data storage"},
        )
        execute_merge(conn, "sess1", [new])
        row = conn.execute("SELECT superseded_by FROM events WHERE id = ?", (old_id,)).fetchone()
        assert row["superseded_by"] is not None


# ── execute_merge — baseline gates (embeddings off) ───────────────────────────

class TestExecuteMergeBaselineGates:
    """With embed_events=False (default), supersession runs through the gated
    finder — the temporal/authority/provenance gates are the baseline, not the
    legacy ungated lexical path."""

    def test_cross_session_overlap_supersedes(self, conn: sqlite3.Connection) -> None:
        ev1 = store_evidence(conn, "proj1", "sess0", "transcript", b"session-0")
        ev2 = store_evidence(conn, "proj1", "sess1", "transcript", b"session-1")
        old = make_event(
            content_hash="old_hash", created_at=1000, evidence_id=ev1,
            payload={"description": "Use SQLite for persistent local storage"},
        )
        old_id = insert_event(conn, old)
        new = make_event(
            content_hash="new_hash", created_at=2000, evidence_id=ev2,
            payload={"description": "Use SQLite for persistent local data storage"},
        )
        execute_merge(conn, "sess1", [new])
        row = conn.execute("SELECT superseded_by FROM events WHERE id = ?", (old_id,)).fetchone()
        assert row["superseded_by"] is not None

    def test_same_evidence_overlap_not_superseded(self, conn: sqlite3.Connection) -> None:
        """Same transcript = restatement, not evolution — provenance gate holds."""
        ev = store_evidence(conn, "proj1", "sess1", "transcript", b"one-session")
        old = make_event(
            content_hash="old_hash", created_at=1000, evidence_id=ev,
            payload={"description": "Use SQLite for persistent local storage"},
        )
        old_id = insert_event(conn, old)
        new = make_event(
            content_hash="new_hash", created_at=2000, evidence_id=ev,
            payload={"description": "Use SQLite for persistent local data storage"},
        )
        execute_merge(conn, "sess1", [new])
        row = conn.execute("SELECT superseded_by FROM events WHERE id = ?", (old_id,)).fetchone()
        assert row["superseded_by"] is None

    def test_older_candidate_does_not_supersede_newer(self, conn: sqlite3.Connection) -> None:
        """Temporal gate: an incoming event never supersedes a newer existing one."""
        future = make_event(
            content_hash="future_hash", created_at=5000,
            payload={"description": "Use SQLite for persistent local storage"},
        )
        future_id = insert_event(conn, future)
        new = make_event(
            content_hash="now_hash", created_at=2000,
            payload={"description": "Use SQLite for persistent local data storage"},
        )
        execute_merge(conn, "sess1", [new])
        row = conn.execute("SELECT superseded_by FROM events WHERE id = ?", (future_id,)).fetchone()
        assert row["superseded_by"] is None


# ── execute_merge — cascade ───────────────────────────────────────────────────

class TestExecuteMergeCascade:
    def test_cascade_triggered_for_component_status(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO events
                (project_id, session_id, created_at, event_type,
                 payload, content_hash, weight, mention_count)
            VALUES ('proj1', 'sess0', 0, 'COMPONENT_STATUS',
                    '{"path":"src/router.py","status":"stable","dependencies":["src/api.py"]}',
                    'dep1', 0.8, 1)
            """
        )
        conn.commit()
        e = make_event(
            event_type="COMPONENT_STATUS",
            content_hash="new_status",
            payload={"path": "src/api.py", "status": "blocked"},
        )
        stats = execute_merge(conn, "sess1", [e])
        assert stats["cascaded"] >= 1

    def test_no_cascade_for_non_component_status(self, conn: sqlite3.Connection) -> None:
        candidates = [make_event(event_type="DECISION", content_hash="h1")]
        stats = execute_merge(conn, "sess1", candidates)
        assert stats["cascaded"] == 0

    def test_cascade_event_links_to_source_evidence(self, conn: sqlite3.Connection) -> None:
        evidence_id = store_evidence(conn, "proj1", "sess1", "transcript", b"source")
        conn.execute(
            """
            INSERT INTO events
                (project_id, session_id, created_at, event_type,
                 payload, content_hash, weight, mention_count)
            VALUES ('proj1', 'sess0', 0, 'COMPONENT_STATUS',
                    '{"path":"src/router.py","status":"stable","dependencies":["src/api.py"]}',
                    'dep1', 0.8, 1)
            """
        )
        conn.commit()
        e = make_event(
            event_type="COMPONENT_STATUS",
            content_hash="new_status_with_evidence",
            payload={"path": "src/api.py", "status": "blocked"},
            evidence_id=evidence_id,
        )

        execute_merge(conn, "sess1", [e])

        row = conn.execute(
            """
            SELECT ep.raw_evidence_id
            FROM events ev
            JOIN event_provenance ep ON ep.event_id = ev.id
            WHERE ev.content_hash != 'new_status_with_evidence'
              AND json_extract(ev.payload, '$.status') = 'needs_review'
            """
        ).fetchone()
        assert row is not None
        assert row["raw_evidence_id"] == evidence_id


# ── execute_merge — decay and archive ─────────────────────────────────────────

class TestExecuteMergeDecay:
    def test_archived_count_in_stats(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO events
                (project_id, session_id, created_at, event_type,
                 payload, content_hash, weight, mention_count)
            VALUES ('proj1', 'sess_old', 0, 'DECISION',
                    '{"description":"stale"}', 'stale_hash', 0.04, 1)
            """
        )
        conn.commit()
        stats = execute_merge(conn, "sess_new", [make_event(content_hash="new_h")])
        assert stats["archived"] >= 1

    def test_decay_not_applied_to_current_session(self, conn: sqlite3.Connection) -> None:
        e = make_event(content_hash="h1", weight=1.0, session_id="sess_current")
        execute_merge(conn, "sess_current", [e])
        row = conn.execute("SELECT weight FROM events WHERE content_hash='h1'").fetchone()
        assert row["weight"] == pytest.approx(1.0)


# ── execute_merge — replay idempotency (audit P1) ─────────────────────────────

class TestExecuteMergeReplayIdempotency:
    """A recovered job replays the SAME evidence slice (merge committed, but the
    worker was killed before the ingest cursor advanced). The merge must be a
    no-op on replay — no second ×0.92 decay tick, no extra mention_count bump.
    The guard keys on event_provenance(raw_evidence_id), which is written inside
    the merge transaction, so it witnesses only a fully-committed prior merge.
    """

    def test_replay_same_evidence_is_skipped(self, conn: sqlite3.Connection) -> None:
        ev = store_evidence(conn, "proj1", "sess1", "transcript", b"slice-1")
        first = execute_merge(conn, "sess1", [make_event(content_hash="hx", evidence_id=ev)])
        assert first["inserted"] == 1
        second = execute_merge(conn, "sess1", [make_event(content_hash="hx", evidence_id=ev)])
        assert second == {
            "inserted": 0, "updated": 0, "superseded": 0, "cascaded": 0, "archived": 0,
        }

    def test_replay_does_not_rebump_mention_count(self, conn: sqlite3.Connection) -> None:
        ev = store_evidence(conn, "proj1", "sess1", "transcript", b"slice-2")
        # Pre-existing cross-session original so the merge folds an echo (+1 mention).
        conn.execute(
            "INSERT INTO events (project_id, session_id, created_at, event_type, "
            "payload, content_hash, weight, mention_count) VALUES "
            "('proj1','sess0',0,'DECISION','{\"description\":\"Use SQLite for local storage\"}',"
            "'hash_a',1.0,1)"
        )
        conn.commit()
        execute_merge(conn, "sess1", [make_event(session_id="sess1", content_hash="hash_a", evidence_id=ev)])
        assert get_row(conn, "hash_a")["mention_count"] == 2
        # Replay the same evidence slice — must not climb to 3.
        execute_merge(conn, "sess1", [make_event(session_id="sess1", content_hash="hash_a", evidence_id=ev)])
        assert get_row(conn, "hash_a")["mention_count"] == 2

    def test_replay_does_not_reapply_decay(self, conn: sqlite3.Connection) -> None:
        ev = store_evidence(conn, "proj1", "sess_new", "transcript", b"slice-3")
        conn.execute(
            "INSERT INTO events (project_id, session_id, created_at, event_type, "
            "payload, content_hash, weight, mention_count) VALUES "
            "('proj1','sess_old',0,'DECISION','{\"description\":\"stale\"}','stale_hash',0.5,1)"
        )
        conn.commit()
        execute_merge(conn, "sess_new", [make_event(content_hash="newh", session_id="sess_new", evidence_id=ev)])
        w1 = conn.execute("SELECT weight FROM events WHERE content_hash='stale_hash'").fetchone()["weight"]
        assert w1 == pytest.approx(0.5 * DECAY_FACTOR)
        # Replay — the ×0.92 decay tick must NOT fire a second time.
        execute_merge(conn, "sess_new", [make_event(content_hash="newh", session_id="sess_new", evidence_id=ev)])
        w2 = conn.execute("SELECT weight FROM events WHERE content_hash='stale_hash'").fetchone()["weight"]
        assert w2 == pytest.approx(w1)

    def test_distinct_evidence_still_merges(self, conn: sqlite3.Connection) -> None:
        """The guard keys on evidence id, not content: a genuinely new evidence
        slice merges normally — it must not be mistaken for a replay."""
        ev1 = store_evidence(conn, "proj1", "sess1", "transcript", b"ev-a")
        ev2 = store_evidence(conn, "proj1", "sess1", "transcript", b"ev-b")
        # Distinct descriptions so the R3 echo guard doesn't fold the second event.
        execute_merge(conn, "sess1", [make_event(
            content_hash="h1", payload={"description": "Use Redis for the cache layer"}, evidence_id=ev1)])
        stats = execute_merge(conn, "sess1", [make_event(
            content_hash="h2", payload={"description": "Use Postgres for durable storage"}, evidence_id=ev2)])
        assert stats["inserted"] == 1

    def test_no_evidence_id_is_not_guarded(self, conn: sqlite3.Connection) -> None:
        """Merges without an evidence id (no durable witness) keep prior behavior:
        the guard is inert, so back-to-back merges still process."""
        execute_merge(conn, "sess1", [make_event(
            content_hash="ng1", payload={"description": "Adopt FastAPI for the gateway"})])
        stats = execute_merge(conn, "sess1", [make_event(
            content_hash="ng2", payload={"description": "Adopt Uvicorn as the ASGI server"})])
        assert stats["inserted"] == 1


# ── execute_merge — transaction and rollback ──────────────────────────────────

class TestExecuteMergeTransaction:
    def test_exception_writes_to_extraction_failures(
        self, conn: sqlite3.Connection
    ) -> None:
        # Ensure extraction_failures table exists
        # (it's part of the migration, so conn fixture already has it)
        bad_event = make_event(content_hash="bad")
        with patch(
            "memlora.delta.merge._insert_or_update",
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                execute_merge(conn, "sess1", [bad_event])
        row = conn.execute("SELECT * FROM extraction_failures").fetchone()
        assert row is not None
        assert row["stage"] == "delta.merge"
        assert "boom" in row["error_message"]

    def test_exception_rolls_back_events(self, conn: sqlite3.Connection) -> None:
        bad_event = make_event(content_hash="bad")
        with patch(
            "memlora.delta.merge._insert_or_update",
            side_effect=RuntimeError("rollback test"),
        ):
            with pytest.raises(RuntimeError):
                execute_merge(conn, "sess1", [bad_event])
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        assert count == 0

    def test_projection_invalidated_after_merge(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            INSERT INTO state_projections
                (project_id, built_at, event_id_high_water,
                 hard_constraints, ranked_decisions, component_map,
                 graveyard, active_threads, summary)
            VALUES ('proj1', 0, 100, '[]', '[]', '{}', '[]', '[]', '')
            """
        )
        conn.commit()
        execute_merge(conn, "sess1", [make_event()])
        row = conn.execute(
            "SELECT event_id_high_water FROM state_projections WHERE project_id='proj1'"
        ).fetchone()
        assert row["event_id_high_water"] == -1

    def test_event_id_set_on_candidate_after_insert(self, conn: sqlite3.Connection) -> None:
        e = make_event(content_hash="h1")
        assert e.id is None
        execute_merge(conn, "sess1", [e])
        assert e.id is not None and e.id > 0


# ── MAX_EVENT_WEIGHT ───────────────────────────────────────────────────────────────

class TestMaxWeightConstant:
    def test_max_weight_value(self) -> None:
        assert MAX_EVENT_WEIGHT == pytest.approx(5.0)

