"""Tests for the A-5 MCP tools: get_unprocessed_evidence + store_extracted_events.

These tools live in `memlora.integration.mcp_server` and are registered with
FastMCP via `@_mcp.tool`. The tests call the underlying functions directly
(FastMCP's decoration leaves the original callable accessible) so they don't
need to spin up an MCP server transport.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from memlora.config import Config
from memlora.extraction.llm_enrich import LLM_EXTRACTOR_VERSION
from memlora.integration.mcp_server import (
    get_unprocessed_evidence,
    store_extracted_events,
)
from memlora.integration.session import init_project
from memlora.storage import enrichment_jobs as ej
from memlora.storage.connection import get_connection, get_db_path, hash_project_path
from memlora.storage.evidence import store_evidence


@pytest.fixture
def project(tmp_path: Path, monkeypatch):
    """Create a fresh project DB rooted at tmp_path and return its absolute path."""
    memlora_dir = tmp_path / "memlora_data"
    monkeypatch.setenv("MEMLORA_DIR", str(memlora_dir))

    project_path = tmp_path / "proj"
    project_path.mkdir()
    cfg = Config(memlora_dir=memlora_dir)
    init_project(project_path, config=cfg)
    return str(project_path)


def _seed_evidence(
    project_path: str,
    *,
    session_id: str = "sess-1",
    content: bytes | None = None,
) -> int:
    """Store a transcript-like blob in raw_evidence and return its id.

    `raw_evidence` has UNIQUE (project_id, content_sha256), so distinct
    seedings must use distinct content. Callers seeding multiple rows in a
    single test should pass `content` explicitly.
    """
    cfg = Config.load()
    project_id = hash_project_path(project_path)
    db_path = get_db_path(cfg, project_id)
    if content is None:
        content = (
            b"User:\nUse argon2id.\n\nAssistant:\nUnderstood for "
            + session_id.encode()
            + b".\n"
        )
    with get_connection(db_path) as conn:
        return store_evidence(
            conn, project_id, session_id, "transcript", content,
        )


# ── get_unprocessed_evidence ─────────────────────────────────────────────────


class TestGetUnprocessedEvidence:
    def test_uninitialised_returns_empty(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("MEMLORA_DIR", str(tmp_path / "memlora_data"))
        result = get_unprocessed_evidence(str(tmp_path / "no_project"))
        assert result["items"] == []
        assert result["extractor_version"] == LLM_EXTRACTOR_VERSION

    def test_returns_evidence_for_initialised_project(self, project: str) -> None:
        _seed_evidence(project)
        result = get_unprocessed_evidence(project)
        assert len(result["items"]) == 1
        item = result["items"][0]
        assert "transcript_text" in item
        assert "argon2id" in item["transcript_text"]
        assert item["evidence_id"] > 0

    def test_excludes_already_processed_evidence(self, project: str) -> None:
        eid = _seed_evidence(project)
        # Mark the evidence as already-processed at the current version.
        cfg = Config.load()
        project_id = hash_project_path(project)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            conn.execute(
                "UPDATE raw_evidence SET llm_extractor_version=? WHERE id=?",
                (LLM_EXTRACTOR_VERSION, eid),
            )
            conn.commit()

        result = get_unprocessed_evidence(project)
        assert result["items"] == []

    def test_enqueues_pending_jobs(self, project: str) -> None:
        eid = _seed_evidence(project)
        get_unprocessed_evidence(project)

        cfg = Config.load()
        project_id = hash_project_path(project)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            pending = ej.list_pending_for_version(conn, project_id, LLM_EXTRACTOR_VERSION)
        assert len(pending) == 1
        assert pending[0].evidence_id == eid

    def test_limit_5_items_per_call(self, project: str) -> None:
        for i in range(7):
            _seed_evidence(project, session_id=f"sess-{i}")
        result = get_unprocessed_evidence(project)
        assert len(result["items"]) == 5


# ── store_extracted_events ───────────────────────────────────────────────────


def _good_event(**overrides) -> dict:
    base = {
        "event_type": "DECISION",
        "description": "Use argon2id.",
        "subject": "argon2id",
        "rationale": "OWASP.",
        "confidence": 0.9,
        "captured_at_role": "user",
    }
    base.update(overrides)
    return base


class TestStoreExtractedEvents:
    def test_well_formed_batch_persists_and_bumps_version(self, project: str) -> None:
        eid = _seed_evidence(project)
        result = store_extracted_events(
            project, eid, [_good_event()], LLM_EXTRACTOR_VERSION,
        )
        assert result["version_bumped"] is True
        assert len(result["inserted"]) == 1
        assert result["errors"] == []

    def test_bad_extractor_version_rejected(self, project: str) -> None:
        eid = _seed_evidence(project)
        result = store_extracted_events(
            project, eid, [_good_event()], "llm-v999",
        )
        assert result["version_bumped"] is False
        assert any("unknown extractor_version" in e["reason"] for e in result["errors"])
        assert result["inserted"] == []

    def test_invalid_evidence_id_rejected(self, project: str) -> None:
        result = store_extracted_events(
            project, 99999, [_good_event()], LLM_EXTRACTOR_VERSION,
        )
        assert result["version_bumped"] is False
        assert any("not found" in e["reason"] for e in result["errors"])

    def test_partial_failure_does_not_bump_version(self, project: str) -> None:
        eid = _seed_evidence(project)
        result = store_extracted_events(
            project, eid,
            [_good_event(), _good_event(event_type="BOGUS")],
            LLM_EXTRACTOR_VERSION,
        )
        assert result["version_bumped"] is False
        assert len(result["inserted"]) == 1
        assert len(result["errors"]) == 1

    def test_partial_failure_leaves_inserted_events_in_db(self, project: str) -> None:
        eid = _seed_evidence(project)
        store_extracted_events(
            project, eid,
            [_good_event(), _good_event(event_type="BOGUS")],
            LLM_EXTRACTOR_VERSION,
        )
        # The good event was inserted regardless of the bad sibling.
        cfg = Config.load()
        project_id = hash_project_path(project)
        db_path = get_db_path(cfg, project_id)
        with get_connection(db_path) as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM events WHERE project_id=?",
                (project_id,),
            ).fetchone()[0]
        assert n >= 1

    def test_retry_after_partial_succeeds_when_clean(self, project: str) -> None:
        eid = _seed_evidence(project)
        # First call: partial — bad event mixed with good
        store_extracted_events(
            project, eid,
            [_good_event(), _good_event(event_type="BOGUS")],
            LLM_EXTRACTOR_VERSION,
        )

        # Retry with only the good event — should bump version cleanly
        result = store_extracted_events(
            project, eid, [_good_event()], LLM_EXTRACTOR_VERSION,
        )
        assert result["version_bumped"] is True

    def test_empty_batch_bumps_version(self, project: str) -> None:
        """A genuinely empty extraction (LLM found nothing new) still counts
        as successful processing of this evidence_id at this version."""
        eid = _seed_evidence(project)
        result = store_extracted_events(
            project, eid, [], LLM_EXTRACTOR_VERSION,
        )
        assert result["version_bumped"] is True
        assert result["inserted"] == []
