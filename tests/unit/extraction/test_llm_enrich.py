"""Tests for memlora.extraction.llm_enrich — Phase A-5 prompt + parser."""
from __future__ import annotations

import json

import pytest

from memlora.extraction.authority import LLM
from memlora.extraction.llm_enrich import (
    LLM_EXTRACTOR_VERSION,
    LlmExtractedEvent,
    VALID_LLM_EVENT_TYPES,
    build_extraction_prompt,
    parse_extraction_response,
    to_storage_event,
)


# ── build_extraction_prompt ──────────────────────────────────────────────────


class TestBuildExtractionPrompt:
    def test_includes_transcript(self) -> None:
        prompt = build_extraction_prompt("User:\nHello.", existing_trie_events=[])
        assert "User:\nHello." in prompt

    def test_handles_empty_trie_events(self) -> None:
        prompt = build_extraction_prompt("transcript", existing_trie_events=[])
        assert "(none" in prompt

    def test_renders_trie_subjects_as_bullets(self) -> None:
        events = [
            {"subject": "PostgreSQL", "description": "Use PostgreSQL"},
            {"subject": "argon2id", "description": "Use argon2id"},
        ]
        prompt = build_extraction_prompt("transcript", existing_trie_events=events)
        assert "- PostgreSQL" in prompt
        assert "- argon2id" in prompt

    def test_falls_back_to_description_when_subject_missing(self) -> None:
        events = [{"description": "Use PostgreSQL not SQLite", "subject": ""}]
        prompt = build_extraction_prompt("transcript", existing_trie_events=events)
        assert "PostgreSQL" in prompt

    def test_does_not_contain_markdown_fences_in_template_directives(self) -> None:
        """Defense against accidentally telling the LLM to wrap in ```json```
        — the response parser expects bare JSON."""
        prompt = build_extraction_prompt("transcript", existing_trie_events=[])
        assert "Do NOT wrap" in prompt


# ── parse_extraction_response ────────────────────────────────────────────────


class TestParseExtractionResponse:
    def test_well_formed_response_accepted(self) -> None:
        raw = json.dumps({
            "events": [
                {
                    "event_type": "DECISION",
                    "description": "Use argon2id for password hashing.",
                    "subject": "argon2id",
                    "rationale": "OWASP recommendation.",
                    "confidence": 0.9,
                    "captured_at_role": "user",
                }
            ]
        })
        result = parse_extraction_response(raw)
        assert len(result.accepted) == 1
        assert result.rejected == []
        assert result.accepted[0].subject == "argon2id"

    def test_empty_events_list_is_valid(self) -> None:
        result = parse_extraction_response('{"events": []}')
        assert result.accepted == []
        assert result.rejected == []

    def test_json_decode_error_rejected(self) -> None:
        result = parse_extraction_response("not json")
        assert result.accepted == []
        assert len(result.rejected) == 1
        assert "json_decode_error" in result.rejected[0].reason

    def test_non_object_root_rejected(self) -> None:
        result = parse_extraction_response('["not an object"]')
        assert result.accepted == []
        assert result.rejected
        assert "not an object" in result.rejected[0].reason

    def test_events_field_not_list_rejected(self) -> None:
        result = parse_extraction_response('{"events": "string"}')
        assert result.accepted == []

    def test_partial_validation_independent_per_event(self) -> None:
        """One bad event should not poison the well-formed siblings."""
        raw = json.dumps({
            "events": [
                {
                    "event_type": "DECISION",
                    "description": "Good event.",
                    "subject": "X",
                    "rationale": "",
                    "confidence": 0.7,
                    "captured_at_role": "user",
                },
                {
                    "event_type": "BOGUS_TYPE",
                    "description": "Bad event.",
                    "subject": "Y",
                    "rationale": "",
                    "confidence": 0.7,
                    "captured_at_role": "user",
                },
            ]
        })
        result = parse_extraction_response(raw)
        assert len(result.accepted) == 1
        assert len(result.rejected) == 1
        assert result.rejected[0].index == 1
        assert "invalid event_type" in result.rejected[0].reason

    def test_invalid_role_rejected(self) -> None:
        raw = json.dumps({
            "events": [{
                "event_type": "DECISION", "description": "d", "subject": "s",
                "rationale": "", "confidence": 0.5, "captured_at_role": "system",
            }]
        })
        result = parse_extraction_response(raw)
        assert result.accepted == []
        assert "invalid captured_at_role" in result.rejected[0].reason

    def test_confidence_out_of_range_rejected(self) -> None:
        raw = json.dumps({
            "events": [{
                "event_type": "DECISION", "description": "d", "subject": "s",
                "rationale": "", "confidence": 1.5, "captured_at_role": "user",
            }]
        })
        result = parse_extraction_response(raw)
        assert result.accepted == []
        assert "out of range" in result.rejected[0].reason

    def test_empty_description_rejected(self) -> None:
        raw = json.dumps({
            "events": [{
                "event_type": "DECISION", "description": "  ", "subject": "s",
                "rationale": "", "confidence": 0.5, "captured_at_role": "user",
            }]
        })
        result = parse_extraction_response(raw)
        assert result.accepted == []
        assert "description is empty" in result.rejected[0].reason


# ── to_storage_event ─────────────────────────────────────────────────────────


class TestToStorageEvent:
    def test_sets_llm_authority_and_provenance(self) -> None:
        extracted = LlmExtractedEvent(
            event_type="DECISION",
            description="Use argon2id.",
            subject="argon2id",
            rationale="OWASP.",
            confidence=0.9,
            captured_at_role="user",
        )
        event = to_storage_event(extracted, project_id="p1", session_id="s1", evidence_id=42)
        assert event.payload["authority"] == LLM
        assert event.payload["provenance"] == "llm"
        assert event.evidence_id == 42

    def test_normalizes_description(self) -> None:
        """A-1 normalize_description must run on LLM output too."""
        extracted = LlmExtractedEvent(
            event_type="DECISION",
            description="Confirm we'll use argon2id",  # ends without period
            subject="argon2id",
            rationale="",
            confidence=0.8,
            captured_at_role="user",
        )
        event = to_storage_event(extracted, project_id="p1", session_id="s1")
        # Period appended; "Confirm " stripped.
        assert event.payload["description"].endswith(".")
        assert not event.payload["description"].startswith("Confirm ")

    def test_content_hash_computed(self) -> None:
        extracted = LlmExtractedEvent(
            event_type="DECISION",
            description="Use argon2id.",
            subject="argon2id",
            rationale="",
            confidence=0.9,
            captured_at_role="user",
        )
        event = to_storage_event(extracted, project_id="p1", session_id="s1")
        assert event.content_hash
        assert len(event.content_hash) == 64  # SHA-256 hex


# ── constants ────────────────────────────────────────────────────────────────


class TestConstants:
    def test_version_is_a_versioned_string(self) -> None:
        assert LLM_EXTRACTOR_VERSION.startswith("llm-v")

    def test_valid_event_types_excludes_component_status(self) -> None:
        """The LLM should NOT emit COMPONENT_STATUS — the file_mentions
        extractor owns that domain."""
        assert "COMPONENT_STATUS" not in VALID_LLM_EVENT_TYPES

    def test_valid_event_types_excludes_thread_close(self) -> None:
        assert "THREAD_CLOSE" not in VALID_LLM_EVENT_TYPES
