"""Codex rollout -> transcript adapter (Sprint L / L1).

Pins the rollout schema against a real-shape fixture and asserts the converter
keeps only user/assistant prose, drops system/developer/reasoning/function-call
and the event_msg UI duplicates, and never raises on garbage.
"""
from __future__ import annotations

from pathlib import Path

from memlora.extraction.codex_converter import codex_rollout_to_transcript
from memlora.extraction.transcript import transcript_from_source

FIXTURE = Path(__file__).parents[2] / "fixtures" / "codex_rollout_sample.jsonl"


class TestCodexConverter:
    def test_golden_keeps_user_and_assistant_prose(self) -> None:
        out = codex_rollout_to_transcript(FIXTURE.read_text(encoding="utf-8"))
        assert "User:\nUse Postgres for the primary datastore, not MySQL." in out
        assert "Assistant:\nUnderstood. We will use Postgres" in out
        # exactly one user + one assistant section (no event_msg double-count)
        assert out.count("User:") == 1
        assert out.count("Assistant:") == 1

    def test_drops_developer_environment_and_tooling_noise(self) -> None:
        out = codex_rollout_to_transcript(FIXTURE.read_text(encoding="utf-8"))
        assert "permissions instructions" not in out      # developer role
        assert "<environment_context>" not in out         # injected user context
        assert "I should confirm" not in out              # reasoning item
        assert "function_call" not in out and "\"command\"" not in out

    def test_tolerant_of_garbage_and_empty(self) -> None:
        assert codex_rollout_to_transcript("") == ""
        bad = 'not json\n{"type":"response_item"}\n{"type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}}'
        out = codex_rollout_to_transcript(bad)
        assert out == "User:\nhi"

    def test_dispatcher_routes_by_source_type(self) -> None:
        raw = FIXTURE.read_text(encoding="utf-8")
        assert transcript_from_source("codex_rollout", raw) == codex_rollout_to_transcript(raw)
        # unknown / plain -> passthrough
        assert transcript_from_source("transcript", "plain prose") == "plain prose"
        assert transcript_from_source(None, "plain prose") == "plain prose"
