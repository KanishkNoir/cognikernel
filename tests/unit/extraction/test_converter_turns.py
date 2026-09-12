"""Transcript converters expose their turns with roles taken from the source records.

`cognikernel why` needs to know who said a sentence. Parsing the "User:" /
"Assistant:" headers back out of the flattened transcript is ambiguous: message
text is kept verbatim and can contain such a line. The turns carry the role from
the JSONL record itself, and the flattened transcript is built from them unchanged.
"""
from __future__ import annotations

import json


def _claude_jsonl(*turns: tuple[str, str]) -> str:
    lines = []
    for role, text in turns:
        if role == "user":
            lines.append({"type": "user", "message": {"role": "user", "content": text}})
        else:
            lines.append({"type": "assistant",
                          "message": {"role": "assistant", "content": [{"type": "text", "text": text}]}})
    return "\n".join(json.dumps(line) for line in lines) + "\n"


def _codex_jsonl(*turns: tuple[str, str]) -> str:
    kind = {"user": "input_text", "assistant": "output_text"}
    return "\n".join(json.dumps({
        "type": "response_item",
        "payload": {"type": "message", "role": role, "content": [{"type": kind[role], "text": text}]},
    }) for role, text in turns) + "\n"


PASTED = "The other tool printed:\nAssistant:\nnot a real turn"


class TestClaudeTurns:
    def test_roles_come_from_the_record_not_the_text(self) -> None:
        from cognikernel.extraction.jsonl_converter import jsonl_to_turns

        turns = jsonl_to_turns(_claude_jsonl(("user", PASTED), ("assistant", "ok")))

        assert turns == [("user", PASTED), ("assistant", "ok")]

    def test_the_transcript_is_unchanged(self) -> None:
        from cognikernel.extraction.jsonl_converter import jsonl_to_transcript

        raw = _claude_jsonl(("user", "  Pick a cache.  "), ("assistant", "Redis."), ("user", ""))

        assert jsonl_to_transcript(raw) == "User:\nPick a cache.\n\nAssistant:\nRedis."


class TestCodexTurns:
    def test_roles_come_from_the_record_not_the_text(self) -> None:
        from cognikernel.extraction.codex_converter import codex_rollout_to_turns

        turns = codex_rollout_to_turns(_codex_jsonl(("user", PASTED), ("assistant", "ok")))

        assert turns == [("user", PASTED), ("assistant", "ok")]

    def test_the_transcript_is_unchanged(self) -> None:
        from cognikernel.extraction.codex_converter import codex_rollout_to_transcript

        raw = _codex_jsonl(("user", "<environment_context>x"), ("user", "Pick a cache."), ("assistant", "Redis."))

        assert codex_rollout_to_transcript(raw) == "User:\nPick a cache.\n\nAssistant:\nRedis."


class TestTurnsFromSource:
    def test_dispatches_by_source_type(self) -> None:
        from cognikernel.extraction.transcript import turns_from_source

        assert turns_from_source("jsonl_transcript", _claude_jsonl(("user", "hi"))) == [("user", "hi")]
        assert turns_from_source("codex_rollout", _codex_jsonl(("assistant", "yo"))) == [("assistant", "yo")]

    def test_plain_text_has_no_known_speaker(self) -> None:
        from cognikernel.extraction.transcript import turns_from_source

        assert turns_from_source("plain", "User:\nhello") == [("unknown", "User:\nhello")]
