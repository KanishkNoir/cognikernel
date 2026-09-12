"""Single dispatch point: evidence ``source_type`` -> plain-text transcript.

Capture stores raw evidence verbatim and tags it with a ``source_type``; the
extraction worker converts that raw bytes blob to prose at extraction time. Keeping
the source_type -> converter mapping in one place means a new platform (Codex, and
later others) is one branch here instead of N patched call sites.
"""
from __future__ import annotations

from cognikernel.extraction.codex_converter import codex_rollout_to_transcript, codex_rollout_to_turns
from cognikernel.extraction.jsonl_converter import jsonl_to_transcript, jsonl_to_turns


def turns_from_source(source_type: str | None, raw: str) -> list[tuple[str, str]]:
    """*raw* evidence as (role, text) turns by *source_type*, each role taken from
    its source record. Unknown/plain source types are one turn with role "unknown":
    a "User:" line in plain text is not evidence of who said it.
    """
    if source_type == "jsonl_transcript":
        return jsonl_to_turns(raw)
    if source_type == "codex_rollout":
        return codex_rollout_to_turns(raw)
    return [("unknown", raw)] if raw.strip() else []


def transcript_from_source(source_type: str | None, raw: str) -> str:
    """Convert *raw* evidence to a plain-text transcript by *source_type*.

    Unknown/plain source types pass through unchanged (they are already prose).
    """
    if source_type == "jsonl_transcript":
        return jsonl_to_transcript(raw)
    if source_type == "codex_rollout":
        return codex_rollout_to_transcript(raw)
    return raw
