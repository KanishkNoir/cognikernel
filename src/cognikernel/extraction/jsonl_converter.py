"""Convert a Claude Code session JSONL file to plain text for the extraction pipeline.

Claude Code stores sessions at ~/.claude/projects/<hash>/<session-id>.jsonl.
Each line is a JSON object. Only user/assistant text blocks carry signal;
everything else (tool calls, thinking, metadata, file snapshots) is noise.
"""
from __future__ import annotations

import json


_ROLE_HEADER = {"user": "User", "assistant": "Assistant"}


def turns_to_transcript(turns: list[tuple[str, str]]) -> str:
    """Flatten turns into the User: / Assistant: prose the trie scanner reads."""
    return "\n\n".join(f"{_ROLE_HEADER[role]}:\n{text}" for role, text in turns)


def jsonl_to_transcript(jsonl_text: str) -> str:
    """Return a plain-text transcript extracted from a Claude Code JSONL session.

    Keeps only human-readable user and assistant text blocks, formatted as
    ## User / ## Assistant sections so the trie scanner gets clean prose.
    """
    return turns_to_transcript(jsonl_to_turns(jsonl_text))


def jsonl_to_turns(jsonl_text: str) -> list[tuple[str, str]]:
    """The user and assistant text turns of a Claude Code JSONL session, in order.

    Each role comes from its JSONL record, never from the text: message text is
    kept verbatim and can itself contain a line such as "Assistant:".
    """
    turns: list[tuple[str, str]] = []

    for raw_line in jsonl_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type")

        if msg_type == "user":
            if obj.get("isMeta"):
                continue
            text = _user_text(obj)
            if text:
                turns.append(("user", text))

        elif msg_type == "assistant":
            text = _assistant_text(obj)
            if text:
                turns.append(("assistant", text))

    return turns


# ── internals ────────────────────────────────────────────────────────────────

def _user_text(obj: dict) -> str:
    content = obj.get("message", {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return _join_text_blocks(content)
    return ""


def _assistant_text(obj: dict) -> str:
    content = obj.get("message", {}).get("content", [])
    if not isinstance(content, list):
        return ""
    return _join_text_blocks(content)


def _join_text_blocks(blocks: list) -> str:
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text", "").strip()
            if t:
                parts.append(t)
    return "\n".join(parts)
