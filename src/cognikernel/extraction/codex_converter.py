"""Convert a Codex CLI session *rollout* JSONL file to plain text for extraction.

Codex stores sessions at ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl. Each line
is a record ``{"timestamp", "type", "payload"}``. Only ``response_item`` records
whose payload is a user/assistant ``message`` carry signal; system/``developer``
prompts, reasoning, function calls, and the ``event_msg`` UI duplicates
(``user_message``/``agent_message``) are noise and would double-count.

This is the Codex analogue of ``jsonl_converter.jsonl_to_transcript`` and emits the
same ``User:`` / ``Assistant:`` prose shape the trie scanner expects, so the whole
downstream pipeline (delta-slice, dedup, classify, merge) is unchanged.
"""
from __future__ import annotations

import json

# Injected machine context that arrives under the *user* role but is not user
# prose (Codex prepends these every session). Matched against the stripped head of
# the text, case-insensitively; kept deliberately small to avoid eating real prose.
_SYSTEM_USER_PREFIXES = (
    "<environment_context>",
    "<permissions",
    "<user_instructions>",
)


def codex_rollout_to_transcript(jsonl_text: str) -> str:
    """Return a plain-text transcript from a Codex rollout JSONL session.

    Tolerant by design: unparseable lines and unknown record types are skipped,
    never raised — a truncated/garbage rollout yields a partial transcript.
    """
    sections: list[str] = []

    for raw_line in jsonl_text.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            obj = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        if obj.get("type") != "response_item":
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "message":
            continue

        role = payload.get("role")
        if role == "user":
            text = _message_text(payload)
            if text and not _is_system_injection(text):
                sections.append(f"User:\n{text}")
        elif role == "assistant":
            text = _message_text(payload)
            if text:
                sections.append(f"Assistant:\n{text}")
        # developer/system and any other role: drop.

    return "\n\n".join(sections)


# ── internals ────────────────────────────────────────────────────────────────

def _message_text(payload: dict) -> str:
    content = payload.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") in ("input_text", "output_text"):
            t = block.get("text", "")
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
    return "\n".join(parts)


def _is_system_injection(text: str) -> bool:
    head = text.lstrip()[:40].lower()
    return any(head.startswith(p) for p in _SYSTEM_USER_PREFIXES)
