"""Slash command installer for `/memlora-extract` — Phase A-5.

Writes a per-project `<project>/.claude/commands/memlora-extract.md` file that
Claude Code (and Cursor / Codex with similar conventions) treat as a user-
triggered slash command. The markdown body is the prompt the in-session LLM
runs against; it reads transcripts via `get_unprocessed_evidence` and stores
new events via `store_extracted_events`.

Per-project install means the extractor_version doesn't get smuggled into
prose; the slash command reads it from the MCP response at call time. Project
A and Project B can ship different versions without stepping on each other.

The command file is idempotent — `install_slash_command` skips writing when
the file already exists, so user edits are preserved across `memlora init`
re-runs.
"""
from __future__ import annotations

from pathlib import Path


SLASH_COMMAND_BODY = """\
---
description: Extract decisions and constraints the trie missed using the in-session LLM
---

Run an LLM enrichment pass over unprocessed CogniKernel transcripts.

## Step 1 — fetch pending evidence

Call `mcp__cognikernel__get_unprocessed_evidence` with `project_path` set to
the absolute path of the current project root.

The response contains:
- `extractor_version` — a string like `"llm-v1"`. Use this exact value in step 3.
- `items` — a list of up to 5 transcripts. Each item has:
  - `evidence_id` (int)
  - `session_id` (str)
  - `transcript_text` (str — the full session transcript)
  - `existing_trie_events` (list — already-captured subjects you can skip)
  - `captured_at` (epoch ms)

If `items` is empty, all transcripts are up-to-date — stop and tell the user.

## Step 2 — extract for each item

For each item, identify decisions and constraints NOT already in
`existing_trie_events`. Focus on:

- Library/framework choices stated as natural language ("we'll use X",
  "stick with Y", "going with Z")
- Convention rules (naming, casing, URL prefixes, file-structure rules)
- Negative constraints ("no X", "without Y", "never use Z")
- Architectural decisions paired with rationale
- Open work threads not yet captured

Skip:
- Restatements of trie-captured subjects (check the `existing_trie_events`
  list — match against `subject` field)
- File mentions (handled by a separate extractor)
- Implementation chatter that didn't end in a decision
- Code blocks, build output, error logs

## Step 3 — store new events

For each item that produced new events, call
`mcp__cognikernel__store_extracted_events` once with:

```json
{
  "project_path": "<same absolute path as step 1>",
  "evidence_id": <from the item>,
  "extractor_version": "<value from step 1's response>",
  "events": [
    {
      "event_type": "CONSTRAINT_HARD" | "CONSTRAINT_SOFT" | "DECISION" |
                    "APPROACH_ABANDONED" | "APPROACH_ABANDONED_DO_NOT_RETRY" |
                    "THREAD_OPEN",
      "description": "one clean sentence in present tense, no prompt verbs",
      "subject": "the X — short noun phrase, no leading articles",
      "rationale": "the WHY if stated, else \\"\\"",
      "confidence": 0.0-1.0,
      "captured_at_role": "user" | "assistant"
    }
  ]
}
```

The MCP tool reports:
- `inserted` — newly persisted event ids
- `skipped` — duplicates collapsed by content_hash
- `errors` — events that failed validation or insert
- `version_bumped` — true only when all events succeeded

If `version_bumped` is false, the same evidence_id will be re-offered on the
next call to `get_unprocessed_evidence`. That's expected — the retry path
re-attempts the previously-errored events while leaving inserted ones alone.

## Step 4 — report

After processing all items, tell the user how many events landed across all
transcripts and whether any errored.
"""


def install_slash_command(project_path: str | Path, *, overwrite: bool = False) -> Path:
    """Write the `/memlora-extract` slash command file inside the project.

    Returns the path of the written (or already-existing) command file.
    Skips writing when the file exists unless `overwrite=True`.
    """
    project_root = Path(project_path).resolve()
    commands_dir = project_root / ".claude" / "commands"
    commands_dir.mkdir(parents=True, exist_ok=True)

    target = commands_dir / "memlora-extract.md"
    if target.exists() and not overwrite:
        return target

    target.write_text(SLASH_COMMAND_BODY, encoding="utf-8")
    return target
