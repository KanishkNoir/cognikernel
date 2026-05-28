"""LLM enrichment layer — Phase A-5.

The trie + pattern engine catches phrases stated using the ~150 signal patterns
in `signals.py` and `patterns.py`. Many real decisions are paraphrased outside
those patterns. The LLM enrichment layer fills that gap by running the user's
in-session LLM (Claude Code, Cursor, Codex) against unprocessed raw_evidence
via two MCP tools and a `/memlora-extract` slash command.

This module owns:
  - `LLM_EXTRACTOR_VERSION` constant — bumped to invalidate prior enrichments
  - `build_extraction_prompt()` — the prompt the LLM consumes
  - `parse_extraction_response()` — validator that converts LLM JSON to Events
  - `LlmExtractedEvent` dataclass for type-safety in MCP tool boundaries

The MCP tool layer (`integration/mcp_server.py`) provides the network surface;
this module is pure data + helpers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from memlora.extraction.authority import LLM
from memlora.extraction.hashing import compute_content_hash
from memlora.extraction.normalize import normalize_description
from memlora.storage.events import Event


# ── versioning ───────────────────────────────────────────────────────────────


LLM_EXTRACTOR_VERSION: str = "llm-v1"
"""Current extractor identifier. Bump to invalidate existing enrichments.

Stored on `raw_evidence.llm_extractor_version` after a successful enrichment.
The MCP tool returns the current value to the slash command so the prompt
template never has to hardcode it.
"""


# ── event-type whitelist ─────────────────────────────────────────────────────


VALID_LLM_EVENT_TYPES = frozenset({
    "CONSTRAINT_HARD",
    "CONSTRAINT_SOFT",
    "DECISION",
    "APPROACH_ABANDONED",
    "APPROACH_ABANDONED_DO_NOT_RETRY",
    "THREAD_OPEN",
})
"""Event types the LLM is allowed to emit.

`COMPONENT_STATUS` is intentionally excluded — file mentions belong to the
file_mentions extractor, not the LLM. Same for THREAD_CLOSE (low-value).
"""


VALID_ROLES = frozenset({"user", "assistant"})


# ── data model ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LlmExtractedEvent:
    """Validated payload returned by the LLM. Caller converts to storage.Event."""
    event_type: str
    description: str
    subject: str
    rationale: str
    confidence: float
    captured_at_role: str


@dataclass(frozen=True)
class ValidationError:
    index: int
    reason: str


@dataclass(frozen=True)
class ParseResult:
    accepted: list[LlmExtractedEvent]
    rejected: list[ValidationError]


# ── prompt construction ──────────────────────────────────────────────────────


_PROMPT_TEMPLATE = """\
You are extracting structured decisions and constraints from a Claude Code
transcript. The trie-based pre-pass has already captured the obvious phrases
(stored as `existing_trie_events`); your job is to fill the recall gap by
identifying facts the trie missed.

## Transcript
{transcript}

## Already-extracted trie events (subjects you can skip)
{existing_trie_summary}

## What to look for (in order of priority)
- Library/framework choices: "we'll use X", "stick with Y", "going with Z"
- Convention rules: naming, casing, URL prefixes, file structure conventions
- Negative constraints: "no X", "without Y", "never use Z"
- Architectural decisions with rationale ("we chose X because Y")
- Open work threads not yet captured ("next we need to...")

## What to skip
- Restatements of trie events (check subjects above)
- File mentions (a separate extractor handles those)
- Implementation-level chat ("I'll write the loop...")
- Code blocks, error messages, build output

## Output format
Return a JSON object with a single key `"events"` whose value is a list of
event objects. Each event MUST have all of:

  event_type:        one of {{CONSTRAINT_HARD, CONSTRAINT_SOFT, DECISION,
                     APPROACH_ABANDONED, APPROACH_ABANDONED_DO_NOT_RETRY,
                     THREAD_OPEN}}
  description:       one clean sentence in present tense, no prompt verbs
  subject:           the X in "use X" — short noun phrase, no articles
  rationale:         the WHY if stated in the transcript, else ""
  confidence:        float 0.0-1.0; use 1.0 only when text is unambiguous
  captured_at_role:  "user" or "assistant" — whose statement is this

If you find no new events, return `{{"events": []}}`. Do NOT wrap the JSON
in markdown fences or add explanatory prose around it.
"""


def build_extraction_prompt(
    transcript: str,
    existing_trie_events: list[dict[str, Any]],
) -> str:
    """Return a single-string prompt ready for the in-session LLM.

    `existing_trie_events` is the list-of-dicts the MCP tool returns alongside
    each transcript. We render their subjects as a bullet list so the LLM can
    skip already-known facts.
    """
    summary = _render_trie_summary(existing_trie_events)
    return _PROMPT_TEMPLATE.format(
        transcript=transcript.strip(),
        existing_trie_summary=summary,
    )


def _render_trie_summary(events: list[dict[str, Any]]) -> str:
    if not events:
        return "(none — this is the first extraction pass for this transcript)"
    lines: list[str] = []
    for e in events:
        subject = (e.get("subject") or "").strip()
        if not subject:
            # Fall back to first 60 chars of description.
            subject = (e.get("description") or "").strip()[:60]
        if subject:
            lines.append(f"- {subject}")
    if not lines:
        return "(none — trie events lacked extractable subjects)"
    return "\n".join(lines)


# ── response parsing ─────────────────────────────────────────────────────────


def parse_extraction_response(raw: str) -> ParseResult:
    """Validate an LLM JSON response into a typed ParseResult.

    Each event is validated independently — a malformed event in position 2
    does not reject the well-formed events in positions 0, 1, 3, ...
    Returns:
      accepted — events that passed validation
      rejected — (index, reason) pairs for diagnostic feedback
    """
    accepted: list[LlmExtractedEvent] = []
    rejected: list[ValidationError] = []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ParseResult(
            accepted=[],
            rejected=[ValidationError(index=-1, reason=f"json_decode_error: {exc}")],
        )

    if not isinstance(data, dict):
        return ParseResult(
            accepted=[],
            rejected=[ValidationError(index=-1, reason="response root is not an object")],
        )

    events_raw = data.get("events", [])
    if not isinstance(events_raw, list):
        return ParseResult(
            accepted=[],
            rejected=[ValidationError(index=-1, reason="`events` field is not a list")],
        )

    for i, item in enumerate(events_raw):
        if not isinstance(item, dict):
            rejected.append(ValidationError(i, "event is not an object"))
            continue
        result = _validate_event(item, i)
        if isinstance(result, ValidationError):
            rejected.append(result)
        else:
            accepted.append(result)

    return ParseResult(accepted=accepted, rejected=rejected)


def _validate_event(item: dict[str, Any], index: int) -> LlmExtractedEvent | ValidationError:
    """Validate one event dict; return either a typed event or a ValidationError."""
    event_type = item.get("event_type", "")
    if event_type not in VALID_LLM_EVENT_TYPES:
        return ValidationError(index, f"invalid event_type {event_type!r}")

    description = (item.get("description") or "").strip()
    if not description:
        return ValidationError(index, "description is empty")
    if len(description) > 500:
        return ValidationError(index, "description exceeds 500 chars")

    role = item.get("captured_at_role", "")
    if role not in VALID_ROLES:
        return ValidationError(index, f"invalid captured_at_role {role!r}")

    try:
        confidence = float(item.get("confidence", 0.5))
    except (TypeError, ValueError):
        return ValidationError(index, "confidence is not numeric")
    if not 0.0 <= confidence <= 1.0:
        return ValidationError(index, f"confidence out of range: {confidence}")

    subject = (item.get("subject") or "").strip()
    rationale = (item.get("rationale") or "").strip()

    return LlmExtractedEvent(
        event_type=event_type,
        description=description,
        subject=subject,
        rationale=rationale,
        confidence=confidence,
        captured_at_role=role,
    )


# ── Event construction ──────────────────────────────────────────────────────


def to_storage_event(
    extracted: LlmExtractedEvent,
    *,
    project_id: str,
    session_id: str,
    evidence_id: int | None = None,
) -> Event:
    """Convert a validated LLM-extracted event to a storage.Event.

    Applies A-1 normalize_description and computes content_hash so caller
    just inserts. provenance and authority are set to 'llm' so the
    Pending Confirmation suppression treats LLM events as confirming.
    """
    description = normalize_description(extracted.description)
    return Event(
        project_id=project_id,
        session_id=session_id,
        event_type=extracted.event_type,
        payload={
            "description": description,
            "rationale": extracted.rationale,
            "subject": extracted.subject,
            "confidence": extracted.confidence,
            "source_role": extracted.captured_at_role,
            "matched_phrase": "LLM",
            "affected_files": [],
            "authority": LLM,
            "provenance": "llm",
        },
        content_hash=compute_content_hash(extracted.event_type, description),
        weight=extracted.confidence,
        evidence_id=evidence_id,
    )
