"""Compose the text that gets embedded for an event (E1).

Embedding the bare `description` throws away the structured metadata CogniKernel
already captured. The composed input leads with the most semantically-salient
fields so the model clusters by *what the decision is about*:

  - subject (payload.subject, else payload.triple.subject) — the topic;
  - description — the statement itself;
  - a light scope tag — `path` for component status (file the event concerns).

Kept deliberately short: over-stuffing dilutes the vector. The composition is
deterministic, so a change here is a model_version-level change (re-embed).
"""
from __future__ import annotations

from typing import Any


def _subject_of(payload: dict[str, Any]) -> str:
    subj = (payload.get("subject") or "").strip()
    if subj:
        return subj
    triple = payload.get("triple")
    if isinstance(triple, dict):
        return (triple.get("subject") or triple.get("object") or "").strip()
    return ""


def embedding_input(payload: dict[str, Any], event_type: str) -> str:
    """Return the composed text to embed for an event, or '' if nothing usable."""
    description = (payload.get("description") or "").strip()

    if event_type == "COMPONENT_STATUS":
        # Component events concern a file; lead with the path + intent/status.
        path = (payload.get("path") or "").strip()
        intent = (payload.get("intent") or payload.get("status") or "").strip()
        parts = [p for p in (path, intent, description) if p]
        return " — ".join(parts)

    subject = _subject_of(payload)
    if subject and description:
        return f"{subject}: {description}"
    return subject or description
