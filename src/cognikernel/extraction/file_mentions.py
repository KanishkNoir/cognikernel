"""Extract COMPONENT_STATUS events from file paths mentioned in assistant turns.

Scans assistant sentences for file path patterns and emits a COMPONENT_STATUS
event for each unique file that appears near an action verb. This populates the
component map without requiring a git diff.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from cognikernel.extraction.git_augment import infer_intent_from_path
from cognikernel.extraction.hashing import compute_content_hash
from cognikernel.storage.events import Event
from cognikernel.utils.paths import canonicalize_path, is_bare_basename

if TYPE_CHECKING:
    from cognikernel.extraction.tokenize import Sentence

_FILE_PATTERN = re.compile(
    r"(?<![a-zA-Z0-9_./\\])"
    r"(?:[a-zA-Z0-9_][a-zA-Z0-9_/.-]*/)*"
    r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*\."
    r"(?:py|ts|tsx|js|jsx|mjs|json|yaml|yml|sql|md|toml|env|cfg|ini|go|rs|java|cs)"
    r"(?![a-zA-Z0-9_])",
    re.ASCII,
)

_WRITE_VERBS = re.compile(
    r"\b(?:edit(?:ed|ing)?|writ(?:e|ing|ten)|wrot[e]|modif(?:y|ied|ying)"
    r"|creat(?:e|ed|ing)|updat(?:e|ed|ing)|chang(?:e|ed|ing)|add(?:ed|ing)?)\b",
    re.IGNORECASE,
)

_READ_VERBS = re.compile(
    r"\b(?:read(?:ing)?|check(?:ed|ing)?|look(?:ed|ing)?\s+at"
    r"|review(?:ed|ing)?|found\s+in|defined\s+in|located\s+in|see(?:ing)?|scan(?:ned|ning)?)\b",
    re.IGNORECASE,
)

_ANY_ACTION_VERB = re.compile(
    r"\b(?:edit(?:ed|ing)?|writ(?:e|ing|ten)|wrot[e]|modif(?:y|ied|ying)"
    r"|creat(?:e|ed|ing)|updat(?:e|ed|ing)|chang(?:e|ed|ing)|add(?:ed|ing)?"
    r"|read(?:ing)?|check(?:ed|ing)?|look(?:ed|ing)?\s+at"
    r"|review(?:ed|ing)?|found\s+in|defined\s+in|located\s+in|see(?:ing)?|scan(?:ned|ning)?)\b",
    re.IGNORECASE,
)

_WEIGHT_REFERENCED = 0.4
_WEIGHT_MODIFIED = 0.6


def extract_file_mention_events(
    sentences: list[Sentence],
    project_id: str,
    session_id: str,
) -> list[Event]:
    """Return COMPONENT_STATUS events for files mentioned in assistant turns.

    Only emits an event when an action verb appears within ±1 sentence of the
    file mention, avoiding false positives from bare filenames in explanations.
    Each unique path is emitted at most once per call.
    """
    events: list[Event] = []
    seen_paths: set[str] = set()
    n = len(sentences)

    for i, sentence in enumerate(sentences):
        if sentence.role != "assistant" or sentence.is_code_block:
            continue

        for match in _FILE_PATTERN.finditer(sentence.text):
            # Canonicalize so backslash / double-slash variants collapse into
            # a single key. Bare basenames (no directory component) are dropped
            # at insertion time — they're extractor noise that conflicts with
            # the prefixed canonical form (C4). Mirrors the same filter in
            # storage/projections.py:rebuild_projection.
            path = canonicalize_path(match.group(0))
            if not path or is_bare_basename(path) or path in seen_paths:
                continue

            # Build a window of ±1 sentences for verb detection.
            window_texts = [sentence.text]
            if i > 0:
                window_texts.append(sentences[i - 1].text)
            if i < n - 1:
                window_texts.append(sentences[i + 1].text)
            combined = " ".join(window_texts)

            if not _ANY_ACTION_VERB.search(combined):
                continue

            seen_paths.add(path)

            if _WRITE_VERBS.search(combined):
                status = "modified"
                weight = _WEIGHT_MODIFIED
            else:
                status = "referenced"
                weight = _WEIGHT_REFERENCED

            intent = infer_intent_from_path(path)
            description = f"{path} {status} (transcript mention)"
            payload = {
                "path": path,
                "status": status,
                "intent": intent,
                "source": "transcript",
                "description": description,
                "rationale": "",
                "authority": "inferred_from_code",
                "provenance": "file_mention",
            }
            events.append(
                Event(
                    project_id=project_id,
                    session_id=session_id,
                    event_type="COMPONENT_STATUS",
                    payload=payload,
                    content_hash=compute_content_hash("COMPONENT_STATUS", description),
                    weight=weight,
                )
            )

    return events
