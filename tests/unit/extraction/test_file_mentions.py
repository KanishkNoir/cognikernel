"""Tests for cognikernel.extraction.file_mentions.

Focus: bare-basename rejection at insertion time. Mentions like ``env.py``
without a directory prefix are extractor noise (they collide with the
qualified form ``alembic/env.py``) and must not produce a COMPONENT_STATUS
event. The C2 reference filter lives in storage/projections.py:rebuild_projection;
this test pins the upstream version.
"""
from __future__ import annotations

from cognikernel.extraction.file_mentions import extract_file_mention_events
from cognikernel.extraction.tokenize import Sentence


def _assistant_sentence(text: str, idx: int = 0) -> Sentence:
    return Sentence(
        text=text,
        start_offset=0,
        end_offset=len(text),
        role="assistant",
        is_code_block=False,
        sentence_index=idx,
    )


class TestBareBasenameRejection:
    def test_bare_basename_does_not_emit_event(self) -> None:
        """A bare ``env.py`` mention should produce zero events."""
        sentences = [_assistant_sentence("I modified env.py to override the test fixture.")]
        events = extract_file_mention_events(sentences, "p1", "s1")
        assert events == [], (
            "extract_file_mention_events emitted an event for a bare-basename "
            "mention; must drop bare basenames at extraction time."
        )

    def test_qualified_path_still_emits_event(self) -> None:
        """The same name with a directory prefix should produce one event."""
        sentences = [_assistant_sentence(
            "I modified alembic/env.py to override the test fixture."
        )]
        events = extract_file_mention_events(sentences, "p1", "s1")
        assert len(events) == 1
        assert events[0].payload["path"] == "alembic/env.py"
        assert events[0].event_type == "COMPONENT_STATUS"

    def test_mixed_mentions_drops_bare_keeps_qualified(self) -> None:
        sentences = [_assistant_sentence(
            "Updated config.py and backend/app/core/config.py; "
            "also edited alembic.ini and backend/alembic.ini."
        )]
        events = extract_file_mention_events(sentences, "p1", "s1")
        paths = sorted(e.payload["path"] for e in events)
        # Only the qualified forms should survive.
        assert "config.py" not in paths
        assert "alembic.ini" not in paths
        assert "backend/app/core/config.py" in paths
        assert "backend/alembic.ini" in paths
