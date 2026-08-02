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


# ── path recall + non-truncation (spec §0.1) ─────────────────────────────────

from hypothesis import given, strategies as st  # noqa: E402

from cognikernel.extraction.file_mentions import _FILE_PATTERN  # noqa: E402
from cognikernel.utils.paths import canonicalize_path  # noqa: E402


def _match(text: str) -> list[str]:
    return [m.group(0) for m in _FILE_PATTERN.finditer(text)]


class TestPathRecall:
    """Shapes that produced NO match at all before the widening."""

    def test_matches_dotdir_path(self) -> None:
        assert _match("edited .claude/settings.json today")

    def test_matches_dot_relative_path(self) -> None:
        assert _match("edited ./src/storage/connection.py today")

    def test_matches_absolute_posix_path(self) -> None:
        assert _match("edited /srv/app/src/storage/connection.py today")

    def test_matches_windows_backslash_path(self) -> None:
        assert _match(r"edited C:\Users\Admin\src\storage\connection.py today")

    def test_windows_path_canonicalizes_to_forward_slashes(self) -> None:
        hits = _match(r"edited src\storage\connection.py today")
        assert hits
        assert canonicalize_path(hits[0]) == "src/storage/connection.py"

    def test_still_matches_plain_relative_path(self) -> None:
        assert _match("edited src/storage/connection.py today") == [
            "src/storage/connection.py"
        ]


class TestEndToEndEventProduction:
    """Matching the regex is not enough — an event must actually come out.

    canonicalize_path returns '' for ANY absolute path when project_root is
    None (paths.py rule 7), and file_mentions drops empty results. Without the
    project_root plumbing, absolute paths match the pattern and then vanish.
    """

    def _paths(self, text: str, project_root: str | None = None) -> list[str]:
        events = extract_file_mention_events(
            [_assistant_sentence(text)], "p1", "s1", project_root=project_root
        )
        return [e.payload["path"] for e in events]

    def test_relative_path_produces_event(self) -> None:
        assert "src/storage/connection.py" in self._paths(
            "I edited src/storage/connection.py today"
        )

    def test_dotdir_path_produces_event(self) -> None:
        assert ".claude/settings.json" in self._paths(
            "I edited .claude/settings.json today"
        )

    def test_windows_relative_path_produces_event(self) -> None:
        assert "src/storage/connection.py" in self._paths(
            r"I edited src\storage\connection.py today"
        )

    def test_absolute_path_needs_project_root(self) -> None:
        text = r"I edited C:\proj\src\storage\connection.py today"
        assert self._paths(text) == []
        assert "src/storage/connection.py" in self._paths(text, project_root=r"C:\proj")

    def test_path_outside_project_root_is_dropped(self) -> None:
        assert self._paths(
            r"I edited C:\elsewhere\other.py today", project_root=r"C:\proj"
        ) == []


class TestNoTruncation:
    """D1 regression guard. Expected to pass on first run — its job is to fail
    loudly if the 2026-05-10 first-character-strip behaviour ever returns."""

    @given(
        segments=st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8),
            min_size=1,
            max_size=4,
        ),
        stem=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10),
    )
    def test_extracted_path_is_never_a_prefix_truncation(
        self, segments: list[str], stem: str
    ) -> None:
        path = "/".join(segments) + "/" + stem + ".py"
        hits = _match(f"edited {path} today")
        assert hits, f"path vanished entirely: {path!r}"
        assert canonicalize_path(hits[0]) == path
