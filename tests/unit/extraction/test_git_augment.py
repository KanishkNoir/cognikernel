"""Tests for git diff augmentation and cross-referencing."""
import pytest
from memlora.extraction.git_augment import (
    FileChange,
    cross_reference_signals,
    extract_git_events,
    infer_intent_from_path,
    parse_diff,
)
from memlora.storage.events import Event

_SAMPLE_DIFF = """\
M\tsrc/auth/middleware.py
A\tsrc/routes/users.py
D\tsrc/legacy/redis_client.py
R100\tsrc/old_name.py\tsrc/new_name.py
src/auth/middleware.py | 42 ++++++++++-----
src/routes/users.py   | 15 +++++++++++++++
src/legacy/redis_client.py | 8 --------
"""


class TestParseDiff:
    def test_parses_modified_file(self) -> None:
        changes = parse_diff(_SAMPLE_DIFF)
        modified = [c for c in changes if c.change_type == "modified"]
        assert any(c.path == "src/auth/middleware.py" for c in modified)

    def test_parses_added_file(self) -> None:
        changes = parse_diff(_SAMPLE_DIFF)
        added = [c for c in changes if c.change_type == "added"]
        assert any(c.path == "src/routes/users.py" for c in added)

    def test_parses_deleted_file(self) -> None:
        changes = parse_diff(_SAMPLE_DIFF)
        deleted = [c for c in changes if c.change_type == "deleted"]
        assert any(c.path == "src/legacy/redis_client.py" for c in deleted)

    def test_parses_renamed_file_new_path(self) -> None:
        changes = parse_diff(_SAMPLE_DIFF)
        renamed = [c for c in changes if c.change_type == "renamed"]
        assert any(c.path == "src/new_name.py" for c in renamed)

    def test_empty_diff_returns_empty(self) -> None:
        assert parse_diff("") == []


class TestExtractGitEvents:
    def test_produces_component_status_events(self) -> None:
        events = extract_git_events(_SAMPLE_DIFF, "proj1", "sess1")
        assert all(e.event_type == "COMPONENT_STATUS" for e in events)

    def test_one_event_per_changed_file(self) -> None:
        events = extract_git_events(_SAMPLE_DIFF, "proj1", "sess1")
        paths = {e.payload["path"] for e in events}
        assert "src/auth/middleware.py" in paths
        assert "src/routes/users.py" in paths

    def test_weight_formula_for_high_churn(self) -> None:
        diff = "M\tbig_file.py\nbig_file.py | 200 " + "+" * 100 + "-" * 100
        events = extract_git_events(diff, "p1", "s1")
        if events:
            assert events[0].weight <= 0.9 + 1e-9  # 0.5 + 0.4 max

    def test_weight_formula_for_low_churn(self) -> None:
        diff = "M\tsmall_file.py\nsmall_file.py | 2 +-"
        events = extract_git_events(diff, "p1", "s1")
        if events:
            assert events[0].weight >= 0.5

    def test_events_have_content_hash(self) -> None:
        events = extract_git_events(_SAMPLE_DIFF, "proj1", "sess1")
        assert all(len(e.content_hash) == 64 for e in events)

    def test_intent_in_payload(self) -> None:
        events = extract_git_events(_SAMPLE_DIFF, "proj1", "sess1")
        assert all("intent" in e.payload for e in events)


class TestInferIntentFromPath:
    def test_auth_directory(self) -> None:
        intent = infer_intent_from_path("src/auth/middleware.py")
        assert intent == "authentication"

    def test_routes_directory(self) -> None:
        intent = infer_intent_from_path("src/routes/users.py")
        assert intent in ("API routes", "routes")

    def test_tests_directory(self) -> None:
        intent = infer_intent_from_path("tests/unit/test_storage.py")
        assert intent == "tests"

    def test_migrations_directory(self) -> None:
        intent = infer_intent_from_path("db/migrations/001_initial.sql")
        assert intent == "database migration"

    def test_utils_directory(self) -> None:
        intent = infer_intent_from_path("src/utils/validators.py")
        assert intent == "utilities"

    def test_unknown_path_returns_path(self) -> None:
        intent = infer_intent_from_path("some/unusual/file.py")
        assert intent == "some/unusual/file.py"

    def test_windows_backslash_path(self) -> None:
        intent = infer_intent_from_path("src\\auth\\middleware.py")
        assert intent == "authentication"


class TestCrossReferenceSignals:
    def _make_abandoned_event(self, description: str) -> Event:
        return Event(
            project_id="p1", session_id="s1",
            event_type="APPROACH_ABANDONED",
            payload={"description": description, "rationale": ""},
            content_hash="abc", weight=0.9,
        )

    def _make_git_event(self, path: str) -> Event:
        return Event(
            project_id="p1", session_id="s1",
            event_type="COMPONENT_STATUS",
            payload={"path": path, "description": f"{path} modified"},
            content_hash="def", weight=0.6,
        )

    def test_matching_library_boosts_weight(self) -> None:
        transcript_events = [self._make_abandoned_event("We abandoned the redis approach.")]
        git_events = [self._make_git_event("src/legacy/redis_client.py")]
        result = cross_reference_signals(transcript_events, git_events)
        assert result[0].weight > 0.9

    def test_matching_event_marked_corroborated(self) -> None:
        transcript_events = [self._make_abandoned_event("We abandoned the redis approach.")]
        git_events = [self._make_git_event("src/legacy/redis_client.py")]
        result = cross_reference_signals(transcript_events, git_events)
        assert result[0].payload.get("git_corroborated") is True

    def test_no_match_weight_unchanged(self) -> None:
        transcript_events = [self._make_abandoned_event("We abandoned the approach.")]
        git_events = [self._make_git_event("src/routes/users.py")]
        original_weight = transcript_events[0].weight
        result = cross_reference_signals(transcript_events, git_events)
        assert result[0].weight == original_weight

    def test_non_abandoned_events_not_boosted(self) -> None:
        event = Event(
            project_id="p1", session_id="s1",
            event_type="DECISION",
            payload={"description": "We decided to use redis."},
            content_hash="abc", weight=0.9,
        )
        git_events = [self._make_git_event("src/redis_client.py")]
        result = cross_reference_signals([event], git_events)
        assert result[0].weight == 0.9  # unchanged

    def test_weight_capped_at_two(self) -> None:
        event = self._make_abandoned_event("We abandoned the redis approach.")
        event.weight = 1.95
        git_events = [self._make_git_event("src/redis.py")]
        result = cross_reference_signals([event], git_events)
        assert result[0].weight <= 2.0
