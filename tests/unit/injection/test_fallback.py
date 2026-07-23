"""Tests for fallback rendering."""
import pytest
from cognikernel.injection.fallback import (
    ProjectionCorruptedError,
    _fallback_corrupted,
    _fallback_empty_projection,
    _fallback_uninitialized,
    render_or_fallback,
)
from cognikernel.injection.template import InjectionContext
from cognikernel.storage.events import Event


def _make_ctx(**overrides) -> InjectionContext:
    defaults = dict(
        project_name="myproject",
        session_number=1,
        total_sessions=1,
        state_version=1,
        hard_constraints=[],
        graveyard=[],
        components=[],
        decisions=[],
        active_threads=[],
        summary_text="",
        token_budget=800,
    )
    defaults.update(overrides)
    return InjectionContext(**defaults)


def _decision(description: str = "Use SQLite.") -> Event:
    return Event(
        project_id="p1", session_id="s1",
        event_type="DECISION",
        payload={"description": description, "rationale": ""},
        content_hash=description[:32].ljust(64, "0"),
        weight=1.0,
    )


class TestFallbackUninitialized:
    def test_contains_cognikernel_init(self) -> None:
        out = _fallback_uninitialized()
        assert "cognikernel init" in out

    def test_contains_header_marker(self) -> None:
        assert "auto-generated" in _fallback_uninitialized()

    def test_never_empty(self) -> None:
        assert len(_fallback_uninitialized()) > 0


class TestFallbackEmptyProjection:
    def test_contains_project_name(self) -> None:
        ctx = _make_ctx(project_name="coolproject")
        out = _fallback_empty_projection(ctx)
        assert "coolproject" in out

    def test_mentions_first_session(self) -> None:
        ctx = _make_ctx()
        out = _fallback_empty_projection(ctx)
        assert "first session" in out.lower()

    def test_contains_header_marker(self) -> None:
        ctx = _make_ctx()
        assert "auto-generated" in _fallback_empty_projection(ctx)

    def test_never_empty(self) -> None:
        ctx = _make_ctx()
        assert len(_fallback_empty_projection(ctx)) > 0


class TestFallbackCorrupted:
    def test_contains_project_name(self) -> None:
        ctx = _make_ctx(project_name="brokenproject")
        out = _fallback_corrupted(ctx)
        assert "brokenproject" in out

    def test_mentions_doctor_command(self) -> None:
        ctx = _make_ctx()
        assert "cognikernel doctor" in _fallback_corrupted(ctx)

    def test_contains_header_marker(self) -> None:
        ctx = _make_ctx()
        assert "auto-generated" in _fallback_corrupted(ctx)

    def test_never_empty(self) -> None:
        ctx = _make_ctx()
        assert len(_fallback_corrupted(ctx)) > 0


class TestRenderOrFallback:
    def test_none_ctx_returns_uninitialized(self) -> None:
        out = render_or_fallback(None)
        assert "cognikernel init" in out

    def test_empty_ctx_returns_first_session_message(self) -> None:
        ctx = _make_ctx()
        out = render_or_fallback(ctx)
        assert "first session" in out.lower()

    def test_valid_ctx_returns_full_block(self) -> None:
        ctx = _make_ctx(decisions=[_decision("Use SQLite.")])
        out = render_or_fallback(ctx)
        assert "Use SQLite." in out

    def test_never_returns_empty_string(self) -> None:
        for ctx in [None, _make_ctx(), _make_ctx(decisions=[_decision()])]:
            assert len(render_or_fallback(ctx)) > 0

    def test_corrupted_error_returns_corrupted_fallback(self) -> None:
        from unittest.mock import patch
        ctx = _make_ctx(decisions=[_decision("Use SQLite.")])
        with patch(
            "cognikernel.injection.template.render_with_budget_enforcement",
            side_effect=ProjectionCorruptedError("disk read error"),
        ):
            out = render_or_fallback(ctx)
        assert "cognikernel doctor" in out

    def test_hard_constraint_ctx_renders_full_block(self) -> None:
        hc = Event(
            project_id="p1", session_id="s1",
            event_type="CONSTRAINT_HARD",
            payload={"description": "No Redis.", "rationale": ""},
            content_hash="a" * 64, weight=1.0,
        )
        ctx = _make_ctx(hard_constraints=[hc])
        out = render_or_fallback(ctx)
        assert "No Redis." in out
