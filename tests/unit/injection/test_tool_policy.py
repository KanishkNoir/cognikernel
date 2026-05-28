"""Tests for the Tool Policy injection section (Stage B-1).

Verifies that the section only renders under `hook_policy='strict'`, contains
the language Claude needs to understand the deny rules, and stays absent
under advisory (legacy) policy.
"""
from __future__ import annotations

import pytest

from memlora.injection.template import (
    InjectionContext,
    _render_tool_policy,
    render_injection,
)


def _ctx(**overrides) -> InjectionContext:
    defaults = dict(
        project_name="proj",
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


class TestRenderToolPolicy:
    def test_advisory_returns_empty(self) -> None:
        ctx = _ctx(hook_policy="advisory")
        assert _render_tool_policy(ctx) == ""

    def test_default_policy_is_advisory(self) -> None:
        """Backwards-compat invariant: existing callers that don't set policy
        get the legacy behavior (no Tool Policy section)."""
        ctx = _ctx()
        assert _render_tool_policy(ctx) == ""

    def test_strict_renders_section_with_header(self) -> None:
        ctx = _ctx(hook_policy="strict")
        section = _render_tool_policy(ctx)
        assert section.startswith("### Tool policy")

    def test_strict_mentions_deny_and_skeleton(self) -> None:
        ctx = _ctx(hook_policy="strict")
        section = _render_tool_policy(ctx)
        assert "DENIED" in section
        assert "skeleton" in section.lower()

    def test_strict_mentions_retry_window(self) -> None:
        ctx = _ctx(hook_policy="strict", retry_window_seconds=60)
        section = _render_tool_policy(ctx)
        assert "60 seconds" in section

    def test_retry_window_value_is_dynamic(self) -> None:
        """If the config has a non-default retry window, the text reflects it."""
        ctx = _ctx(hook_policy="strict", retry_window_seconds=90)
        section = _render_tool_policy(ctx)
        assert "90 seconds" in section
        assert "60 seconds" not in section

    def test_strict_mentions_reread_rule(self) -> None:
        ctx = _ctx(hook_policy="strict")
        section = _render_tool_policy(ctx)
        # The re-read protection is a universal invariant — must be in the text.
        assert "Re-read" in section or "re-reads" in section.lower()


class TestSectionAppearsInFullInjection:
    def test_strict_injection_contains_tool_policy(self) -> None:
        ctx = _ctx(hook_policy="strict")
        block = render_injection(ctx)
        assert "### Tool policy" in block

    def test_advisory_injection_omits_tool_policy(self) -> None:
        ctx = _ctx(hook_policy="advisory")
        block = render_injection(ctx)
        assert "Tool policy" not in block

    def test_tool_policy_appears_before_hard_constraints(self) -> None:
        """Positioning matters: Tool Policy must come BEFORE the first deny
        message Claude is likely to see, which is right after the header."""
        from memlora.storage.events import Event

        ctx = _ctx(
            hook_policy="strict",
            hard_constraints=[Event(
                project_id="p", session_id="s",
                event_type="CONSTRAINT_HARD",
                payload={"description": "Use PostgreSQL"},
                content_hash="h" * 64, weight=1.0,
            )],
        )
        block = render_injection(ctx)
        tp_pos = block.index("### Tool policy")
        hc_pos = block.index("### Hard constraints")
        assert tp_pos < hc_pos
