"""Tests for the injection template engine."""
import pytest
from cognikernel.injection.template import (
    InjectionContext,
    _render_active_thread,
    _render_components,
    _render_decisions,
    _render_graveyard,
    _render_hard_constraints,
    _render_header,
    _render_summary,
    count_tokens_accurate,
    generate_summary,
    render_injection,
    render_with_budget_enforcement,
)
from cognikernel.storage.events import Event


def _event(
    event_type: str = "DECISION",
    description: str = "Use SQLite.",
    rationale: str = "",
    session_id: str = "sess1",
    content_hash: str | None = None,
    **payload_extra,
) -> Event:
    payload = {"description": description, "rationale": rationale, **payload_extra}
    return Event(
        project_id="p1", session_id=session_id,
        event_type=event_type, payload=payload,
        content_hash=content_hash or description[:32].ljust(64, "0"),
        weight=1.0,
    )


def _make_ctx(**overrides) -> InjectionContext:
    defaults = dict(
        project_name="myproject",
        session_number=3,
        total_sessions=5,
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


# ── header ────────────────────────────────────────────────────────────────────

class TestRenderHeader:
    def test_contains_project_name(self) -> None:
        ctx = _make_ctx(project_name="myrepo")
        assert "myrepo" in _render_header(ctx)

    def test_contains_session_numbers(self) -> None:
        ctx = _make_ctx(session_number=3, total_sessions=7)
        header = _render_header(ctx)
        assert "session 3" in header
        assert "of 7" in header

    def test_contains_state_version(self) -> None:
        ctx = _make_ctx(state_version=42)
        assert "v42" in _render_header(ctx)

    def test_auto_generated_marker(self) -> None:
        ctx = _make_ctx()
        assert "auto-generated" in _render_header(ctx)


# ── hard constraints ──────────────────────────────────────────────────────────

class TestRenderHardConstraints:
    def test_empty_returns_empty_string(self) -> None:
        assert _render_hard_constraints([]) == ""

    def test_contains_description_as_bullet(self) -> None:
        c = _event(event_type="CONSTRAINT_HARD", description="No Redis.")
        out = _render_hard_constraints([c])
        assert "- No Redis." in out

    def test_contains_description(self) -> None:
        c = _event(event_type="CONSTRAINT_HARD", description="Never log tokens.")
        out = _render_hard_constraints([c])
        assert "Never log tokens." in out

    def test_rationale_rendered_when_present(self) -> None:
        c = _event(event_type="CONSTRAINT_HARD", description="No Redis.",
                   rationale="Latency requirements.")
        out = _render_hard_constraints([c])
        assert "Latency requirements." in out

    def test_no_rationale_suffix_when_empty(self) -> None:
        c = _event(event_type="CONSTRAINT_HARD", description="No Redis.", rationale="")
        lines = _render_hard_constraints([c]).splitlines()
        bullet = next(l for l in lines if l.startswith("- "))
        assert "No Redis." in bullet
        assert bullet.count("—") == 0

    def test_stable_sorted_by_content_hash(self) -> None:
        a = _event(event_type="CONSTRAINT_HARD", description="AAA", content_hash="z" * 64)
        b = _event(event_type="CONSTRAINT_HARD", description="BBB", content_hash="a" * 64)
        out = _render_hard_constraints([a, b])
        # b.content_hash ("aaa...") sorts before a.content_hash ("zzz...")
        assert out.index("BBB") < out.index("AAA")

    def test_section_header_present(self) -> None:
        c = _event(event_type="CONSTRAINT_HARD", description="No Redis.")
        out = _render_hard_constraints([c])
        assert "Hard constraints" in out


# ── graveyard ─────────────────────────────────────────────────────────────────

class TestRenderGraveyard:
    def test_empty_returns_empty_string(self) -> None:
        assert _render_graveyard([]) == ""

    def test_arrow_separator_with_reason(self) -> None:
        e = _event(event_type="APPROACH_ABANDONED_DO_NOT_RETRY",
                   description="Redis caching", rationale="Too slow.")
        out = _render_graveyard([e])
        assert "Redis caching -> Too slow." in out

    def test_no_arrow_when_no_reason(self) -> None:
        e = _event(event_type="APPROACH_ABANDONED_DO_NOT_RETRY",
                   description="Redis caching", rationale="")
        out = _render_graveyard([e])
        assert "→" not in out

    def test_stable_sorted_by_hash(self) -> None:
        a = _event(event_type="APPROACH_ABANDONED_DO_NOT_RETRY",
                   description="Alpha", content_hash="z" * 64)
        b = _event(event_type="APPROACH_ABANDONED_DO_NOT_RETRY",
                   description="Beta", content_hash="a" * 64)
        out = _render_graveyard([a, b])
        assert out.index("Beta") < out.index("Alpha")

    def test_section_header_present(self) -> None:
        e = _event(event_type="APPROACH_ABANDONED_DO_NOT_RETRY", description="Redis")
        assert "Do not retry" in _render_graveyard([e])


# ── components ────────────────────────────────────────────────────────────────

class TestRenderComponents:
    def test_empty_returns_empty_string(self) -> None:
        assert _render_components([]) == ""

    def test_path_in_output(self) -> None:
        c = _event(event_type="COMPONENT_STATUS", description="Modified",
                   path="src/auth/middleware.py")
        out = _render_components([c])
        assert "src/auth/middleware.py" in out

    def test_intent_rendered_when_present(self) -> None:
        c = _event(event_type="COMPONENT_STATUS", description="Modified",
                   path="src/auth/middleware.py", intent="authentication")
        out = _render_components([c])
        assert "authentication" in out
        assert "—" in out

    def test_no_dash_when_no_intent(self) -> None:
        c = _event(event_type="COMPONENT_STATUS", description="Modified",
                   path="src/auth.py")
        out = _render_components([c])
        assert "—" not in out

    def test_section_header_present(self) -> None:
        c = _event(event_type="COMPONENT_STATUS", description="Modified",
                   path="src/auth.py")
        assert "Component state" in _render_components([c])


# ── decisions ─────────────────────────────────────────────────────────────────

class TestRenderDecisions:
    def test_empty_returns_empty_string(self) -> None:
        assert _render_decisions([]) == ""

    def test_numbered_list(self) -> None:
        d = _event(event_type="DECISION", description="Use SQLite.")
        out = _render_decisions([d])
        assert "1." in out

    def test_description_in_output(self) -> None:
        d = _event(event_type="DECISION", description="Use SQLite.")
        assert "Use SQLite." in _render_decisions([d])

    def test_rationale_rendered(self) -> None:
        d = _event(event_type="DECISION", description="Use SQLite.",
                   rationale="Local-first.")
        out = _render_decisions([d])
        assert "Local-first." in out

    def test_session_id_in_output(self) -> None:
        d = _event(event_type="DECISION", description="Use SQLite.", session_id="mysess")
        out = _render_decisions([d])
        assert "mysess" in out

    def test_multiple_decisions_numbered_sequentially(self) -> None:
        events = [_event(description=f"Decision {i}") for i in range(3)]
        out = _render_decisions(events)
        assert "1." in out
        assert "2." in out
        assert "3." in out

    def test_section_header_present(self) -> None:
        d = _event(event_type="DECISION", description="Use SQLite.")
        assert "Key decisions" in _render_decisions([d])


# ── active thread ─────────────────────────────────────────────────────────────

class TestRenderActiveThread:
    def test_empty_returns_empty_string(self) -> None:
        assert _render_active_thread([]) == ""

    def test_description_present(self) -> None:
        t = _event(event_type="THREAD_OPEN", description="Implementing auth middleware.")
        out = _render_active_thread([t])
        assert "Implementing auth middleware." in out

    def test_state_rendered_when_present(self) -> None:
        t = _event(event_type="THREAD_OPEN", description="Auth",
                   state="50% complete")
        out = _render_active_thread([t])
        assert "50% complete" in out

    def test_next_steps_rendered_when_present(self) -> None:
        t = _event(event_type="THREAD_OPEN", description="Auth",
                   next_steps="Add JWT validation")
        out = _render_active_thread([t])
        assert "Add JWT validation" in out

    def test_only_first_thread_rendered(self) -> None:
        t1 = _event(event_type="THREAD_OPEN", description="Thread A")
        t2 = _event(event_type="THREAD_OPEN", description="Thread B")
        out = _render_active_thread([t1, t2])
        assert "Thread A" in out
        assert "Thread B" not in out

    def test_section_header_present(self) -> None:
        t = _event(event_type="THREAD_OPEN", description="Auth")
        assert "Active thread" in _render_active_thread([t])


# ── summary ───────────────────────────────────────────────────────────────────

class TestRenderSummary:
    def test_empty_string_returns_empty(self) -> None:
        assert _render_summary("") == ""

    def test_non_empty_includes_section_header(self) -> None:
        out = _render_summary("A Python project.")
        assert "Summary" in out
        assert "A Python project." in out


# ── render_injection ──────────────────────────────────────────────────────────

class TestRenderInjection:
    def test_returns_non_empty_string(self) -> None:
        ctx = _make_ctx()
        out = render_injection(ctx)
        assert isinstance(out, str) and len(out) > 0

    def test_header_always_present(self) -> None:
        ctx = _make_ctx()
        assert "auto-generated" in render_injection(ctx)

    def test_empty_sections_omitted(self) -> None:
        ctx = _make_ctx()
        out = render_injection(ctx)
        assert "Hard constraints" not in out
        assert "Do not retry" not in out

    def test_sections_appear_in_correct_order(self) -> None:
        ctx = _make_ctx(
            hard_constraints=[_event(event_type="CONSTRAINT_HARD", description="No Redis.")],
            decisions=[_event(description="Use SQLite.")],
            active_threads=[_event(event_type="THREAD_OPEN", description="Auth work.")],
        )
        out = render_injection(ctx)
        hc_pos     = out.index("Hard constraints")
        thread_pos = out.index("Active thread")
        dec_pos    = out.index("Key decisions")
        # Active thread promoted to position 2 (before decisions)
        assert hc_pos < thread_pos < dec_pos

    def test_sections_separated_by_double_newline(self) -> None:
        ctx = _make_ctx(
            hard_constraints=[_event(event_type="CONSTRAINT_HARD", description="A.")],
            decisions=[_event(description="B.")],
        )
        out = render_injection(ctx)
        assert "\n\n" in out

    def test_all_sections_rendered_when_present(self) -> None:
        ctx = _make_ctx(
            hard_constraints=[_event(event_type="CONSTRAINT_HARD", description="HC.")],
            graveyard=[_event(event_type="APPROACH_ABANDONED_DO_NOT_RETRY",
                              description="GY.")],
            components=[_event(event_type="COMPONENT_STATUS", description="CP.",
                               path="src/x.py")],
            decisions=[_event(description="DEC.")],
            active_threads=[_event(event_type="THREAD_OPEN", description="THR.")],
            summary_text="Summary here.",
        )
        out = render_injection(ctx)
        for marker in ("Hard constraints", "Do not retry", "Component state",
                       "Key decisions", "Active thread", "Summary"):
            assert marker in out


# ── generate_summary ──────────────────────────────────────────────────────────

class TestGenerateSummary:
    def test_python_project_detected(self) -> None:
        ctx = _make_ctx(
            components=[_event(event_type="COMPONENT_STATUS", description="M",
                               path="src/app.py")],
        )
        assert "Python" in generate_summary(ctx)

    def test_typescript_project_detected(self) -> None:
        ctx = _make_ctx(
            components=[_event(event_type="COMPONENT_STATUS", description="M",
                               path="src/app.ts")],
        )
        assert "TypeScript" in generate_summary(ctx)

    def test_node_detected_via_package_json(self) -> None:
        ctx = _make_ctx(
            components=[_event(event_type="COMPONENT_STATUS", description="M",
                               path="package.json")],
        )
        assert "Node" in generate_summary(ctx)

    def test_active_thread_included(self) -> None:
        ctx = _make_ctx(
            active_threads=[_event(event_type="THREAD_OPEN",
                                   description="implementing user auth")],
        )
        summary = generate_summary(ctx)
        assert "implementing user auth" in summary

    def test_in_flux_warning_included(self) -> None:
        ctx = _make_ctx(
            components=[_event(event_type="COMPONENT_STATUS", description="M",
                               path="src/middleware.py", status="in_flux")],
        )
        assert "middleware.py" in generate_summary(ctx)

    def test_no_components_returns_default(self) -> None:
        ctx = _make_ctx()
        assert generate_summary(ctx) == "Project state is being established."


# ── count_tokens_accurate ─────────────────────────────────────────────────────

class TestCountTokensAccurate:
    def test_returns_positive_int(self) -> None:
        assert count_tokens_accurate("Hello world") > 0

    def test_longer_text_costs_more(self) -> None:
        short = count_tokens_accurate("Hi")
        long  = count_tokens_accurate("Hi " * 100)
        assert long > short

    def test_empty_string_at_least_zero(self) -> None:
        assert count_tokens_accurate("") >= 0


# ── render_with_budget_enforcement ───────────────────────────────────────────

class TestRenderWithBudgetEnforcement:
    def test_result_within_budget(self) -> None:
        decisions = [_event(description=f"Decision {i} " + "x" * 50)
                     for i in range(20)]
        ctx = _make_ctx(decisions=decisions, token_budget=100)
        out = render_with_budget_enforcement(ctx)
        assert count_tokens_accurate(out) <= 100 + 20  # allow small rounding

    def test_does_not_drop_hard_constraints(self) -> None:
        hc = [_event(event_type="CONSTRAINT_HARD", description="Never use Redis.")]
        decisions = [_event(description=f"Dec {i} " + "x" * 60) for i in range(30)]
        ctx = _make_ctx(hard_constraints=hc, decisions=decisions, token_budget=80)
        out = render_with_budget_enforcement(ctx)
        assert "Never use Redis." in out

    def test_does_not_modify_original_ctx(self) -> None:
        decisions = [_event(description=f"Decision {i}") for i in range(5)]
        ctx = _make_ctx(decisions=decisions, token_budget=30)
        original_len = len(ctx.decisions)
        render_with_budget_enforcement(ctx)
        assert len(ctx.decisions) == original_len

    def test_no_drop_when_within_budget(self) -> None:
        ctx = _make_ctx(
            decisions=[_event(description="Short.")],
            token_budget=800,
        )
        out = render_with_budget_enforcement(ctx)
        assert "Short." in out
