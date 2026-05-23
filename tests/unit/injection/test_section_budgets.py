"""Tests for per-section budget enforcement in the injection template."""
from memlora.config import SectionBudgets
from memlora.injection.template import (
    InjectionContext,
    _enforce_section_budget,
    _render_decisions,
    _render_hard_constraints,
    count_tokens_accurate,
    render_injection,
)
from memlora.storage.events import Event


def _event(
    event_type: str = "CONSTRAINT_HARD",
    description: str = "no SQL DELETE",
    rationale: str = "",
    weight: float = 1.0,
    content_hash: str | None = None,
) -> Event:
    payload = {"description": description, "rationale": rationale}
    return Event(
        project_id="p1",
        session_id="s1",
        event_type=event_type,
        payload=payload,
        content_hash=content_hash or (description + str(weight))[:32].ljust(64, "0"),
        weight=weight,
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
        token_budget=2000,
    )
    defaults.update(overrides)
    return InjectionContext(**defaults)


# ── _enforce_section_budget unit tests ────────────────────────────────────────

class TestEnforceSectionBudget:
    def test_returns_all_events_when_under_budget(self) -> None:
        events = [_event(description=f"rule_{i}", weight=1.0) for i in range(3)]
        kept = _enforce_section_budget(events, _render_hard_constraints, budget=1000)
        assert len(kept) == 3

    def test_drops_lowest_weight_first(self) -> None:
        high = _event(description="HIGH_WEIGHT_RULE", weight=3.0, content_hash="h" * 64)
        low  = _event(description="LOW_WEIGHT_RULE",  weight=0.5, content_hash="l" * 64)
        mid  = _event(description="MID_WEIGHT_RULE",  weight=1.5, content_hash="m" * 64)
        # Make filler bytes long enough that rendering all 3 exceeds the tight budget
        filler = "x" * 80
        events = [
            _event(description=f"HIGH {filler}", weight=3.0, content_hash="h" * 64),
            _event(description=f"LOW  {filler}", weight=0.5, content_hash="l" * 64),
            _event(description=f"MID  {filler}", weight=1.5, content_hash="m" * 64),
        ]
        # Budget that fits ~2 events but not all 3
        all_rendered = _render_hard_constraints(events)
        full_tokens = count_tokens_accurate(all_rendered)
        budget = int(full_tokens * 0.7)
        kept = _enforce_section_budget(events, _render_hard_constraints, budget=budget)
        # Lowest weight event must be dropped first
        kept_descs = [e.payload["description"] for e in kept]
        assert not any("LOW" in d for d in kept_descs)

    def test_never_drops_last_event_even_when_over_budget(self) -> None:
        events = [_event(description="x" * 500, weight=1.0)]
        # Budget impossibly tight
        kept = _enforce_section_budget(events, _render_hard_constraints, budget=1)
        assert len(kept) == 1

    def test_empty_input_returns_empty(self) -> None:
        kept = _enforce_section_budget([], _render_hard_constraints, budget=100)
        assert kept == []


# ── render_injection with section_budgets ─────────────────────────────────────

class TestRenderInjectionWithSectionBudgets:
    def test_section_budgets_none_keeps_all_events(self) -> None:
        events = [_event(description=f"rule_{i}", weight=1.0) for i in range(5)]
        ctx = _ctx(section_budgets=None, hard_constraints=events)
        block = render_injection(ctx)
        # All 5 events should appear in the rendered output
        for i in range(5):
            assert f"rule_{i}" in block

    def test_tight_budget_drops_some_events(self) -> None:
        # 5 verbose events
        events = [
            _event(description=f"rule_{i} " + "x" * 100, weight=float(i + 1))
            for i in range(5)
        ]
        # Very tight per-section budget — should drop lower-weight events
        budgets = SectionBudgets(hard_constraints=30)
        ctx = _ctx(section_budgets=budgets, hard_constraints=events)
        block = render_injection(ctx)
        # Highest weight (rule_4) should survive
        assert "rule_4" in block
        # Lowest weight (rule_0) should be dropped
        assert "rule_0" not in block

    def test_section_budget_respected_for_decisions(self) -> None:
        events = [
            _event(event_type="DECISION", description=f"dec_{i} " + "y" * 80, weight=float(i + 1))
            for i in range(4)
        ]
        budgets = SectionBudgets(decisions=40)
        ctx = _ctx(section_budgets=budgets, decisions=events)
        block = render_injection(ctx)
        # Highest-weight decision must survive
        assert "dec_3" in block
