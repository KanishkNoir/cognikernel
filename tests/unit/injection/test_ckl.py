"""Tests for CKL (CogniKernel Language) V1 and V2 rendering."""
from memlora.injection.ckl import CKL_LEGEND, CKL_OPS_LEGEND, render_event_ckl
from memlora.injection.template import InjectionContext, render_injection
from memlora.storage.events import Event


def _event(
    event_type: str = "CONSTRAINT_HARD",
    description: str = "we will never issue SQL DELETE",
    rationale: str = "",
    session_id: str = "sess1",
    content_hash: str | None = None,
) -> Event:
    payload = {"description": description, "rationale": rationale}
    return Event(
        project_id="p1",
        session_id=session_id,
        event_type=event_type,
        payload=payload,
        content_hash=content_hash or description[:32].ljust(64, "0"),
        weight=1.0,
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


# ── render_event_ckl unit tests ───────────────────────────────────────────────

class TestRenderEventCkl:
    def test_with_rationale_uses_hash_separator(self) -> None:
        e = _event(description="no SQL DELETE", rationale="recoverability")
        out = render_event_ckl(e, "CSTR")
        assert out == "CSTR: no SQL DELETE  # recoverability"

    def test_without_rationale_omits_hash(self) -> None:
        e = _event(description="no SQL DELETE", rationale="")
        out = render_event_ckl(e, "CSTR")
        assert out == "CSTR: no SQL DELETE"
        assert "#" not in out

    def test_description_truncated_at_desc_cap(self) -> None:
        long_desc = "x" * 250
        e = _event(description=long_desc)
        out = render_event_ckl(e, "DEC", desc_cap=100)
        # "DEC: " + 100 chars = 105 chars total
        assert len(out) == 105
        assert out.startswith("DEC: ")

    def test_rationale_truncated_at_rationale_cap(self) -> None:
        e = _event(description="short", rationale="y" * 200)
        out = render_event_ckl(e, "DEAD", rationale_cap=35)
        # rationale truncated to 35 chars
        assert "  # " + "y" * 35 in out

    def test_prefix_is_used_verbatim(self) -> None:
        e = _event(description="celery", rationale="broker")
        assert render_event_ckl(e, "DEAD").startswith("DEAD: ")
        assert render_event_ckl(e, "CSTR").startswith("CSTR: ")
        assert render_event_ckl(e, "DEC").startswith("DEC: ")


# ── render_injection with ckl_mode ────────────────────────────────────────────

class TestRenderInjectionCklMode:
    def test_ckl_mode_true_emits_legend(self) -> None:
        ctx = _ctx(
            ckl_mode=True,
            hard_constraints=[_event(description="no DELETE")],
        )
        block = render_injection(ctx)
        assert CKL_LEGEND in block

    def test_ckl_mode_false_omits_legend(self) -> None:
        ctx = _ctx(
            ckl_mode=False,
            hard_constraints=[_event(description="no DELETE")],
        )
        block = render_injection(ctx)
        assert CKL_LEGEND not in block

    def test_ckl_mode_renders_constraint_prefix(self) -> None:
        ctx = _ctx(
            ckl_mode=True,
            hard_constraints=[_event(description="no DELETE", rationale="recoverability")],
        )
        block = render_injection(ctx)
        assert "CSTR: no DELETE  # recoverability" in block
        # The prose bullet form must NOT appear
        assert "- no DELETE — recoverability" not in block

    def test_ckl_mode_renders_decision_prefix(self) -> None:
        ctx = _ctx(
            ckl_mode=True,
            decisions=[_event(event_type="DECISION", description="use SQLite", rationale="local")],
        )
        block = render_injection(ctx)
        assert "DEC: use SQLite  # local" in block

    def test_ckl_mode_renders_graveyard_prefix(self) -> None:
        ctx = _ctx(
            ckl_mode=True,
            graveyard=[_event(
                event_type="APPROACH_ABANDONED_DO_NOT_RETRY",
                description="celery",
                rationale="broker dependency",
            )],
        )
        block = render_injection(ctx)
        assert "DEAD: celery  # broker dependency" in block

    def test_ckl_mode_false_keeps_prose_rendering(self) -> None:
        """Default (non-CKL) behaviour must remain unchanged for backwards compatibility."""
        ctx = _ctx(
            hard_constraints=[_event(description="no DELETE", rationale="recoverability")],
            decisions=[_event(event_type="DECISION", description="use SQLite", rationale="local")],
        )
        block = render_injection(ctx)
        assert "- no DELETE — recoverability" in block
        assert "CSTR:" not in block
        assert "DEC:" not in block


# ── render_event_ckl V2 unit tests ───────────────────────────────────────────

def _event_with_triple(
    event_type: str = "CONSTRAINT_HARD",
    description: str = "no DELETE",
    rationale: str = "",
    triple: dict | None = None,
    session_id: str = "sess1",
) -> Event:
    payload: dict = {"description": description, "rationale": rationale}
    if triple is not None:
        payload["triple"] = triple
    return Event(
        project_id="p1",
        session_id=session_id,
        event_type=event_type,
        payload=payload,
        content_hash=description[:32].ljust(64, "0"),
        weight=1.0,
    )


class TestRenderEventCklV2:
    def test_negation_triple_renders_op_before_object(self) -> None:
        e = _event_with_triple(
            triple={"operator": "¬", "subject": "", "object": "SQL DELETE"},
            rationale="recoverability",
        )
        out = render_event_ckl(e, "CSTR", v2=True)
        assert out == "CSTR: ¬ SQL DELETE  # recoverability"

    def test_implication_triple_subject_op_object(self) -> None:
        e = _event_with_triple(
            event_type="DECISION",
            triple={"operator": "→", "subject": "pagination", "object": "page+page_size"},
        )
        out = render_event_ckl(e, "DEC", v2=True)
        assert out == "DEC: pagination → page+page_size"

    def test_null_triple_subject_op_no_object(self) -> None:
        e = _event_with_triple(
            event_type="APPROACH_ABANDONED_DO_NOT_RETRY",
            triple={"operator": "∅", "subject": "celery", "object": ""},
            rationale="broker dep",
        )
        out = render_event_ckl(e, "DEAD", v2=True)
        assert out == "DEAD: celery ∅  # broker dep"

    def test_v2_fallback_when_no_triple_in_payload(self) -> None:
        e = _event_with_triple(description="no DELETE", rationale="safety", triple=None)
        out = render_event_ckl(e, "CSTR", v2=True)
        # Must fall back to V1 prose
        assert out == "CSTR: no DELETE  # safety"
        assert "¬" not in out

    def test_v2_false_ignores_triple(self) -> None:
        e = _event_with_triple(
            triple={"operator": "¬", "subject": "", "object": "SQL DELETE"},
            description="no DELETE",
            rationale="safety",
        )
        out = render_event_ckl(e, "CSTR", v2=False)
        # Should render V1 prose even though triple is present
        assert "¬" not in out
        assert "no DELETE" in out

    def test_triple_with_both_subject_and_object(self) -> None:
        e = _event_with_triple(
            event_type="DECISION",
            triple={"operator": "→", "subject": "auth", "object": "JWT"},
        )
        out = render_event_ckl(e, "DEC", v2=True)
        assert out == "DEC: auth → JWT"

    def test_triple_subject_only(self) -> None:
        e = _event_with_triple(
            event_type="DECISION",
            triple={"operator": "→", "subject": "Redis", "object": ""},
        )
        out = render_event_ckl(e, "DEC", v2=True)
        assert out == "DEC: Redis →"

    def test_rationale_truncated_to_cap(self) -> None:
        e = _event_with_triple(
            triple={"operator": "¬", "subject": "", "object": "X"},
            rationale="r" * 200,
        )
        out = render_event_ckl(e, "CSTR", v2=True, rationale_cap=10)
        # rationale portion should not exceed cap
        assert out.endswith("  # " + "r" * 10)


# ── ops legend in header ──────────────────────────────────────────────────────

class TestOpsLegendInHeader:
    def test_ops_legend_emitted_when_ckl_v2_true(self) -> None:
        ctx = _ctx(
            ckl_mode=True,
            ckl_v2=True,
            hard_constraints=[_event(description="no DELETE")],
        )
        block = render_injection(ctx)
        assert CKL_OPS_LEGEND in block

    def test_ops_legend_omitted_when_ckl_v2_false(self) -> None:
        ctx = _ctx(
            ckl_mode=True,
            ckl_v2=False,
            hard_constraints=[_event(description="no DELETE")],
        )
        block = render_injection(ctx)
        assert CKL_OPS_LEGEND not in block

    def test_ops_legend_present_without_ckl_mode(self) -> None:
        # ckl_v2=True should add the ops legend even if ckl_mode=False
        ctx = _ctx(
            ckl_mode=False,
            ckl_v2=True,
            hard_constraints=[_event(description="no DELETE")],
        )
        block = render_injection(ctx)
        assert CKL_OPS_LEGEND in block
