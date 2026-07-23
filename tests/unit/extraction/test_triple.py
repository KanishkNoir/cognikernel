"""Tests for CKL V2 triple extraction (cognikernel.extraction.triple)."""
import pytest
from cognikernel.extraction.triple import (
    OP_IMPL,
    OP_NEG,
    OP_NULL,
    augment_with_triple,
    extract_triple,
)
from cognikernel.storage.events import Event


def _event(
    event_type: str,
    description: str,
    rationale: str = "",
    content_hash: str | None = None,
) -> Event:
    payload: dict = {"description": description, "rationale": rationale}
    return Event(
        project_id="p1",
        session_id="s1",
        event_type=event_type,
        payload=payload,
        content_hash=content_hash or description[:32].ljust(64, "0"),
        weight=1.0,
    )


# ── negation / CONSTRAINT_HARD ────────────────────────────────────────────────

class TestNegationTriple:
    def test_never_verb(self) -> None:
        t = extract_triple("never issue SQL DELETE", "CONSTRAINT_HARD")
        assert t is not None
        assert t["operator"] == OP_NEG
        assert t["subject"] == ""
        assert "SQL DELETE" in t["object"]

    def test_do_not_verb(self) -> None:
        t = extract_triple("do not add created_at to responses", "CONSTRAINT_HARD")
        assert t is not None
        assert t["operator"] == OP_NEG
        assert "created_at" in t["object"]

    def test_must_not_verb(self) -> None:
        t = extract_triple("must not expose passwords in logs", "CONSTRAINT_HARD")
        assert t is not None
        assert t["operator"] == OP_NEG
        assert "expose passwords" in t["object"]

    def test_filler_prefix_stripped(self) -> None:
        with_filler = extract_triple("we will never issue SQL DELETE", "CONSTRAINT_HARD")
        without     = extract_triple("never issue SQL DELETE", "CONSTRAINT_HARD")
        assert with_filler is not None
        assert without is not None
        assert with_filler["object"] == without["object"]

    def test_cannot_verb(self) -> None:
        t = extract_triple("cannot use global mutable state", "CONSTRAINT_HARD")
        assert t is not None
        assert t["operator"] == OP_NEG

    def test_object_compacted_at_break(self) -> None:
        # Object with a comma — should compact to the part before the comma
        t = extract_triple("never use DELETE, only soft-delete", "CONSTRAINT_HARD")
        assert t is not None
        assert "," not in t["object"]

    def test_object_truncated_to_cap(self) -> None:
        long_action = "x" * 100
        t = extract_triple(f"never {long_action}", "CONSTRAINT_HARD")
        assert t is not None
        assert len(t["object"]) <= 35

    def test_soft_constraint_returns_none(self) -> None:
        assert extract_triple("never do X", "CONSTRAINT_SOFT") is None

    def test_empty_object_returns_none(self) -> None:
        assert extract_triple("never", "CONSTRAINT_HARD") is None

    def test_empty_description_returns_none(self) -> None:
        assert extract_triple("", "CONSTRAINT_HARD") is None


# ── implication / DECISION ────────────────────────────────────────────────────

class TestImplicationTriple:
    def test_uses_verb(self) -> None:
        t = extract_triple("pagination uses page+page_size", "DECISION")
        assert t is not None
        assert t["operator"] == OP_IMPL
        assert t["subject"] == "pagination"
        assert "page+page_size" in t["object"]

    def test_use_for_pattern(self) -> None:
        t = extract_triple("use SQLite for local storage", "DECISION")
        assert t is not None
        assert t["operator"] == OP_IMPL
        assert "SQLite" in t["subject"]
        assert "local storage" in t["object"]

    def test_use_plain_pattern(self) -> None:
        t = extract_triple("use Redis for caching", "DECISION")
        assert t is not None
        assert t["operator"] == OP_IMPL
        assert "Redis" in t["subject"]

    def test_uses_with_trailing_dash_content_compacted(self) -> None:
        t = extract_triple("pagination uses page+page_size — max 100 server-side", "DECISION")
        assert t is not None
        assert "page+page_size" in t["object"]
        assert "max" not in t["object"]

    def test_via_verb(self) -> None:
        t = extract_triple("auth via JWT tokens", "DECISION")
        assert t is not None
        assert t["operator"] == OP_IMPL
        assert "auth" in t["subject"]

    def test_constraint_hard_does_not_use_implication(self) -> None:
        # CONSTRAINT_HARD should not be parsed as implication
        t = extract_triple("pagination uses page+page_size", "CONSTRAINT_HARD")
        # The negation parser should find nothing (no negation verb)
        assert t is None

    def test_unrecognised_decision_pattern_returns_none(self) -> None:
        t = extract_triple("chosen FastAPI because team knows it", "DECISION")
        assert t is None


# ── null / APPROACH_ABANDONED ─────────────────────────────────────────────────

class TestNullTriple:
    def test_em_dash_separator(self) -> None:
        t = extract_triple("celery — broker dependency", "APPROACH_ABANDONED_DO_NOT_RETRY")
        assert t is not None
        assert t["operator"] == OP_NULL
        assert t["subject"] == "celery"
        assert t["object"] == ""

    def test_hyphen_separator(self) -> None:
        t = extract_triple("global state - race conditions", "APPROACH_ABANDONED_DO_NOT_RETRY")
        assert t is not None
        assert t["operator"] == OP_NULL
        assert "global state" in t["subject"]

    def test_short_description_no_separator(self) -> None:
        t = extract_triple("celery", "APPROACH_ABANDONED_DO_NOT_RETRY")
        assert t is not None
        assert t["operator"] == OP_NULL
        assert t["subject"] == "celery"

    def test_approach_abandoned_type_also_handled(self) -> None:
        t = extract_triple("celery — broker", "APPROACH_ABANDONED")
        assert t is not None
        assert t["operator"] == OP_NULL

    def test_long_description_no_separator_returns_none(self) -> None:
        long_desc = "x" * 60
        t = extract_triple(long_desc, "APPROACH_ABANDONED_DO_NOT_RETRY")
        assert t is None


# ── unsupported event types ───────────────────────────────────────────────────

class TestUnsupportedTypes:
    @pytest.mark.parametrize("etype", [
        "COMPONENT_STATUS",
        "CONSTRAINT_SOFT",
        "THREAD_OPEN",
        "THREAD_CLOSE",
    ])
    def test_unsupported_event_type_returns_none(self, etype: str) -> None:
        assert extract_triple("never use global state", etype) is None


# ── augment_with_triple ───────────────────────────────────────────────────────

class TestAugmentWithTriple:
    def test_adds_triple_key_when_pattern_found(self) -> None:
        e = _event("CONSTRAINT_HARD", "never issue SQL DELETE")
        augment_with_triple(e)
        assert "triple" in e.payload
        assert e.payload["triple"]["operator"] == OP_NEG

    def test_no_triple_key_when_no_pattern(self) -> None:
        e = _event("CONSTRAINT_HARD", "something unclear that has no negation verb")
        augment_with_triple(e)
        assert "triple" not in e.payload

    def test_noop_for_component_status(self) -> None:
        e = _event("COMPONENT_STATUS", "crud.py · MODIFIED — in_flux")
        augment_with_triple(e)
        assert "triple" not in e.payload

    def test_idempotent(self) -> None:
        e = _event("CONSTRAINT_HARD", "never issue SQL DELETE")
        augment_with_triple(e)
        first = e.payload["triple"].copy()
        augment_with_triple(e)
        assert e.payload["triple"] == first

    def test_decision_triple_added(self) -> None:
        e = _event("DECISION", "pagination uses page+page_size")
        augment_with_triple(e)
        assert "triple" in e.payload
        assert e.payload["triple"]["operator"] == OP_IMPL

    def test_graveyard_triple_added(self) -> None:
        e = _event("APPROACH_ABANDONED_DO_NOT_RETRY", "celery — broker dependency")
        augment_with_triple(e)
        assert "triple" in e.payload
        assert e.payload["triple"]["operator"] == OP_NULL
