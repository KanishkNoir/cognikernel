"""Tests for token estimation."""
import pytest
from memlora.compression.token_count import estimate_tokens
from memlora.storage.events import Event


def _make_event(**overrides) -> Event:
    defaults = dict(
        project_id="p1", session_id="s1",
        event_type="DECISION",
        payload={"description": "Use SQLite", "rationale": ""},
        content_hash="a" * 64, weight=1.0,
    )
    defaults.update(overrides)
    return Event(**defaults)


class TestEstimateTokens:
    def test_returns_positive_int(self) -> None:
        result = estimate_tokens(_make_event())
        assert isinstance(result, int) and result >= 1

    def test_longer_description_costs_more(self) -> None:
        short = _make_event(payload={"description": "Short.", "rationale": ""})
        long  = _make_event(payload={
            "description": "A much longer description that spans many characters.",
            "rationale": "",
        })
        assert estimate_tokens(long) >= estimate_tokens(short)

    def test_rationale_adds_to_cost(self) -> None:
        no_rationale = _make_event(payload={"description": "Use SQLite", "rationale": ""})
        with_rationale = _make_event(payload={
            "description": "Use SQLite",
            "rationale": "Because it's local-first and doesn't need a server.",
        })
        assert estimate_tokens(with_rationale) > estimate_tokens(no_rationale)

    def test_empty_event_still_returns_at_least_one(self) -> None:
        e = _make_event(payload={})
        assert estimate_tokens(e) >= 1

    def test_affected_files_add_to_cost(self) -> None:
        no_files  = _make_event(payload={"description": "Use SQLite", "rationale": ""})
        with_files = _make_event(payload={
            "description": "Use SQLite", "rationale": "",
            "affected_files": ["src/a.py", "src/b.py", "src/c.py"],
        })
        assert estimate_tokens(with_files) > estimate_tokens(no_files)

    def test_path_field_adds_to_cost(self) -> None:
        no_path   = _make_event(payload={"description": "Use SQLite", "rationale": ""})
        with_path = _make_event(payload={
            "description": "Use SQLite", "rationale": "", "path": "src/auth/middleware.py"
        })
        assert estimate_tokens(with_path) > estimate_tokens(no_path)

    def test_approximation_roughly_four_chars_per_token(self) -> None:
        # "DECISION | Use SQLite" ≈ 21 chars → 5 tokens
        e = _make_event(payload={"description": "Use SQLite", "rationale": ""})
        result = estimate_tokens(e)
        assert 3 <= result <= 15  # loose bounds for approximation

    def test_component_status_event_with_path(self) -> None:
        e = _make_event(
            event_type="COMPONENT_STATUS",
            payload={"path": "src/auth/middleware.py", "description": "Modified"},
        )
        assert estimate_tokens(e) >= 1


class TestSingleCounter:
    """Selection and enforcement must use the one canonical counter."""

    def test_count_tokens_matches_renderer_counter(self) -> None:
        from memlora.compression.token_count import count_tokens
        from memlora.injection.template import count_tokens_accurate
        for s in ("", "Use SQLite for local storage.",
                  "### Hard constraints\n- never log secrets — security"):
            assert count_tokens(s) == count_tokens_accurate(s)
