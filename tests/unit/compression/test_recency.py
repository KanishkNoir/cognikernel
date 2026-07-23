"""Tests for hyperbolic recency decay."""
import pytest
from cognikernel.compression.recency import recency_factor


class TestRecencyFactor:
    def test_zero_sessions_returns_one(self) -> None:
        assert recency_factor(0) == pytest.approx(1.0)

    def test_decays_below_one_for_positive_sessions(self) -> None:
        assert recency_factor(1) < 1.0

    def test_monotonically_decreasing(self) -> None:
        values = [recency_factor(t) for t in range(0, 50)]
        assert values == sorted(values, reverse=True)

    def test_hyperbolic_formula_at_session_8(self) -> None:
        # 1 / (1 + 0.15 * 8) = 1 / 2.2 ≈ 0.4545
        assert recency_factor(8) == pytest.approx(1.0 / 2.2, rel=1e-6)

    def test_hyperbolic_formula_at_session_30(self) -> None:
        assert recency_factor(30) == pytest.approx(1.0 / (1 + 0.15 * 30), rel=1e-6)

    def test_never_reaches_zero_at_50_sessions(self) -> None:
        assert recency_factor(50) > 0.0

    def test_approaches_but_never_reaches_zero(self) -> None:
        assert recency_factor(1000) > 0.0

    def test_negative_sessions_treated_as_zero(self) -> None:
        assert recency_factor(-5) == pytest.approx(1.0)

    def test_custom_alpha_lower_decays_slower(self) -> None:
        slow = recency_factor(10, alpha=0.05)
        fast = recency_factor(10, alpha=0.30)
        assert slow > fast

    def test_custom_alpha_zero_never_decays(self) -> None:
        assert recency_factor(100, alpha=0.0) == pytest.approx(1.0)

    def test_long_tail_slower_than_exponential(self) -> None:
        # At t=50, hyperbolic ≈ 0.12; exponential with same half-life would be near 0
        import math
        hyperbolic = recency_factor(50)
        # exponential with λ=0.087 (half-life ≈8): e^(-0.087*50) ≈ 0.013
        exponential = math.exp(-0.087 * 50)
        assert hyperbolic > exponential
