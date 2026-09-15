"""Unit tests for Time-to-Collision (TTC)."""

from dataclasses import FrozenInstanceError
from math import inf

import pytest

from surrogate_safety_abm.ssm.ttc import TTCResult, compute_ttc


class TestComputeTTC:
    """Unit tests for compute_ttc."""

    def test_typical_approach(self) -> None:
        result = compute_ttc(gap=20.0, relative_speed=10.0)
        assert isinstance(result, TTCResult)
        assert result.ttc_seconds == pytest.approx(2.0)
        assert result.is_converging is True

    def test_zero_relative_speed_returns_inf(self) -> None:
        result = compute_ttc(20.0, 0.0)
        assert result.ttc_seconds == inf
        assert result.is_converging is False

    def test_separating_returns_inf(self) -> None:
        result = compute_ttc(20.0, -5.0)
        assert result.ttc_seconds == inf
        assert result.is_converging is False

    def test_zero_gap_implies_imminent_collision(self) -> None:
        result = compute_ttc(0.0, 10.0)
        assert result.ttc_seconds == 0.0
        assert result.is_converging is True

    def test_negative_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="gap"):
            compute_ttc(-1.0, 5.0)

    def test_result_is_frozen(self) -> None:
        r = compute_ttc(10.0, 1.0)
        with pytest.raises(FrozenInstanceError):
            r.ttc_seconds = 99.0  # type: ignore[misc]
