"""Unit tests for Deceleration Rate to Avoid Crash (DRAC)."""

from math import inf

import pytest

from surrogate_safety_abm.ssm.drac import (
    DRAC_CRITICAL_THRESHOLD_MS2,
    DRACResult,
    compute_drac,
)


class TestComputeDRAC:
    """Unit tests for compute_drac."""

    def test_typical_approach(self) -> None:
        # DRAC = v^2 / (2*gap) = 100 / 40 = 2.5
        result = compute_drac(gap=20.0, relative_speed=10.0)
        assert isinstance(result, DRACResult)
        assert result.drac_ms2 == pytest.approx(2.5)
        assert result.is_critical is False  # below 3.0 threshold

    def test_aggressive_approach_is_critical(self) -> None:
        # DRAC = 400 / 40 = 10.0 -> critical
        result = compute_drac(20.0, 20.0)
        assert result.drac_ms2 == pytest.approx(10.0)
        assert result.is_critical is True

    def test_no_approach_returns_zero(self) -> None:
        result = compute_drac(20.0, 0.0)
        assert result.drac_ms2 == 0.0
        assert result.is_critical is False

    def test_zero_gap_returns_inf(self) -> None:
        result = compute_drac(0.0, 10.0)
        assert result.drac_ms2 == inf
        assert result.is_critical is True

    def test_custom_threshold(self) -> None:
        result = compute_drac(20.0, 10.0, critical_threshold_ms2=1.0)
        assert result.is_critical is True

    def test_negative_gap_raises(self) -> None:
        with pytest.raises(ValueError, match="gap"):
            compute_drac(-1.0, 5.0)

    def test_non_positive_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="critical_threshold_ms2"):
            compute_drac(10.0, 5.0, critical_threshold_ms2=0.0)

    def test_default_threshold_value(self) -> None:
        assert DRAC_CRITICAL_THRESHOLD_MS2 == 3.0
