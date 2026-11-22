"""SSM boundary and invalid-input regression tests.

Pins the behavior verified in the 2026-11-22 probe: real code matched
mpmath arbitrary-precision arithmetic on every case. These tests lock
those values so a refactor cannot silently change them.

Note: the conventions below (v<=0 -> inf, gap=0 -> 0 for TTC, etc.)
are the code's conventions, not derived physics. If the paper's SSM
definitions differ, these tests need to change too.
"""
from __future__ import annotations
import math
import pytest
from surrogate_safety_abm.ssm.drac import compute_drac
from surrogate_safety_abm.ssm.pet  import compute_pet
from surrogate_safety_abm.ssm.ttc  import compute_ttc

NAN, INF = float("nan"), float("inf")


class TestTTCEdgeCases:
    @pytest.mark.parametrize("gap,closing,expected", [
        (10.0, 5.0, 2.0),   # hand-worked: 10m / 5 m/s
        (20.0, 4.0, 5.0),   # 20m / 4 m/s
        (0.0,  5.0, 0.0),   # contact
    ])
    def test_finite_cases(self, gap, closing, expected):
        assert compute_ttc(gap, closing).ttc_seconds == pytest.approx(expected)

    def test_negative_gap_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            compute_ttc(-1.0, 5.0)

    @pytest.mark.parametrize("closing", [0.0, -5.0])
    def test_not_approaching_is_infinite(self, closing):
        r = compute_ttc(10.0, closing)
        assert math.isinf(r.ttc_seconds)
        assert r.is_converging is False

    def test_nan_propagates(self):
        assert math.isnan(compute_ttc(NAN, 5.0).ttc_seconds)


class TestDRACEdgeCases:
    @pytest.mark.parametrize("gap,closing,expected", [
        (10.0, 5.0,  1.25),  # 5^2 / (2*10)
        (2.0, 10.0, 25.0),   # 10^2 / (2*2)
    ])
    def test_finite_cases(self, gap, closing, expected):
        assert compute_drac(gap, closing).drac_ms2 == pytest.approx(expected)

    def test_zero_gap_is_infinite(self):
        assert math.isinf(compute_drac(0.0, 5.0).drac_ms2)

    def test_infinite_gap_is_zero(self):
        assert compute_drac(INF, 5.0).drac_ms2 == 0.0

    def test_zero_closing_is_zero(self):
        assert compute_drac(10.0, 0.0).drac_ms2 == 0.0

    def test_negative_gap_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            compute_drac(-1.0, 5.0)

    def test_nan_propagates(self):
        assert math.isnan(compute_drac(NAN, 5.0).drac_ms2)


class TestPETEdgeCases:
    @pytest.mark.parametrize("exit_t,entry_t,expected_pet,collision", [
        (2.0, 5.0, 3.0, False),
        (0.0, 1.0, 1.0, False),
        (1.0, 0.0, 0.0, True),
        (1.0, 1.0, 0.0, True),
    ])
    def test_boundary_cases(self, exit_t, entry_t, expected_pet, collision):
        r = compute_pet(exit_t, entry_t)
        assert r.pet_seconds == pytest.approx(expected_pet)
        assert r.is_collision is collision

    def test_nan_propagates(self):
        assert math.isnan(compute_pet(NAN, 1.0).pet_seconds)
