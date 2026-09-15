"""Property-based tests for the SSM core using Hypothesis.

These tests assert invariants that must hold for every valid input, not
just the specific cases covered by unit tests.
"""

from math import isinf

from hypothesis import given
from hypothesis import strategies as st

from surrogate_safety_abm.ssm.drac import compute_drac
from surrogate_safety_abm.ssm.pet import compute_pet
from surrogate_safety_abm.ssm.ttc import compute_ttc

# Non-negative floats excluding NaN and infinity for realistic physical inputs
_nonneg = st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)
_positive = st.floats(min_value=1e-6, max_value=1_000.0, allow_nan=False, allow_infinity=False)


# ----- PET -----


@given(first_exit=_nonneg, second_entry=_nonneg)
def test_pet_non_negative(first_exit: float, second_entry: float) -> None:
    """PET is always non-negative."""
    assert compute_pet(first_exit, second_entry).pet_seconds >= 0.0


@given(first_exit=_nonneg, delta=_positive)
def test_pet_monotonic_in_second_entry(first_exit: float, delta: float) -> None:
    """Holding first_exit fixed, later second_entry yields >= PET."""
    a = compute_pet(first_exit, first_exit + 1.0).pet_seconds
    b = compute_pet(first_exit, first_exit + 1.0 + delta).pet_seconds
    assert b >= a


@given(first_exit=_nonneg, second_entry=_nonneg)
def test_pet_collision_iff_non_positive(first_exit: float, second_entry: float) -> None:
    """Collision flag is set exactly when second_entry <= first_exit."""
    r = compute_pet(first_exit, second_entry)
    assert r.is_collision == (second_entry <= first_exit)


# ----- TTC -----


@given(gap=_nonneg, rel_speed=_positive)
def test_ttc_non_negative(gap: float, rel_speed: float) -> None:
    """TTC is always non-negative (or +inf)."""
    ttc = compute_ttc(gap, rel_speed).ttc_seconds
    assert ttc >= 0.0


@given(gap=_positive, rel_speed=_positive)
def test_ttc_consistent_with_gap_and_speed(gap: float, rel_speed: float) -> None:
    """ttc * rel_speed equals gap for converging pairs."""
    ttc = compute_ttc(gap, rel_speed).ttc_seconds
    assert not isinf(ttc)
    assert abs(ttc * rel_speed - gap) < 1e-6


@given(gap=_positive, rel_speed=_nonneg)
def test_ttc_infinite_when_not_converging(gap: float, rel_speed: float) -> None:
    """TTC is +inf when the pair is not approaching."""
    if rel_speed == 0.0:
        assert compute_ttc(gap, rel_speed).ttc_seconds == float("inf")


@given(gap=_positive, s1=_positive, s2=_positive)
def test_ttc_monotonic_in_relative_speed(gap: float, s1: float, s2: float) -> None:
    """TTC decreases as closing speed increases (fixed gap)."""
    lo, hi = sorted((s1, s2))
    ttc_lo = compute_ttc(gap, lo).ttc_seconds
    ttc_hi = compute_ttc(gap, hi).ttc_seconds
    assert ttc_hi <= ttc_lo


# ----- DRAC -----


@given(gap=_nonneg, rel_speed=_positive)
def test_drac_non_negative(gap: float, rel_speed: float) -> None:
    """DRAC is always non-negative (or +inf)."""
    assert compute_drac(gap, rel_speed).drac_ms2 >= 0.0


@given(gap=_positive, rel_speed=_positive)
def test_drac_formula(gap: float, rel_speed: float) -> None:
    """DRAC equals v^2 / (2 * gap)."""
    expected = (rel_speed**2) / (2.0 * gap)
    assert abs(compute_drac(gap, rel_speed).drac_ms2 - expected) < 1e-6


@given(gap=_positive, s1=_positive, s2=_positive)
def test_drac_monotonic_in_relative_speed(gap: float, s1: float, s2: float) -> None:
    """DRAC increases with closing speed (fixed gap)."""
    lo, hi = sorted((s1, s2))
    assert compute_drac(gap, hi).drac_ms2 >= compute_drac(gap, lo).drac_ms2


@given(g1=_positive, g2=_positive, rel_speed=_positive)
def test_drac_decreases_with_gap(g1: float, g2: float, rel_speed: float) -> None:
    """DRAC decreases as gap increases (fixed closing speed)."""
    lo, hi = sorted((g1, g2))
    assert compute_drac(hi, rel_speed).drac_ms2 <= compute_drac(lo, rel_speed).drac_ms2
