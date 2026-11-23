"""Differential tests for SSM computations (TTC / PET / DRAC).

Each function is checked against two independent implementations on
hand-checkable cases. A mismatch means at least one implementation is
wrong — investigate before trusting the module.

Run standalone:   pytest tests/differential/test_ssm.py -v
Run in normal CI: automatically collected by the default pytest config.
"""
import math
import pytest

from surrogate_safety_abm.ssm import compute_ttc, compute_pet, compute_drac

INF = math.inf
REL_TOL, ABS_TOL = 1e-9, 1e-12


def eq(a, b):
    if a == INF and b == INF:
        return True
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


# ---------- TTC ----------
@pytest.mark.parametrize("gap, rel, want_ttc, want_conv", [
    (20,   5,    4.0,   True),
    (40,   20,   2.0,   True),
    (10,   10,   1.0,   True),
    (0,    5,    0.0,   True),
    (100,  0,    INF,   False),
    (100,  -5,   INF,   False),
    (0.001, 1,   0.001, True),
    (1e9,  1,    1e9,   True),
])
def test_ttc_matches_independent(gap, rel, want_ttc, want_conv):
    r = compute_ttc(gap, rel)
    # independent alt
    if gap == 0:
        alt = (0.0, True)
    elif rel <= 0:
        alt = (INF, False)
    else:
        alt = (gap / rel, True)
    assert eq(r.ttc_seconds, want_ttc)
    assert r.is_converging == want_conv
    assert eq(r.ttc_seconds, alt[0]) and r.is_converging == alt[1]


def test_ttc_negative_gap_raises():
    with pytest.raises(ValueError):
        compute_ttc(-1, 5)


# ---------- PET ----------
@pytest.mark.parametrize("first, second, want_pet, want_coll", [
    (5,    8,       3.0,  False),
    (10,   10,      0.0,  True),
    (10,   7,       0.0,  True),
    (0,    0.5,     0.5,  False),
    (1e6,  1e6 + 1, 1.0,  False),
])
def test_pet_matches_independent(first, second, want_pet, want_coll):
    r = compute_pet(first, second)
    d = second - first
    alt = (0.0, True) if d <= 0 else (d, False)
    assert eq(r.pet_seconds, want_pet)
    assert r.is_collision == want_coll
    assert eq(r.pet_seconds, alt[0]) and r.is_collision == alt[1]


@pytest.mark.parametrize("first, second", [(-1, 5), (5, -1)])
def test_pet_negative_raises(first, second):
    with pytest.raises(ValueError):
        compute_pet(first, second)


# ---------- DRAC ----------
@pytest.mark.parametrize("gap, rel, thr, want_drac, want_crit", [
    (20,   10,   3.0,  2.5,   False),
    (10,   10,   3.0,  5.0,   True),
    (0,    10,   3.0,  INF,   True),
    (100,  0,    3.0,  0.0,   False),
    (100,  -5,   3.0,  0.0,   False),
    (0.001, 1,   3.0,  500.0, True),
    (10,   10,   10.0, 5.0,   False),
    (6,    6,    3.0,  3.0,   True),   # exactly at threshold → >= says critical
])
def test_drac_matches_independent(gap, rel, thr, want_drac, want_crit):
    r = compute_drac(gap, rel, thr)
    if gap == 0:
        alt = (INF, True)
    elif rel <= 0:
        alt = (0.0, False)
    else:
        d = rel * rel / (2.0 * gap)
        alt = (d, d >= thr)
    assert eq(r.drac_ms2, want_drac)
    assert r.is_critical == want_crit
    assert eq(r.drac_ms2, alt[0]) and r.is_critical == alt[1]


@pytest.mark.parametrize("gap, rel, thr", [(-1, 10, 3.0), (10, 10, 0.0)])
def test_drac_invalid_raises(gap, rel, thr):
    with pytest.raises(ValueError):
        compute_drac(gap, rel, thr)
