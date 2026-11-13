"""Property-based tests for the accuracy layer."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from surrogate_safety_abm.analysis import (
    compare,
    ks_test,
    mae,
    mean_diff_ci95,
    pearson_r,
    rmse,
    wasserstein_distance,
)

_finite = st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)
_pair = st.tuples(_finite, _finite)
_pairs = st.lists(_pair, min_size=2, max_size=50)


@given(pairs=_pairs)
def test_rmse_non_negative(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    assert rmse(obs, sim) >= 0.0


@given(pairs=_pairs)
def test_rmse_zero_for_identical(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    # x - x == 0 exactly in float64, so mean(sq) == 0, sqrt == 0
    assert rmse(obs, obs) == 0.0


@given(pairs=_pairs)
def test_rmse_symmetric(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    # (a-b)^2 == (b-a)^2 exactly, so sums are bit-identical
    assert rmse(obs, sim) == rmse(sim, obs)


@given(pairs=_pairs)
def test_mae_non_negative(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    assert mae(obs, sim) >= 0.0


@given(pairs=_pairs)
def test_mae_le_rmse(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    # Jensen: mean(|d|) <= sqrt(mean(d^2)). Tolerance for ULP round-off.
    assert mae(obs, sim) <= rmse(obs, sim) + 1e-9


@given(pairs=_pairs)
def test_mae_symmetric(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    assert mae(obs, sim) == mae(sim, obs)


@given(pairs=_pairs)
def test_pearson_r_in_range(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    r, _ = pearson_r(obs, sim)
    # Constant samples give nan; assert only on finite r
    if math.isfinite(r):
        assert -1.0 - 1e-9 <= r <= 1.0 + 1e-9


@given(pairs=_pairs)
def test_pearson_r_symmetric(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    r1, _ = pearson_r(obs, sim)
    r2, _ = pearson_r(sim, obs)
    if math.isfinite(r1) and math.isfinite(r2):
        assert r1 == pytest.approx(r2)


@given(pairs=_pairs)
def test_ks_statistic_in_range(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    d, p = ks_test(obs, sim)
    assert 0.0 <= d <= 1.0
    assert 0.0 <= p <= 1.0


@given(pairs=_pairs)
def test_mean_diff_ci_contains_point(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    diff, lo, hi = mean_diff_ci95(obs, sim)
    assert lo <= diff + 1e-9
    assert diff <= hi + 1e-9


@given(pairs=_pairs)
def test_compare_preserves_shapes(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    rep = compare(obs, sim)
    assert rep.n_observed == len(obs)
    assert rep.n_simulated == len(sim)


@given(pairs=_pairs)
def test_compare_ci_contains_diff(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    rep = compare(obs, sim)
    assert rep.mean_diff_ci95_low <= rep.mean_diff + 1e-9
    assert rep.mean_diff <= rep.mean_diff_ci95_high + 1e-9


@given(pairs=_pairs)
def test_wasserstein_non_negative(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    assert wasserstein_distance(obs, sim) >= 0.0


@given(pairs=_pairs)
def test_wasserstein_zero_for_identical(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    assert wasserstein_distance(obs, obs) == 0.0


@given(pairs=_pairs)
def test_wasserstein_symmetric(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    assert wasserstein_distance(obs, sim) == pytest.approx(wasserstein_distance(sim, obs))


@given(pairs=_pairs)
def test_compare_reports_wasserstein(pairs: list[tuple[float, float]]) -> None:
    obs = [p[0] for p in pairs]
    sim = [p[1] for p in pairs]
    rep = compare(obs, sim)
    assert rep.wasserstein >= 0.0
    assert math.isfinite(rep.wasserstein)
