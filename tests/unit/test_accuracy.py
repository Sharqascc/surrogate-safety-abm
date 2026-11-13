"""Tests for analysis.accuracy."""

from __future__ import annotations

import math

import pytest

from surrogate_safety_abm.analysis.accuracy import (
    AccuracyReport,
    compare,
    ks_test,
    mae,
    mean_diff_ci95,
    pearson_r,
    rmse,
    wasserstein_distance,
)


class TestRMSE:
    """rmse() paired-sample error."""

    def test_perfect_match_is_zero(self) -> None:
        assert rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_constant_offset(self) -> None:
        # offsets all = 2, squared = 4, mean = 4, sqrt = 2
        assert rmse([1.0, 2.0, 3.0], [3.0, 4.0, 5.0]) == pytest.approx(2.0)

    def test_known_value(self) -> None:
        # errors: 1, -1, 1 -> squares 1,1,1 -> rmse = 1
        assert rmse([1.0, 2.0, 3.0], [2.0, 1.0, 4.0]) == pytest.approx(1.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            rmse([1.0, 2.0], [1.0])


class TestMAE:
    """mae() paired-sample error."""

    def test_perfect_match_is_zero(self) -> None:
        assert mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_value(self) -> None:
        # errors: 1, 1, 1 -> mae = 1
        assert mae([1.0, 2.0, 3.0], [2.0, 1.0, 4.0]) == pytest.approx(1.0)

    def test_constant_offset(self) -> None:
        assert mae([1.0, 2.0, 3.0], [3.0, 4.0, 5.0]) == pytest.approx(2.0)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            mae([1.0, 2.0], [1.0])


class TestPearson:
    """pearson_r() paired correlation."""

    def test_perfect_positive_correlation(self) -> None:
        r, p = pearson_r([1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 4.0, 6.0, 8.0, 10.0])
        assert r == pytest.approx(1.0)
        assert p < 0.01

    def test_perfect_negative_correlation(self) -> None:
        r, _ = pearson_r([1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 8.0, 6.0, 4.0, 2.0])
        assert r == pytest.approx(-1.0)

    def test_too_few_pairs_raises(self) -> None:
        with pytest.raises(ValueError, match="two pairs"):
            pearson_r([1.0], [2.0])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="equal length"):
            pearson_r([1.0, 2.0], [1.0])


class TestKSTest:
    """ks_test() two-sample KS."""

    def test_identical_samples_low_statistic(self) -> None:
        obs = [float(i) for i in range(50)]
        d, _ = ks_test(obs, obs[:])
        assert d < 1e-9

    def test_disjoint_samples_high_statistic(self) -> None:
        d, _ = ks_test([1.0] * 30, [10.0] * 30)
        assert d == pytest.approx(1.0)

    def test_returns_bounded_statistic(self) -> None:
        d, p = ks_test([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
        assert 0.0 <= d <= 1.0
        assert 0.0 <= p <= 1.0

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ks_test([], [1.0])


class TestMeanDiffCI:
    """mean_diff_ci95() Welch interval."""

    def test_identical_samples_zero_diff(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        diff, lo, hi = mean_diff_ci95(xs, xs)
        assert diff == 0.0
        # CI straddles zero: samples are identical in distribution but
        # both have nonzero variance, so the Welch interval has width.
        assert lo < 0.0 < hi

    def test_known_offset_bounded(self) -> None:
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        sim = [3.0, 4.0, 5.0, 6.0, 7.0]
        diff, lo, hi = mean_diff_ci95(obs, sim)
        assert diff == pytest.approx(2.0)
        assert lo < diff < hi

    def test_constant_inputs_zero_width(self) -> None:
        diff, lo, hi = mean_diff_ci95([5.0, 5.0], [7.0, 7.0])
        assert diff == pytest.approx(2.0)
        assert lo == diff
        assert hi == diff

    def test_single_value_raises(self) -> None:
        with pytest.raises(ValueError, match="two values"):
            mean_diff_ci95([1.0], [2.0, 3.0])


class TestCheckInputs:
    """_check_inputs() validation, exercised through the public API."""

    def test_non_finite_observed_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            compare([math.nan, 1.0], [1.0, 2.0])

    def test_non_finite_simulated_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            compare([1.0, 2.0], [math.inf, 1.0])

    def test_empty_observed_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            compare([], [1.0])


class TestCompare:
    """compare() full report."""

    def test_paired_comparison_populates_all_fields(self) -> None:
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        sim = [1.1, 2.1, 3.1, 4.1, 5.1]
        rep = compare(obs, sim)
        assert isinstance(rep, AccuracyReport)
        assert rep.n_observed == 5
        assert rep.n_simulated == 5
        assert math.isfinite(rep.rmse)
        assert math.isfinite(rep.mae)
        assert math.isfinite(rep.pearson_r)
        assert math.isfinite(rep.ks_statistic)
        assert math.isfinite(rep.mean_diff)
        assert rep.mean_diff_ci95_low <= rep.mean_diff <= rep.mean_diff_ci95_high

    def test_unpaired_comparison_nan_for_paired_metrics(self) -> None:
        rep = compare([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
        assert math.isnan(rep.rmse)
        assert math.isnan(rep.mae)
        assert math.isnan(rep.pearson_r)
        # unpaired still works
        assert math.isfinite(rep.ks_statistic)
        assert math.isfinite(rep.mean_diff)

    def test_identical_lists_zero_error(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        rep = compare(xs, xs)
        assert rep.rmse == 0.0
        assert rep.mae == 0.0
        assert rep.pearson_r == pytest.approx(1.0)
        assert rep.mean_diff == 0.0


class TestWasserstein:
    """wasserstein_distance() earth-mover distance."""

    def test_identical_samples_zero(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert wasserstein_distance(xs, xs) == pytest.approx(0.0)

    def test_constant_offset_equals_offset(self) -> None:
        # shifting every value by delta gives exactly delta
        assert wasserstein_distance([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]) == pytest.approx(2.0)

    def test_non_negative(self) -> None:
        assert wasserstein_distance([1.0, 2.0], [3.0, 4.0]) >= 0.0

    def test_symmetric(self) -> None:
        a = [1.0, 2.0, 3.0, 4.0]
        b = [2.0, 4.0, 6.0, 8.0]
        assert wasserstein_distance(a, b) == pytest.approx(wasserstein_distance(b, a))

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            wasserstein_distance([], [1.0])

    def test_unequal_lengths_ok(self) -> None:
        # distributional metric works on any two samples
        d = wasserstein_distance([1.0, 2.0, 3.0], [2.0, 3.0])
        assert d >= 0.0
