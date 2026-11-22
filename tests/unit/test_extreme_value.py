"""Regression tests for analysis.extreme_value.fit_ttc_gpd.

NOTE ON METHODOLOGY: this file does NOT use the 3-way pattern. Why?
The source of fit_ttc_gpd is:

    exceedances = [threshold - t for t in ttc_values if t < threshold]
    shape, _, scale = genpareto.fit(exceedances, floc=0.0)

It is a thin wrapper around `scipy.stats.genpareto.fit`. Any "3 independent
implementations" would each call scipy, making the comparison tautological.
The honest test is: (a) contract/boundary behavior, (b) exact agreement
with scipy on the same input (proves the wrapper does what the source says),
(c) sanity on synthetic data (recovery of known shape/scale within tolerance).

For a genuinely independent 3-way GPD fit, you would need a manual MLE
(Nelder-Mead on negative log-likelihood) and a PWM estimator — left as
future work if the paper makes claims that depend on estimator choice.
"""
from __future__ import annotations
import pytest
from scipy.stats import genpareto
from surrogate_safety_abm.analysis.extreme_value import fit_ttc_gpd


class TestContract:
    def test_zero_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold must be positive"):
            fit_ttc_gpd([0.5, 0.6], threshold=0.0)

    def test_negative_threshold_raises(self):
        with pytest.raises(ValueError, match="threshold must be positive"):
            fit_ttc_gpd([0.5, 0.6], threshold=-1.0)

    def test_too_few_exceedances_raises(self):
        # threshold=1.0, only 3 values below 1.0 → 3 exceedances < 10
        with pytest.raises(ValueError, match="need at least 10 exceedances"):
            fit_ttc_gpd([0.1, 0.5, 0.9, 1.5, 2.0], threshold=1.0)

    def test_exactly_ten_exceedances_passes(self):
        ttc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        result = fit_ttc_gpd(ttc, threshold=1.0)
        assert result.n_exceedances == 10

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="need at least 10 exceedances"):
            fit_ttc_gpd([], threshold=1.0)

    def test_all_above_threshold_raises(self):
        # no exceedances when every value >= threshold
        with pytest.raises(ValueError, match="need at least 10 exceedances"):
            fit_ttc_gpd([1.5] * 20, threshold=1.0)


class TestExceedances:
    def test_values_equal_to_threshold_are_not_exceedances(self):
        # t < threshold (strict): t == threshold is not counted
        ttc = [1.0] * 20 + [0.5] * 10   # 10 strict exceedances, 20 at threshold
        result = fit_ttc_gpd(ttc, threshold=1.0)
        assert result.n_exceedances == 10


class TestMatchesScipy:
    """The wrapper must return exactly what scipy returns on the same data."""

    def test_shape_and_scale_match_scipy(self):
        ttc = [0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6, 0.7, 0.75,
               0.8, 0.85, 0.9, 0.92, 0.95]
        threshold = 1.0
        result = fit_ttc_gpd(ttc, threshold=threshold)

        exceedances = [threshold - t for t in ttc if t < threshold]
        s_shape, _, s_scale = genpareto.fit(exceedances, floc=0.0)

        assert result.shape == pytest.approx(s_shape, rel=1e-9, abs=1e-12)
        assert result.scale == pytest.approx(s_scale, rel=1e-9, abs=1e-12)
        assert result.threshold == threshold
        assert result.n_exceedances == len(exceedances)


class TestSyntheticRecovery:
    """Fit a known GPD sample; expect shape/scale in the right ballpark."""

    def test_recover_shape_from_synthetic(self):
        import numpy as np
        rng = np.random.default_rng(seed=42)
        true_shape, true_scale = 0.2, 0.5
        sample = genpareto.rvs(true_shape, loc=0.0, scale=true_scale,
                               size=2000, random_state=rng)
        # exceedances here ARE the sample; wrap them as ttc values where
        # threshold - ttc = sample, i.e. ttc = threshold - sample
        threshold = 10.0
        ttc = [threshold - x for x in sample]
        result = fit_ttc_gpd(ttc, threshold=threshold)
        # loose recovery: sign and ballpark only
        assert result.shape == pytest.approx(true_shape, abs=0.1)
        assert result.scale == pytest.approx(true_scale, rel=0.2)
