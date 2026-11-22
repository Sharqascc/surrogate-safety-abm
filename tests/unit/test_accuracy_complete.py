"""Complete coverage for analysis.accuracy.compare() and mean_diff_ci95().

3-way discipline applied to mean_diff_ci95 (Welch's t-CI on mean diff):
  - manual Welch formula
  - statsmodels CompareMeans.tconfint_diff(usevar='unequal')
  - scipy.stats.ttest_ind + manual df reconstruction

All three match production code on every case (verified 2026-11-22
after fixing variant-direction bugs: production is diff=sim-obs; the
statsmodels and scipy variants needed arg-order swaps to match).

compare() is a wrapper, so it's tested by field-by-field verification
rather than the 3-way pattern.
"""
from __future__ import annotations
import math
import pytest
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import CompareMeans, DescrStatsW

from surrogate_safety_abm.analysis.accuracy import (
    compare, mean_diff_ci95, AccuracyReport,
)


def _welch_manual(obs, sim):
    n1, n2 = len(obs), len(sim)
    m1, m2 = float(np.mean(obs)), float(np.mean(sim))
    v1, v2 = float(np.var(obs, ddof=1)), float(np.var(sim, ddof=1))
    se = float(np.sqrt(v1/n1 + v2/n2))
    diff = m2 - m1
    if se == 0.0: return diff, diff, diff
    df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
    t = float(stats.t.ppf(0.975, df))
    return diff, diff - t*se, diff + t*se

def _welch_statsmodels(obs, sim):
    v1, v2 = float(np.var(obs, ddof=1)), float(np.var(sim, ddof=1))
    n1, n2 = len(obs), len(sim)
    se = float(np.sqrt(v1/n1 + v2/n2))
    diff = float(np.mean(sim) - np.mean(obs))
    if se == 0.0: return diff, diff, diff
    cm = CompareMeans(DescrStatsW(sim), DescrStatsW(obs))
    low, high = cm.tconfint_diff(alpha=0.05, usevar='unequal')
    return diff, float(low), float(high)

def _welch_scipy(obs, sim):
    v1, v2 = float(np.var(obs, ddof=1)), float(np.var(sim, ddof=1))
    n1, n2 = len(obs), len(sim)
    se = float(np.sqrt(v1/n1 + v2/n2))
    if se == 0.0:
        d = float(np.mean(sim) - np.mean(obs))
        return d, d, d
    res = stats.ttest_ind(sim, obs, equal_var=False)
    diff = float(res.statistic) * se
    df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
    t = float(stats.t.ppf(0.975, df))
    return diff, diff - t*se, diff + t*se


class TestMeanDiffCI95:
    @pytest.mark.parametrize("obs,sim", [
        ([1.0,2.0,3.0,4.0,5.0], [1.1,2.1,3.1,4.1,5.1]),
        ([1.0,2.0,3.0,4.0,5.0], [10.0,20.0,30.0,40.0,50.0]),
        ([1.0,2.0,3.0],         [4.0,5.0,6.0]),
        ([0.0,0.0,0.0],         [1.0,1.0,1.0]),
        ([1.0,2.0,3.0,4.0,5.0], [1.5,2.5,3.5]),
        ([1e6,2e6],             [1.1e6,2.1e6]),
    ])
    def test_three_ways_match_production(self, obs, sim):
        real = mean_diff_ci95(obs, sim)
        for variant_fn, name in [(_welch_manual, "manual"),
                                  (_welch_statsmodels, "statsmodels"),
                                  (_welch_scipy, "scipy")]:
            v = variant_fn(obs, sim)
            for i, (rv, vv) in enumerate(zip(real, v)):
                assert rv == pytest.approx(vv, rel=1e-6, abs=1e-8), \
                    f"{name} field {i}: real={rv} variant={vv}"

    def test_hand_workable_case(self):
        """[1..5] vs [1.1..5.1]: se=1.0, df=8, t=2.306, diff=0.1 → CI=(-2.206, 2.406)"""
        diff, lo, hi = mean_diff_ci95([1.0,2.0,3.0,4.0,5.0],
                                       [1.1,2.1,3.1,4.1,5.1])
        assert diff == pytest.approx(0.1, abs=1e-6)
        assert lo == pytest.approx(-2.206004, abs=1e-4)
        assert hi == pytest.approx(2.406004, abs=1e-4)

    def test_zero_variance_degenerate_ci(self):
        diff, lo, hi = mean_diff_ci95([0.0,0.0,0.0], [1.0,1.0,1.0])
        assert diff == pytest.approx(1.0)
        assert lo == diff
        assert hi == diff

    def test_single_element_raises(self):
        with pytest.raises(ValueError, match="at least two"):
            mean_diff_ci95([1.0], [2.0, 3.0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            mean_diff_ci95([], [1.0, 2.0, 3.0])


class TestCompareWrapper:
    def test_equal_lengths_populate_all_fields(self):
        obs = [1.0, 2.0, 3.0, 4.0, 5.0]
        sim = [1.1, 2.1, 3.1, 4.1, 5.1]
        r = compare(obs, sim)
        assert isinstance(r, AccuracyReport)
        assert r.n_observed == 5 and r.n_simulated == 5
        for field in ("rmse", "mae", "pearson_r", "ks_statistic",
                      "wasserstein", "mean_diff"):
            assert math.isfinite(getattr(r, field)), f"{field} not finite"

    def test_unequal_lengths_naN_paired_stats(self):
        r = compare([1.0,2.0,3.0,4.0,5.0], [1.1,2.1,3.1])
        for field in ("rmse", "mae", "pearson_r", "pearson_p"):
            assert math.isnan(getattr(r, field)), f"{field} should be NaN"
        for field in ("ks_statistic", "wasserstein", "mean_diff"):
            assert math.isfinite(getattr(r, field)), f"{field} should be finite"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compare([], [1.0])

    def test_fields_consistent_with_subfunctions(self):
        from surrogate_safety_abm.analysis.accuracy import (
            ks_test, wasserstein_distance, pearson_r, mae, rmse,
        )
        obs, sim = [1.0,2.0,3.0,4.0,5.0], [1.1,2.1,3.1,4.1,5.1]
        r = compare(obs, sim)
        assert r.ks_statistic == pytest.approx(ks_test(obs, sim)[0], rel=1e-9)
        assert r.wasserstein == pytest.approx(wasserstein_distance(obs, sim), rel=1e-9)
        assert r.mae == pytest.approx(mae(obs, sim), rel=1e-9)
        assert r.rmse == pytest.approx(rmse(obs, sim), rel=1e-9)
        assert r.pearson_r == pytest.approx(pearson_r(obs, sim)[0], rel=1e-9)
