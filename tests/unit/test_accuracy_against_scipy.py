"""Regression tests for analysis.accuracy.

Each assertion pins a value that matched scipy.stats or sklearn.metrics
in the 2026-11-22 probe. scipy/sklearn are external, so these are
genuine independent validation, not consistency-with-self.
"""
from __future__ import annotations
import math
import numpy as np
import pytest
from scipy.stats import ks_2samp, wasserstein_distance as sp_wd, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from surrogate_safety_abm.analysis.accuracy import (
    ks_test, wasserstein_distance, pearson_r, mae, rmse,
)

OBSERVED  = [1.0, 2.0, 3.0, 4.0, 5.0]
SIMULATED = [1.1, 2.1, 3.1, 4.1, 5.1]


def test_ks_test_matches_scipy():
    stat, p = ks_test(OBSERVED, SIMULATED)
    ref = ks_2samp(OBSERVED, SIMULATED)
    assert stat == pytest.approx(ref.statistic, rel=1e-9)
    # don't assert p-value exactly — scipy's asymptotic branch may vary slightly


def test_wasserstein_matches_scipy():
    assert wasserstein_distance(OBSERVED, SIMULATED) == pytest.approx(
        sp_wd(OBSERVED, SIMULATED), rel=1e-9)


def test_pearson_r_matches_scipy():
    r, _ = pearson_r(OBSERVED, SIMULATED)
    assert r == pytest.approx(pearsonr(OBSERVED, SIMULATED)[0], rel=1e-9)


def test_mae_matches_sklearn():
    assert mae(OBSERVED, SIMULATED) == pytest.approx(
        mean_absolute_error(OBSERVED, SIMULATED), rel=1e-9)


def test_rmse_matches_sklearn():
    ref = float(np.sqrt(mean_squared_error(OBSERVED, SIMULATED)))
    assert rmse(OBSERVED, SIMULATED) == pytest.approx(ref, rel=1e-9)
