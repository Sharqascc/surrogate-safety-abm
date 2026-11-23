"""Differential tests for analysis.accuracy.compare (metrics correctness).

Each metric compare() returns is checked against two independent
implementations on hand-checkable inputs.
"""
import math

import numpy as np
import pytest
from scipy import stats

from surrogate_safety_abm.analysis import compare

INF = math.inf
REL_TOL, ABS_TOL = 1e-9, 1e-12


def close(a, b):
    if a != a and b != b:
        return True
    return math.isclose(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL)


CASES = [
    ("identity",        [1, 2, 3, 4, 5],        [1, 2, 3, 4, 5]),
    ("offset +1",       [1, 2, 3, 4, 5],        [2, 3, 4, 5, 6]),
    ("offset +0.5",     [1, 2, 3],              [1.5, 2.5, 3.5]),
    ("constant diff",   [1, 1, 1],              [3, 3, 3]),
    ("reverse",         [1, 2, 3],              [3, 2, 1]),
    ("all negative",    [-1, -2, -3],           [-1.5, -2.5, -3.5]),
    ("large 1e10",      [1e10, 2e10, 3e10],     [1e10 + 1, 2e10 + 1, 3e10 + 1]),
    ("hand [2,4]/[4,4]", [2, 4],                [4, 4]),
]


@pytest.mark.parametrize("name, obs, sim", CASES)
def test_rmse_mae_mean_diff(name, obs, sim):
    rep = compare(obs, sim)
    # RMSE key
    key_rmse = math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(obs, sim)) / len(obs))
    assert close(rep.rmse, key_rmse)
    # MAE key
    key_mae = math.fsum(abs(a - b) for a, b in zip(obs, sim)) / len(obs)
    assert close(rep.mae, key_mae)
    # mean_diff: sim - obs
    key_md = math.fsum(sim) / len(sim) - math.fsum(obs) / len(obs)
    assert close(rep.mean_diff, key_md)


@pytest.mark.parametrize("name, obs, sim", CASES)
def test_ks_wasserstein(name, obs, sim):
    rep = compare(obs, sim)
    assert close(rep.ks_statistic, float(stats.ks_2samp(obs, sim).statistic))
    assert close(rep.wasserstein, float(stats.wasserstein_distance(obs, sim)))


@pytest.mark.parametrize("name, obs, sim", CASES)
def test_pearson_when_defined(name, obs, sim):
    if len(obs) < 2 or len(set(obs)) < 2 or len(set(sim)) < 2:
        pytest.skip("pearson_r undefined for constant/single sample")
    rep = compare(obs, sim)
    assert close(rep.pearson_r, float(stats.pearsonr(obs, sim)[0]))


def test_compare_empty_raises():
    with pytest.raises(ValueError):
        compare([], [])
