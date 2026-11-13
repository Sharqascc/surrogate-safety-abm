"""Accuracy metrics for comparing observed and simulated SSM values.

Provides the statistics Table 2 of the paper needs: RMSE, MAE, Pearson
correlation, Kolmogorov-Smirnov test, Wasserstein distance, and a 95% CI
on the difference of means. Paired statistics (RMSE, MAE, Pearson)
require equal-length lists; unpaired statistics (KS, Wasserstein,
mean-diff CI) work on any two samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class AccuracyReport:
    """Comparison between observed and simulated SSM values."""

    n_observed: int
    n_simulated: int
    rmse: float
    mae: float
    pearson_r: float
    pearson_p: float
    ks_statistic: float
    ks_p: float
    wasserstein: float
    mean_observed: float
    mean_simulated: float
    mean_diff: float
    mean_diff_ci95_low: float
    mean_diff_ci95_high: float


def _check_inputs(observed: list[float], simulated: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Validate and coerce inputs to finite float64 arrays.

    Args:
        observed: Observed SSM values.
        simulated: Simulated SSM values.

    Returns:
        Tuple ``(obs, sim)`` of float64 numpy arrays.

    Raises:
        ValueError: If either list is empty or contains non-finite values.
    """
    if not observed or not simulated:
        raise ValueError("observed and simulated must both be non-empty")
    obs = np.asarray(observed, dtype=np.float64)
    sim = np.asarray(simulated, dtype=np.float64)
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(sim)):
        raise ValueError("observed and simulated must contain only finite values")
    return obs, sim


def rmse(observed: list[float], simulated: list[float]) -> float:
    """Root-mean-square error of paired samples.

    Args:
        observed: Observed values.
        simulated: Simulated values, same length as ``observed``.

    Returns:
        The RMSE.

    Raises:
        ValueError: If the two lists differ in length.
    """
    obs = np.asarray(observed, dtype=np.float64)
    sim = np.asarray(simulated, dtype=np.float64)
    if len(obs) != len(sim):
        raise ValueError("rmse requires paired samples of equal length")
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def mae(observed: list[float], simulated: list[float]) -> float:
    """Mean absolute error of paired samples.

    Args:
        observed: Observed values.
        simulated: Simulated values, same length as ``observed``.

    Returns:
        The MAE.

    Raises:
        ValueError: If the two lists differ in length.
    """
    obs = np.asarray(observed, dtype=np.float64)
    sim = np.asarray(simulated, dtype=np.float64)
    if len(obs) != len(sim):
        raise ValueError("mae requires paired samples of equal length")
    return float(np.mean(np.abs(obs - sim)))


def pearson_r(observed: list[float], simulated: list[float]) -> tuple[float, float]:
    """Pearson correlation coefficient and two-sided p-value.

    Args:
        observed: Observed values.
        simulated: Simulated values, same length as ``observed``.

    Returns:
        Tuple ``(r, p)``.

    Raises:
        ValueError: On mismatched lengths or fewer than two pairs.
    """
    obs, sim = _check_inputs(observed, simulated)
    if len(obs) != len(sim):
        raise ValueError("pearson_r requires paired samples of equal length")
    if len(obs) < 2:
        raise ValueError("pearson_r requires at least two pairs")
    result = stats.pearsonr(obs, sim)
    return float(result.statistic), float(result.pvalue)


def ks_test(observed: list[float], simulated: list[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value.

    Args:
        observed: Observed sample.
        simulated: Simulated sample.

    Returns:
        Tuple ``(D, p)`` from ``scipy.stats.ks_2samp``.
    """
    obs, sim = _check_inputs(observed, simulated)
    result = stats.ks_2samp(obs, sim)
    return float(result.statistic), float(result.pvalue)


def wasserstein_distance(observed: list[float], simulated: list[float]) -> float:
    """Wasserstein-1 (earth mover's) distance between two samples.

    Measures how far one empirical distribution must be moved to match
    the other. Unlike KS, which only reports the maximum CDF gap,
    Wasserstein integrates the gap over the full support and is measured
    in the same units as the input (seconds, for TTC).

    Args:
        observed: Observed sample.
        simulated: Simulated sample.

    Returns:
        Non-negative Wasserstein distance in the input's units.
    """
    obs, sim = _check_inputs(observed, simulated)
    return float(stats.wasserstein_distance(obs, sim))


def mean_diff_ci95(observed: list[float], simulated: list[float]) -> tuple[float, float, float]:
    """Difference of means with 95% Welch confidence interval.

    Args:
        observed: Observed sample.
        simulated: Simulated sample.

    Returns:
        Tuple ``(diff, low, high)`` where ``diff = mean(sim) - mean(obs)``.

    Raises:
        ValueError: If either sample has fewer than two values.
    """
    obs, sim = _check_inputs(observed, simulated)
    n1, n2 = len(obs), len(sim)
    if n1 < 2 or n2 < 2:
        raise ValueError("mean_diff_ci95 requires at least two values per sample")
    m1, m2 = float(np.mean(obs)), float(np.mean(sim))
    v1, v2 = float(np.var(obs, ddof=1)), float(np.var(sim, ddof=1))
    se = float(np.sqrt(v1 / n1 + v2 / n2))
    diff = m2 - m1
    if se == 0.0:
        return diff, diff, diff
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    # Underflow guard: if variances are so small that df_num and df_den
    # both round to zero in float64, the Welch df is not representable.
    # Use df=1 (the worst-case small-sample t) as a conservative fallback.
    # The interval is wider than the exact one, never narrower, and never
    # raises or returns NaN.
    if df_den == 0.0 or not np.isfinite(df_den):
        t_crit = float(stats.t.ppf(0.975, 1.0))
        return diff, diff - t_crit * se, diff + t_crit * se
    df = df_num / df_den
    if not np.isfinite(df) or df <= 0.0:
        t_crit = float(stats.t.ppf(0.975, 1.0))
        return diff, diff - t_crit * se, diff + t_crit * se
    t_crit = float(stats.t.ppf(0.975, df))
    return diff, diff - t_crit * se, diff + t_crit * se


def compare(observed: list[float], simulated: list[float]) -> AccuracyReport:
    """Full accuracy report for a Table 2 comparison.

    Paired statistics (RMSE, MAE, Pearson r) are NaN when the two
    samples have different lengths; unpaired statistics (KS, mean diff
    CI) are always computed.

    Args:
        observed: Observed SSM values.
        simulated: Simulated SSM values.

    Returns:
        AccuracyReport with all metrics populated.
    """
    obs, sim = _check_inputs(observed, simulated)
    diff, lo, hi = mean_diff_ci95(observed, simulated)
    ks_stat, ks_p = ks_test(observed, simulated)
    wass = wasserstein_distance(observed, simulated)
    if len(obs) == len(sim):
        rmse_val = rmse(observed, simulated)
        mae_val = mae(observed, simulated)
        r, r_p = pearson_r(observed, simulated)
    else:
        rmse_val = float("nan")
        mae_val = float("nan")
        r = float("nan")
        r_p = float("nan")
    return AccuracyReport(
        n_observed=len(obs),
        n_simulated=len(sim),
        rmse=rmse_val,
        mae=mae_val,
        pearson_r=r,
        pearson_p=r_p,
        ks_statistic=ks_stat,
        ks_p=ks_p,
        wasserstein=wass,
        mean_observed=float(np.mean(obs)),
        mean_simulated=float(np.mean(sim)),
        mean_diff=diff,
        mean_diff_ci95_low=lo,
        mean_diff_ci95_high=hi,
    )
