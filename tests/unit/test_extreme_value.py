"""Unit tests for Extreme Value Theory analysis."""

import math

import numpy as np
import pytest

from surrogate_safety_abm.analysis import GPDFit, fit_ttc_gpd


class TestFitTTCGPD:
    """Unit tests for fit_ttc_gpd."""

    def test_non_positive_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be positive"):
            fit_ttc_gpd([0.5] * 20, threshold=0.0)
        with pytest.raises(ValueError, match="threshold must be positive"):
            fit_ttc_gpd([0.5] * 20, threshold=-1.0)

    def test_too_few_exceedances_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 10 exceedances"):
            fit_ttc_gpd([5.0] * 100, threshold=1.0)

    def test_fit_on_synthetic_tail(self) -> None:
        rng = np.random.default_rng(42)
        samples = list(rng.exponential(scale=2.0, size=200))
        fit = fit_ttc_gpd(samples, threshold=2.0)
        assert isinstance(fit, GPDFit)
        assert fit.n_exceedances >= 10
        assert fit.threshold == 2.0
        assert math.isfinite(fit.scale)
        assert fit.scale > 0.0


class TestCrashProbability:
    """Unit tests for GPDFit.crash_probability."""

    def test_exponential_case(self) -> None:
        fit = GPDFit(threshold=2.0, shape=0.0, scale=1.0, n_exceedances=50)
        assert fit.crash_probability() == pytest.approx(math.exp(-2.0))

    def test_positive_shape_case(self) -> None:
        fit = GPDFit(threshold=2.0, shape=0.1, scale=1.0, n_exceedances=50)
        arg = 1.0 + 0.1 * 2.0 / 1.0
        expected = arg ** (-1.0 / 0.1)
        assert fit.crash_probability() == pytest.approx(expected)

    def test_invalid_argument_returns_zero(self) -> None:
        fit = GPDFit(threshold=100.0, shape=-2.0, scale=1.0, n_exceedances=20)
        assert fit.crash_probability() == 0.0

    def test_probability_in_unit_interval(self) -> None:
        fit = GPDFit(threshold=1.0, shape=0.05, scale=1.0, n_exceedances=50)
        p = fit.crash_probability()
        assert 0.0 <= p <= 1.0

    def test_result_is_frozen(self) -> None:
        from dataclasses import FrozenInstanceError

        fit = GPDFit(threshold=1.0, shape=0.1, scale=1.0, n_exceedances=20)
        with pytest.raises(FrozenInstanceError):
            fit.threshold = 2.0  # type: ignore[misc]
