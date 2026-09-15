"""Extreme Value Theory for crash-probability estimation.

Peaks-over-threshold (POT) approach: fit a Generalised Pareto Distribution
(GPD) to the tail of TTC exceedances below a chosen threshold. The fitted
shape and scale parameters yield the probability of an extreme event
(TTC -> 0), i.e., a collision.

References:
    Songchitruksa, P., & Tarko, A.P. (2006). The extreme value theory
    approach to guardrail safety assessment. Journal of Transportation
    Safety & Security, 1(1), 37-51.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import genpareto  # type: ignore[import-untyped]


@dataclass(frozen=True, slots=True)
class GPDFit:
    """Result of fitting a Generalised Pareto Distribution to TTC tail."""

    threshold: float
    shape: float
    scale: float
    n_exceedances: int

    def crash_probability(self) -> float:
        """Probability that TTC drops to zero given the fitted tail.

        For the GPD with shape xi and scale beta, the probability that an
        exceedance reaches the limit (TTC = 0) is:
            P(crash | exceedance) = max(0, (1 + xi * y / beta) ** (-1/xi))
        where y = threshold (since TTC = 0 is threshold units below the
        threshold). This matches Songchitruksa & Tarko's formulation.

        Returns:
            Estimated crash probability in [0, 1].
        """
        y = self.threshold
        if self.shape == 0.0:
            # Exponential-tail limit: S(y) = exp(-y / beta)
            return float(math.exp(-y / self.scale))
        arg = 1.0 + self.shape * y / self.scale
        if arg <= 0.0:
            return 0.0
        return float(arg ** (-1.0 / self.shape))


def fit_ttc_gpd(
    ttc_values: list[float],
    threshold: float,
) -> GPDFit:
    """Fit a GPD to the tail of TTC values below ``threshold``.

    Args:
        ttc_values: All TTC observations from the simulation.
        threshold: Tail threshold. Exceedances are ``threshold - ttc`` for
            ``ttc < threshold``.

    Returns:
        GPDFit.

    Raises:
        ValueError: If fewer than 10 exceedances are available or the
            threshold is non-positive.
    """
    if threshold <= 0.0:
        raise ValueError("threshold must be positive")

    exceedances = [threshold - t for t in ttc_values if t < threshold]
    if len(exceedances) < 10:
        raise ValueError(f"need at least 10 exceedances, got {len(exceedances)}")

    shape, _, scale = genpareto.fit(exceedances, floc=0.0)
    return GPDFit(
        threshold=threshold,
        shape=float(shape),
        scale=float(scale),
        n_exceedances=len(exceedances),
    )
