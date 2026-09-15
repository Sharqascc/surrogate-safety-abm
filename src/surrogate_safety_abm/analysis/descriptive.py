"""Descriptive statistics for conflict-event datasets."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from surrogate_safety_abm.simulation.recorder import ConflictEvent


@dataclass(frozen=True, slots=True)
class SSMSummary:
    """Summary statistics for one numeric SSM indicator."""

    n: int
    mean: float
    median: float
    p85: float
    minimum: float
    maximum: float


def _percentile(sorted_values: list[float], q: float) -> float:
    """Return the q-th percentile using linear interpolation.

    Args:
        sorted_values: Pre-sorted non-empty list of floats.
        q: Quantile in [0, 1].

    Returns:
        The interpolated quantile value.
    """
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty list")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def summarise(values: list[float]) -> SSMSummary:
    """Compute summary statistics for a list of numeric values.

    Args:
        values: Non-empty list of values.

    Returns:
        SSMSummary containing count, mean, median, P85, min, max.

    Raises:
        ValueError: If ``values`` is empty.
    """
    if not values:
        raise ValueError("values must be non-empty")
    sorted_values = sorted(values)
    return SSMSummary(
        n=len(values),
        mean=mean(values),
        median=median(values),
        p85=_percentile(sorted_values, 0.85),
        minimum=sorted_values[0],
        maximum=sorted_values[-1],
    )


def summarise_ttc(events: list[ConflictEvent]) -> SSMSummary:
    """Summarise TTC values from a list of conflict events."""
    return summarise([ev.ttc_s for ev in events])


def summarise_drac(events: list[ConflictEvent]) -> SSMSummary:
    """Summarise DRAC values from a list of conflict events."""
    return summarise([ev.drac_ms2 for ev in events])
