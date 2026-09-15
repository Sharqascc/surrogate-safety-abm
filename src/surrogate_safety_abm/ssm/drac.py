"""Deceleration Rate to Avoid Crash (DRAC) surrogate safety measure.

DRAC is the minimum constant deceleration the following road user must
apply to avoid a collision with the leading road user.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf

DRAC_CRITICAL_THRESHOLD_MS2: float = 3.0
"""Default threshold above which DRAC is considered critical (m/s^2)."""


@dataclass(frozen=True, slots=True)
class DRACResult:
    """Result of a DRAC computation.

    Attributes:
        drac_ms2: Required deceleration in m/s^2. ``inf`` when gap is zero.
        is_critical: True if ``drac_ms2`` exceeds the critical threshold.
    """

    drac_ms2: float
    is_critical: bool


def compute_drac(
    gap: float,
    relative_speed: float,
    critical_threshold_ms2: float = DRAC_CRITICAL_THRESHOLD_MS2,
) -> DRACResult:
    """Compute DRAC from separation gap and relative closing speed.

    Args:
        gap: Distance between the two agents in metres (>= 0).
        relative_speed: Closing speed in m/s. Positive values indicate
            approach; zero or negative values indicate no convergence.
        critical_threshold_ms2: Threshold above which the result is flagged
            as critical. Defaults to 3.0 m/s^2.

    Returns:
        DRACResult.

    Raises:
        ValueError: If ``gap`` is negative or ``critical_threshold_ms2``
            is non-positive.
    """
    if gap < 0:
        raise ValueError("gap must be non-negative")
    if critical_threshold_ms2 <= 0:
        raise ValueError("critical_threshold_ms2 must be positive")

    if gap == 0:
        return DRACResult(drac_ms2=inf, is_critical=True)
    if relative_speed <= 0:
        return DRACResult(drac_ms2=0.0, is_critical=False)

    drac = (relative_speed**2) / (2.0 * gap)
    return DRACResult(drac_ms2=drac, is_critical=drac >= critical_threshold_ms2)
