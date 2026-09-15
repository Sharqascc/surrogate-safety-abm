"""Time-to-Collision (TTC) surrogate safety measure.

TTC is the time remaining before two road users would collide if their
current trajectories and speeds were maintained. It is only defined while
the two road users are on a collision course.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import inf


@dataclass(frozen=True, slots=True)
class TTCResult:
    """Result of a TTC computation.

    Attributes:
        ttc_seconds: Time-to-collision in seconds. ``inf`` when the pair
            is not converging.
        is_converging: True if the relative speed indicates approach.
    """

    ttc_seconds: float
    is_converging: bool


def compute_ttc(gap: float, relative_speed: float) -> TTCResult:
    """Compute TTC from separation gap and relative closing speed.

    Args:
        gap: Distance between the two agents in metres (>= 0).
        relative_speed: Closing speed in m/s. Positive values indicate
            approach; zero or negative values indicate no convergence.

    Returns:
        TTCResult. ``ttc_seconds`` is ``inf`` when not converging.

    Raises:
        ValueError: If ``gap`` is negative.
    """
    if gap < 0:
        raise ValueError("gap must be non-negative")

    if gap == 0:
        return TTCResult(ttc_seconds=0.0, is_converging=True)
    if relative_speed <= 0:
        return TTCResult(ttc_seconds=inf, is_converging=False)

    return TTCResult(ttc_seconds=gap / relative_speed, is_converging=True)
