"""Conflict severity classification based on SSM thresholds.

Literature convention:
    * Safe       -- TTC > 3.0 s AND DRAC < 3.0 m/s^2
    * Slight     -- 1.5 s <= TTC <= 3.0 s AND DRAC < 3.0 m/s^2
    * Serious    -- TTC < 1.5 s OR DRAC >= 3.0 m/s^2

References:
    Pawar, N.M., et al. (2022). Examining crossing conflicts by vehicle
    type at unsignalized T-intersections using accepted gaps. Journal of
    Transportation Engineering, Part A, 148(6).
"""

from __future__ import annotations

from enum import StrEnum

from surrogate_safety_abm.simulation.recorder import ConflictEvent

TTC_SERIOUS_THRESHOLD_S: float = 1.5
TTC_SLIGHT_THRESHOLD_S: float = 3.0
DRAC_CRITICAL_THRESHOLD_MS2: float = 3.0


class Severity(StrEnum):
    """Classification of conflict severity."""

    SAFE = "safe"
    SLIGHT = "slight"
    SERIOUS = "serious"


def classify_severity(event: ConflictEvent) -> Severity:
    """Classify a conflict event by TTC and DRAC.

    A conflict is *serious* if either indicator crosses its critical
    threshold. Otherwise it is *slight* if TTC is below the slight
    threshold, and *safe* otherwise.

    Args:
        event: A recorded conflict event.

    Returns:
        Severity classification.
    """
    if event.ttc_s < TTC_SERIOUS_THRESHOLD_S or event.drac_ms2 >= DRAC_CRITICAL_THRESHOLD_MS2:
        return Severity.SERIOUS
    if event.ttc_s < TTC_SLIGHT_THRESHOLD_S:
        return Severity.SLIGHT
    return Severity.SAFE


def severity_counts(events: list[ConflictEvent]) -> dict[Severity, int]:
    """Count events by severity class, including zero entries."""
    counts: dict[Severity, int] = {s: 0 for s in Severity}
    for ev in events:
        counts[classify_severity(ev)] += 1
    return counts
