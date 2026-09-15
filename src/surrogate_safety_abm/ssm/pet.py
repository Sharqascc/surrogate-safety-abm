"""Post-Encroachment Time (PET) surrogate safety measure.

PET is the time elapsed between the departure of the first road user from a
conflict zone and the arrival of the second road user at the same conflict
zone. A small PET indicates a high risk of collision.

References:
    Ray Sarkar, D., Ramachandra Rao, K., & Chatterjee, N. (2024). A review
    of surrogate safety measures on road safety at unsignalized intersections
    in developing countries. Accident Analysis & Prevention, 195, 107380.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PETResult:
    """Result of a PET computation.

    Attributes:
        pet_seconds: Post-encroachment time in seconds (>= 0).
        is_collision: True if the two occupancies overlap or touch.
    """

    pet_seconds: float
    is_collision: bool


def compute_pet(first_exit_time: float, second_entry_time: float) -> PETResult:
    """Compute PET between two sequential conflict-zone occupancies.

    Args:
        first_exit_time: Time (s) when the first road user leaves the
            conflict zone. Must be non-negative.
        second_entry_time: Time (s) when the second road user enters the
            conflict zone. Must be non-negative.

    Returns:
        PETResult with the PET value (clamped to 0 on overlap) and a
        collision flag.

    Raises:
        ValueError: If either input time is negative.
    """
    if first_exit_time < 0:
        raise ValueError("first_exit_time must be non-negative")
    if second_entry_time < 0:
        raise ValueError("second_entry_time must be non-negative")

    pet = second_entry_time - first_exit_time
    if pet <= 0:
        return PETResult(pet_seconds=0.0, is_collision=True)
    return PETResult(pet_seconds=pet, is_collision=False)
