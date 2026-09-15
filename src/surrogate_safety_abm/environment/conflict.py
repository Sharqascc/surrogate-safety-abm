"""Conflict points between traffic streams at an intersection.

A conflict point is a location where the paths of two traffic streams
intersect (crossing), merge into the same path (merging), or diverge from
the same path (diverging). SSMs are computed at conflict points.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from surrogate_safety_abm.geometry import Vec2


class ConflictType(StrEnum):
    """Classification of a conflict point."""

    CROSSING = "crossing"
    MERGING = "merging"
    DIVERGING = "diverging"


@dataclass(frozen=True, slots=True)
class ConflictPoint:
    """A conflict point between two traffic streams.

    Attributes:
        conflict_id: Unique identifier.
        location: Position in the world frame (metres).
        conflict_type: Crossing, merging, or diverging.
        arm_a: Identifier of the first conflicting arm.
        arm_b: Identifier of the second conflicting arm.
    """

    conflict_id: str
    location: Vec2
    conflict_type: ConflictType
    arm_a: str
    arm_b: str

    def __post_init__(self) -> None:
        """Validate arm identifiers."""
        if self.arm_a == self.arm_b:
            raise ValueError("arm_a and arm_b must differ")

    def distance_to(self, other: ConflictPoint) -> float:
        """Return Euclidean distance to another conflict point."""
        return self.location.distance_to(other.location)
