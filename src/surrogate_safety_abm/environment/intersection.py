"""Unsignalized intersection geometry for heterogeneous traffic.

The intersection is modelled as a set of approach arms (N, E, S, W for a
four-legged junction). Each arm has one or more lanes. Conflict points
between arms are computed from lane geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin

from surrogate_safety_abm.environment.conflict import (
    ConflictPoint,
    ConflictType,
)
from surrogate_safety_abm.geometry import ORIGIN, Vec2

# Default cardinal directions for the four arms
_ARM_ANGLES_RAD: dict[str, float] = {
    "N": pi / 2,
    "E": 0.0,
    "S": 3 * pi / 2,
    "W": pi,
}


@dataclass(frozen=True, slots=True)
class Approach:
    """A single approach arm of the intersection.

    Attributes:
        arm_id: One of "N", "E", "S", "W" (or a custom identifier).
        num_lanes: Number of lanes on this arm.
        length_m: Length of the modelled approach (metres).
        angle_rad: Direction of travel INTO the intersection (radians).
    """

    arm_id: str
    num_lanes: int = 1
    length_m: float = 50.0
    angle_rad: float = 0.0

    def __post_init__(self) -> None:
        """Validate lane count and approach length."""
        if self.num_lanes < 1:
            raise ValueError("num_lanes must be at least 1")
        if self.length_m <= 0.0:
            raise ValueError("length_m must be positive")

    def entry_point(self) -> Vec2:
        """Return the world-frame position where the arm meets the junction."""
        return Vec2(
            cos(self.angle_rad + pi) * self.length_m / 2.0,
            sin(self.angle_rad + pi) * self.length_m / 2.0,
        )


@dataclass
class Intersection:
    """An unsignalized intersection with four cardinal approaches.

    Attributes:
        intersection_id: Identifier.
        arms: Mapping of arm identifier to Approach.
        centre: World-frame position of the intersection centre.
        conflict_points: Conflict points computed from arm geometry.
    """

    intersection_id: str
    arms: dict[str, Approach] = field(default_factory=dict)
    centre: Vec2 = ORIGIN
    conflict_points: list[ConflictPoint] = field(default_factory=list)

    def add_arm(self, arm: Approach) -> None:
        """Add an approach arm and recompute conflict points."""
        self.arms[arm.arm_id] = arm
        self._rebuild_conflict_points()

    def _rebuild_conflict_points(self) -> None:
        """Recompute the crossing conflict points between all arm pairs."""
        self.conflict_points = []
        arm_ids = sorted(self.arms.keys())
        for i, a in enumerate(arm_ids):
            for b in arm_ids[i + 1 :]:
                cp = ConflictPoint(
                    conflict_id=f"{a}-{b}",
                    location=self.centre,
                    conflict_type=ConflictType.CROSSING,
                    arm_a=a,
                    arm_b=b,
                )
                self.conflict_points.append(cp)

    @classmethod
    def four_legged(
        cls,
        intersection_id: str = "default",
        lanes_per_arm: int = 1,
        arm_length_m: float = 50.0,
    ) -> Intersection:
        """Construct a symmetric four-legged intersection.

        Args:
            intersection_id: Identifier for the intersection.
            lanes_per_arm: Number of lanes on each arm.
            arm_length_m: Length of each approach (metres).

        Returns:
            A four-legged intersection with N, E, S, W arms and six
            crossing conflict points.
        """
        ic = cls(intersection_id=intersection_id)
        for arm_id, angle in _ARM_ANGLES_RAD.items():
            ic.add_arm(
                Approach(
                    arm_id=arm_id,
                    num_lanes=lanes_per_arm,
                    length_m=arm_length_m,
                    angle_rad=angle,
                )
            )
        return ic
