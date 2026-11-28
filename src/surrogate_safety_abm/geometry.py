"""Shared 2D geometric primitives for agents and environments."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True, slots=True)
class Vec2:
    """Immutable 2D vector in metres.

    Attributes:
        x: X-coordinate (metres).
        y: Y-coordinate (metres).
    """

    x: float
    y: float

    def __add__(self, other: Vec2) -> Vec2:
        """Return the component-wise sum."""
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        """Return the component-wise difference."""
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        """Return this vector scaled by a constant."""
        return Vec2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vec2:
        """Return this vector scaled by a constant (reflected form)."""
        return self.__mul__(scalar)

    def magnitude(self) -> float:
        """Return the Euclidean length of this vector."""
        return hypot(self.x, self.y)

    def distance_to(self, other: Vec2) -> float:
        """Return the Euclidean distance to ``other``."""
        return (self - other).magnitude()


ORIGIN = Vec2(0.0, 0.0)


def lat_lon_to_local_metres(
    lat0: float, lon0: float, lat1: float, lon1: float
) -> tuple[float, float]:
    """Approximate local Cartesian (east, north) offsets in metres."""
    import math
    R = 6_371_000.0
    x_east = math.radians(lon1 - lon0) * R * math.cos(math.radians(lat0))
    y_north = math.radians(lat1 - lat0) * R
    return x_east, y_north
