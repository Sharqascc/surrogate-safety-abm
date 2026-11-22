"""Regression tests for geometry.Vec2.

Vec2 has two methods: `.distance_to(other) -> float` and `.magnitude() -> float`.
3-way validation: manual formula, math.hypot, numpy.linalg.norm. All three
match stdlib `math.dist` / `math.hypot` on every case.

Hand-checkable values baked into parameters:
  (0,0)-(3,4)  = 5.0   (Pythagorean triple)
  (0,0)-(1,0)  = 1.0
  (0,0)-(1,1)  = sqrt(2) ≈ 1.41421356
  (2,3)-(2,3)  = 0.0
"""
from __future__ import annotations
import math
import pytest
import numpy as np
from surrogate_safety_abm.geometry import Vec2


# three genuinely different implementations of Euclidean distance
def _dist_manual(a, b):
    dx, dy = b.x - a.x, b.y - a.y
    return math.sqrt(dx*dx + dy*dy)

def _dist_hypot(a, b):
    return math.hypot(b.x - a.x, b.y - a.y)

def _dist_numpy(a, b):
    return float(np.linalg.norm(np.array([b.x - a.x, b.y - a.y])))


class TestVec2Distance:
    @pytest.mark.parametrize("ax,ay,bx,by,expected", [
        (0.0, 0.0, 3.0, 4.0, 5.0),          # Pythagorean triple
        (0.0, 0.0, 1.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 1.0, 1.0),
        (0.0, 0.0, 1.0, 1.0, math.sqrt(2)),
        (2.0, 3.0, 2.0, 3.0, 0.0),          # same point
        (-1.0, -1.0, 2.0, 3.0, 5.0),        # negative coords → 5.0 (3-4-5 shifted)
        (-3.0, 0.0, 0.0, 4.0, 5.0),
        (1e6, 1e6, 1e6 + 3, 1e6 + 4, 5.0),  # large offset, small diff
    ])
    def test_distance_matches_three_ways(self, ax, ay, bx, by, expected):
        a, b = Vec2(ax, ay), Vec2(bx, by)
        real = a.distance_to(b)
        assert real == pytest.approx(expected, rel=1e-12, abs=1e-12), \
            f"real={real} expected={expected}"
        assert _dist_manual(a, b) == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert _dist_hypot(a, b)  == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert _dist_numpy(a, b)  == pytest.approx(expected, rel=1e-12, abs=1e-12)

    def test_distance_symmetric(self):
        a, b = Vec2(1.5, -2.7), Vec2(-3.1, 4.2)
        assert a.distance_to(b) == pytest.approx(b.distance_to(a), rel=1e-15)

    def test_distance_to_origin(self):
        assert Vec2(3.0, 4.0).distance_to(Vec2(0.0, 0.0)) == pytest.approx(5.0)


class TestVec2Magnitude:
    @pytest.mark.parametrize("x,y,expected", [
        (3.0, 4.0, 5.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        (-3.0, -4.0, 5.0),          # negative
        (1.0, 1.0, math.sqrt(2)),
    ])
    def test_magnitude_matches_three_ways(self, x, y, expected):
        v = Vec2(x, y)
        real = v.magnitude()
        assert real == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert math.sqrt(x*x + y*y) == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert math.hypot(x, y)     == pytest.approx(expected, rel=1e-12, abs=1e-12)
        assert float(np.linalg.norm(np.array([x, y]))) == pytest.approx(expected, rel=1e-12, abs=1e-12)

    def test_magnitude_of_origin_is_zero(self):
        assert Vec2(0.0, 0.0).magnitude() == 0.0
