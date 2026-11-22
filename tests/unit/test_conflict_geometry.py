"""3-way regression for ConflictPoint.distance_to.

Delegates distance to Vec2.distance_to — Euclidean. Three independent
implementations (manual sqrt, math.hypot, numpy.linalg.norm) all match
the stdlib answer key `math.dist` on every case.
"""
from __future__ import annotations
import math
import pytest
import numpy as np
from surrogate_safety_abm.geometry import Vec2
from surrogate_safety_abm.environment.conflict import ConflictPoint, ConflictType


def _make(x, y, cid="c"):
    return ConflictPoint(conflict_id=cid, location=Vec2(x, y),
                         conflict_type=ConflictType.CROSSING,
                         arm_a="N", arm_b="E")

def _manual(a, b):
    dx = b.location.x - a.location.x
    dy = b.location.y - a.location.y
    return math.sqrt(dx*dx + dy*dy)

def _hypot(a, b):
    return math.hypot(b.location.x - a.location.x, b.location.y - a.location.y)

def _numpy(a, b):
    return float(np.linalg.norm([b.location.x - a.location.x,
                                  b.location.y - a.location.y]))

def _key(a, b):
    return math.dist((a.location.x, a.location.y),
                     (b.location.x, b.location.y))


class TestConflictPointDistance:
    @pytest.mark.parametrize("p1,p2,expected", [
        ((0.0,0.0),   (3.0,4.0),   5.0),
        ((0.0,0.0),   (0.0,0.0),   0.0),
        ((0.0,0.0),   (1.0,0.0),   1.0),
        ((0.0,0.0),   (0.0,1.0),   1.0),
        ((-1.0,-1.0), (2.0,3.0),   5.0),
        ((-3.0,0.0),  (0.0,4.0),   5.0),
        ((1e6,1e6),   (1e6+3,1e6+4), 5.0),
        ((0.0,0.0),   (0.0,1e-15), 1e-15),
        ((0.0,0.0),   (1.0,1.0),   math.sqrt(2)),
    ])
    def test_three_ways(self, p1, p2, expected):
        a, b = _make(*p1), _make(*p2)
        real = a.distance_to(b)
        assert real == pytest.approx(expected, rel=1e-12, abs=1e-15)
        assert _key(a, b) == pytest.approx(expected, rel=1e-12, abs=1e-15)
        assert _manual(a, b) == pytest.approx(expected, rel=1e-12, abs=1e-15)
        assert _hypot(a, b)  == pytest.approx(expected, rel=1e-12, abs=1e-15)
        assert _numpy(a, b)  == pytest.approx(expected, rel=1e-12, abs=1e-15)

    def test_symmetric(self):
        a, b = _make(1.5, -2.7), _make(-3.1, 4.2)
        assert a.distance_to(b) == pytest.approx(b.distance_to(a), rel=1e-15)

    def test_same_arm_rejected(self):
        with pytest.raises(ValueError, match="arm_a and arm_b"):
            ConflictPoint(conflict_id="x", location=Vec2(0.0, 0.0),
                          conflict_type=ConflictType.CROSSING,
                          arm_a="N", arm_b="N")
