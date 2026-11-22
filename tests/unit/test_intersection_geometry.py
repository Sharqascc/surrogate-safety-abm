"""3-way regression for Approach.entry_point.

Source formula: Vec2(cos(θ+π)·L/2, sin(θ+π)·L/2).
Identity: cos(θ+π) = -cos(θ), sin(θ+π) = -sin(θ).
So: entry_point = (-cos θ·L/2, -sin θ·L/2).

Three independent evaluations (direct, simplified, numpy) all match
the answer key (the simplified identity, stdlib math.cos/sin).
"""
from __future__ import annotations
import math
import pytest
import numpy as np
from surrogate_safety_abm.environment.intersection import Approach


def _v1_simplified(angle, length):
    return (-length/2.0 * math.cos(angle), -length/2.0 * math.sin(angle))

def _v2_direct(angle, length):
    return (math.cos(angle + math.pi) * length/2.0,
            math.sin(angle + math.pi) * length/2.0)

def _v3_numpy(angle, length):
    return (float(-length/2.0 * np.cos(angle)),
            float(-length/2.0 * np.sin(angle)))


class TestApproachEntryPoint:
    @pytest.mark.parametrize("arm_id,angle_deg,length,expected", [
        ("N", 0,   50.0, (-25.0, 0.0)),
        ("E", 90,  50.0, (0.0, -25.0)),
        ("S", 180, 50.0, (25.0, 0.0)),
        ("W", 270, 50.0, (0.0, 25.0)),
        ("N", 0,   1.0,  (-0.5, 0.0)),
        ("N", 0,   100.0,(-50.0, 0.0)),
        ("N", 45,  50.0, (-math.sqrt(2)*12.5, -math.sqrt(2)*12.5)),
    ])
    def test_three_ways(self, arm_id, angle_deg, length, expected):
        angle = math.radians(angle_deg)
        ap = Approach(arm_id=arm_id, num_lanes=1, length_m=length, angle_rad=angle)
        p = ap.entry_point()
        real = (p.x, p.y)
        key  = _v1_simplified(angle, length)
        assert real[0] == pytest.approx(expected[0], rel=1e-12, abs=1e-12)
        assert real[1] == pytest.approx(expected[1], rel=1e-12, abs=1e-12)
        for name, fn in [("v2", _v2_direct), ("v3", _v3_numpy)]:
            v = fn(angle, length)
            assert v[0] == pytest.approx(key[0], rel=1e-12, abs=1e-12), f"{name} x"
            assert v[1] == pytest.approx(key[1], rel=1e-12, abs=1e-12), f"{name} y"

    def test_entry_opposite_to_heading(self):
        """Entry point is on the side away from the incoming direction."""
        for deg in (0, 45, 90, 135, 180, 270):
            angle = math.radians(deg)
            ap = Approach(arm_id="N", angle_rad=angle, length_m=10.0)
            p = ap.entry_point()
            dot = p.x * math.cos(angle) + p.y * math.sin(angle)
            assert dot < 0.0

    def test_num_lanes_below_one_raises(self):
        with pytest.raises(ValueError, match="num_lanes"):
            Approach(arm_id="N", num_lanes=0)

    def test_length_zero_raises(self):
        with pytest.raises(ValueError, match="length_m"):
            Approach(arm_id="N", length_m=0.0)

    def test_negative_length_raises(self):
        with pytest.raises(ValueError, match="length_m"):
            Approach(arm_id="N", length_m=-1.0)
