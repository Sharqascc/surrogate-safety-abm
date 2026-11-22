"""3-way regression for VehicleAgent.integrate (Euler forward).

Kinematic update: x += v·cos(θ)·dt, y += v·sin(θ)·dt.
Confirmed by concrete probe: (v=10, θ=0, dt=1) → position (10, 0).

Three independent implementations (explicit formula, rotation matrix,
complex exponential) all match the answer key (explicit formula).
"""
from __future__ import annotations
import math
import pytest
import numpy as np
from surrogate_safety_abm.geometry import Vec2
from surrogate_safety_abm.agents.vehicle import VehicleAgent


def _v1_direct(x, y, v, theta, dt):
    return (x + v * math.cos(theta) * dt,
            y + v * math.sin(theta) * dt)

def _v2_rotation(x, y, v, theta, dt):
    rot = np.array([[math.cos(theta), -math.sin(theta)],
                    [math.sin(theta),  math.cos(theta)]])
    delta = rot @ np.array([v * dt, 0.0])
    return (float(x + delta[0]), float(y + delta[1]))

def _v3_complex(x, y, v, theta, dt):
    z_new = complex(x, y) + v * dt * complex(math.cos(theta), math.sin(theta))
    return (z_new.real, z_new.imag)


def _make(x, y, v, theta):
    return VehicleAgent(agent_id="t", position=Vec2(x, y),
                        speed=v, heading=theta)


class TestVehicleIntegrate:
    @pytest.mark.parametrize("x0,y0,v,deg,dt,expected", [
        (0.0, 0.0, 10.0, 0,   1.0, (10.0, 0.0)),
        (0.0, 0.0, 10.0, 90,  1.0, (0.0, 10.0)),
        (0.0, 0.0, 10.0, 180, 1.0, (-10.0, 0.0)),
        (0.0, 0.0, 10.0, 270, 1.0, (0.0, -10.0)),
        (0.0, 0.0, 10.0, 0,   0.0, (0.0, 0.0)),
        (0.0, 0.0, 10.0, 0,   0.5, (5.0, 0.0)),
        (0.0, 0.0, 0.0,  0,   1.0, (0.0, 0.0)),
        (5.0, 5.0, 10.0, 0,   1.0, (15.0, 5.0)),
        (0.0, 0.0, 10.0, 45,  1.0, (10.0*math.sqrt(2)/2, 10.0*math.sqrt(2)/2)),
    ])
    def test_three_ways(self, x0, y0, v, deg, dt, expected):
        theta = math.radians(deg)
        agent = _make(x0, y0, v, theta)
        agent.integrate(dt)
        real = (agent.position.x, agent.position.y)
        key  = _v1_direct(x0, y0, v, theta, dt)
        for i in (0, 1):
            assert real[i] == pytest.approx(expected[i], rel=1e-12, abs=1e-12)
            assert key[i]  == pytest.approx(expected[i], rel=1e-12, abs=1e-12)
        for name, fn in [("v2", _v2_rotation), ("v3", _v3_complex)]:
            v_result = fn(x0, y0, v, theta, dt)
            assert v_result[0] == pytest.approx(key[0], rel=1e-12, abs=1e-12), f"{name} x"
            assert v_result[1] == pytest.approx(key[1], rel=1e-12, abs=1e-12), f"{name} y"

    def test_speed_unchanged(self):
        a = _make(0.0, 0.0, 10.0, 0.0)
        a.integrate(1.0)
        assert a.speed == pytest.approx(10.0)

    def test_heading_unchanged(self):
        a = _make(0.0, 0.0, 10.0, math.pi/4)
        a.integrate(1.0)
        assert a.heading == pytest.approx(math.pi/4)

    def test_displacement_magnitude(self):
        a = _make(0.0, 0.0, 10.0, math.radians(37))
        before = Vec2(a.position.x, a.position.y)
        a.integrate(0.5)
        assert before.distance_to(a.position) == pytest.approx(10.0 * 0.5, rel=1e-12)
