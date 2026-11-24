"""Differential tests for data.trajectory (3-way verified)."""
import math
import numpy as np
import pytest
from surrogate_safety_abm.data.trajectory import compute_kinematics, smooth_positions


T = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])


def test_const_velocity_x():
    s, h = compute_kinematics(T, np.array([0,1,2,3,4,5.0]), np.zeros(6))
    assert np.allclose(s, 10.0)
    assert np.allclose(h, 0.0)


def test_stationary_speed_zero():
    s, _ = compute_kinematics(T, np.full(6, 5.0), np.full(6, 5.0))
    assert np.allclose(s, 0.0)
    # heading undefined at zero speed — do not assert


def test_diagonal_speed_and_heading():
    xy = np.array([0,1,2,3,4,5.0])
    s, h = compute_kinematics(T, xy, xy)
    assert np.allclose(s, math.hypot(10, 10))
    assert np.allclose(h, math.pi / 4)


def test_smooth_preserves_linear():
    x = np.array([0,1,2,3,4,5,6.0])
    xs, _ = smooth_positions(x, x, window=7, polyorder=2)
    assert np.allclose(xs, x, atol=1e-6)


def test_smooth_short_input_unchanged():
    short = np.array([1.0, 2.0, 3.0])
    sx, _ = smooth_positions(short, short, window=7, polyorder=2)
    assert np.array_equal(sx, short)
