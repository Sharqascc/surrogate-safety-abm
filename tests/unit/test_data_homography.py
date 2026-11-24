"""Differential tests for data.homography (3-way verified)."""
import numpy as np
import pytest
from surrogate_safety_abm.data.homography import Homography


def test_identity_homography():
    H = Homography.from_gcps(
        [(0,0),(100,0),(0,100),(100,100)],
        [(0,0),(100,0),(0,100),(100,100)],
    )
    assert H.pixel_to_world(50, 50) == pytest.approx((50.0, 50.0))


def test_scale_1_over_100():
    H = Homography.from_gcps(
        [(0,0),(100,0),(0,100),(100,100)],
        [(0,0),(1,0),(0,1),(1,1)],
    )
    assert H.pixel_to_world(250, 500) == pytest.approx((2.5, 5.0))


def test_translate_1000_2000():
    H = Homography.from_gcps(
        [(0,0),(100,0),(0,100),(100,100)],
        [(1000,2000),(1100,2000),(1000,2100),(1100,2100)],
    )
    assert H.pixel_to_world(50, 50) == pytest.approx((1050.0, 2050.0))


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        Homography.from_gcps([(0,0),(1,1)], [(0,0),(1,1)])


def test_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        Homography.from_gcps(
            [(0,0),(1,0),(0,1),(1,1)],
            [(0,0),(1,0),(0,1)],
        )


def test_horizon_raises():
    H = Homography(matrix=np.array([[1,0,0],[0,1,0],[1,1,0.001]]))
    with pytest.raises(ValueError):
        H.pixel_to_world(0, -0.001)
