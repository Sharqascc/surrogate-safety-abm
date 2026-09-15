"""Pixel-to-world mapping via planar homography."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class Homography:
    """A calibrated 3x3 homography from pixel coordinates to metres."""

    matrix: np.ndarray  # shape (3, 3)

    @classmethod
    def from_gcps(
        cls,
        pixel_points: Sequence[tuple[float, float]],
        world_points: Sequence[tuple[float, float]],
    ) -> "Homography":
        """Build a homography from 4+ pixel↔world correspondence pairs."""
        if len(pixel_points) != len(world_points) or len(pixel_points) < 4:
            raise ValueError("need at least 4 correspondence pairs")
        src = np.array(pixel_points, dtype=np.float64)
        dst = np.array(world_points, dtype=np.float64)
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            raise RuntimeError("homography estimation failed")
        return cls(matrix=H)

    def pixel_to_world(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Project one pixel coordinate to world (metres)."""
        v = self.matrix @ np.array([x_px, y_px, 1.0])
        if abs(v[2]) < 1e-9:
            raise ValueError("point lies on the horizon")
        return float(v[0] / v[2]), float(v[1] / v[2])

    def pixel_to_world_array(self, points: np.ndarray) -> np.ndarray:
        """Vectorized pixel-to-world conversion."""
        n = points.shape[0]
        homogeneous = np.concatenate(
            [points, np.ones((n, 1))], axis=1
        )
        transformed = homogeneous @ self.matrix.T
        return transformed[:, :2] / transformed[:, 2:3]
