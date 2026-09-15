"""Trajectory assembly, smoothing, and derivative computation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    """One observation of a tracked object."""

    frame: int
    time_s: float
    x_m: float
    y_m: float


@dataclass(frozen=True, slots=True)
class ProcessedTrajectory:
    """A track with smoothed positions and derived kinematics."""

    track_id: int
    class_name: str
    times_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    speed_mps: np.ndarray
    heading_rad: np.ndarray


def smooth_positions(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 7,
    polyorder: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a Savitzky–Golay filter to a sequence of positions."""
    if len(x) < window:
        return x, y
    return (
        savgol_filter(x, window, polyorder),
        savgol_filter(y, window, polyorder),
    )


def compute_kinematics(
    times_s: np.ndarray,
    x_m: np.ndarray,
    y_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return speed (m/s) and heading (rad) from position history."""
    dx = np.gradient(x_m, times_s)
    dy = np.gradient(y_m, times_s)
    speed = np.hypot(dx, dy)
    heading = np.arctan2(dy, dx)
    return speed, heading


def build_trajectory(
    track_id: int,
    class_name: str,
    points: list[TrajectoryPoint],
    min_length: int = 10,
    smooth_window: int = 7,
) -> ProcessedTrajectory | None:
    """Build a ProcessedTrajectory from raw points; return None if too short."""
    if len(points) < min_length:
        return None
    pts = sorted(points, key=lambda p: p.time_s)
    times = np.array([p.time_s for p in pts])
    x = np.array([p.x_m for p in pts])
    y = np.array([p.y_m for p in pts])
    xs, ys = smooth_positions(x, y, smooth_window)
    speed, heading = compute_kinematics(times, xs, ys)
    return ProcessedTrajectory(
        track_id=track_id,
        class_name=class_name,
        times_s=times,
        x_m=xs,
        y_m=ys,
        speed_mps=speed,
        heading_rad=heading,
    )
