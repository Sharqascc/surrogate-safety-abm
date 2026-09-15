"""Compute surrogate safety measures from observed trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from surrogate_safety_abm.simulation.recorder import ConflictEvent


@dataclass(frozen=True, slots=True)
class ObservedTrack:
    """A single observed vehicle trajectory in world coordinates."""

    track_id: int
    class_name: str
    times_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    speed_mps: np.ndarray
    heading_rad: np.ndarray


def to_observed_tracks(world_trajectories: list) -> list[ObservedTrack]:
    """Convert ProcessedTrajectory objects to ObservedTrack."""
    out: list[ObservedTrack] = []
    for t in world_trajectories:
        out.append(
            ObservedTrack(
                track_id=t.track_id,
                class_name=t.class_name,
                times_s=np.asarray(t.times_s),
                x_m=np.asarray(t.x_m),
                y_m=np.asarray(t.y_m),
                speed_mps=np.asarray(t.speed_mps),
                heading_rad=np.asarray(t.heading_rad),
            )
        )
    return out


def _inside_zone(
    x: float, y: float, cx: float, cy: float, half_m: float,
) -> bool:
    """True if (x, y) is within the intersection zone box."""
    return abs(x - cx) <= half_m and abs(y - cy) <= half_m


def compute_pairwise_conflicts(
    tracks: list[ObservedTrack],
    intersection_center_m: tuple[float, float] | None = None,
    intersection_zone_m: float = 10.0,
    conflict_radius_m: float = 12.0,
    ttc_threshold_s: float = 3.0,
    min_gap_m: float = 2.0,
    min_speed_mps: float = 1.4,
) -> list[ConflictEvent]:
    """Scan all pairs of tracks, emit ConflictEvent records.

    Applies three SSM gates plus one kinematic gate:
        - Both agents inside the intersection zone box
        - Both agents moving faster than ``min_speed_mps``
        - Separation gap in [min_gap_m, conflict_radius_m]
        - TTC <= ttc_threshold_s and converging

    Args:
        tracks: Observed trajectories in world coordinates.
        intersection_center_m: (x, y) of intersection centre in world
            coordinates. If None, uses the median of all track positions.
        intersection_zone_m: Half-side of the square intersection zone (m).
        conflict_radius_m: Max separation gap for evaluation.
        ttc_threshold_s: Max TTC for a conflict to be recorded.
        min_gap_m: Min gap; closer pairs are treated as collisions.
        min_speed_mps: Minimum speed (m/s) for both agents. Filters out
            slow-speed proxemic interactions that dominate mixed traffic
            but rarely produce injury risk. Default 1.4 m/s = 5 km/h.

    Returns:
        List of ConflictEvent.
    """
    if not tracks:
        return []

    if intersection_center_m is None:
        all_x = np.concatenate([t.x_m for t in tracks])
        all_y = np.concatenate([t.y_m for t in tracks])
        cx, cy = float(np.median(all_x)), float(np.median(all_y))
    else:
        cx, cy = intersection_center_m

    events: list[ConflictEvent] = []
    n = len(tracks)

    for i in range(n):
        ti = tracks[i]
        ti_end = float(ti.times_s[-1])
        for j in range(i + 1, n):
            tj = tracks[j]
            t_start = max(float(ti.times_s[0]), float(tj.times_s[0]))
            t_end = min(ti_end, float(tj.times_s[-1]))
            if t_end - t_start < 0.2:
                continue

            for k, t_k in enumerate(ti.times_s):
                t_k = float(t_k)
                if t_k < t_start or t_k > t_end:
                    continue

                # Speed gate for agent A
                if float(ti.speed_mps[k]) < min_speed_mps:
                    continue

                xi, yi = float(ti.x_m[k]), float(ti.y_m[k])
                # Zone gate for agent A
                if not _inside_zone(xi, yi, cx, cy, intersection_zone_m):
                    continue

                m = int(np.argmin(np.abs(tj.times_s - t_k)))
                # Speed gate for agent B
                if float(tj.speed_mps[m]) < min_speed_mps:
                    continue

                xj, yj = float(tj.x_m[m]), float(tj.y_m[m])
                # Zone gate for agent B
                if not _inside_zone(xj, yj, cx, cy, intersection_zone_m):
                    continue

                gap = math.hypot(xi - xj, yi - yj)
                if gap > conflict_radius_m or gap < min_gap_m:
                    continue

                ux = (xj - xi) / gap
                uy = (yj - yi) / gap
                vi_x = float(ti.speed_mps[k]) * math.cos(float(ti.heading_rad[k]))
                vi_y = float(ti.speed_mps[k]) * math.sin(float(ti.heading_rad[k]))
                vj_x = float(tj.speed_mps[m]) * math.cos(float(tj.heading_rad[m]))
                vj_y = float(tj.speed_mps[m]) * math.sin(float(tj.heading_rad[m]))
                v_rel = (vi_x - vj_x) * ux + (vi_y - vj_y) * uy

                if v_rel <= 0:
                    continue
                ttc = gap / v_rel
                if ttc > ttc_threshold_s:
                    continue

                drac = (v_rel * v_rel) / (2.0 * gap)
                events.append(
                    ConflictEvent(
                        time_s=t_k,
                        agent_a=f"obs_{ti.track_id}",
                        agent_b=f"obs_{tj.track_id}",
                        ttc_s=float(ttc),
                        drac_ms2=float(drac),
                        distance_m=float(gap),
                    )
                )
    return events
