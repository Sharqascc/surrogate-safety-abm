"""End-to-end video → trajectory extraction pipeline."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from surrogate_safety_abm.data.detection import Detection, Detector
from surrogate_safety_abm.data.homography import Homography
from surrogate_safety_abm.data.trajectory import (
    ProcessedTrajectory,
    TrajectoryPoint,
    build_trajectory,
)
from surrogate_safety_abm.data.video_loader import VideoLoader


@dataclass(frozen=True, slots=True)
class RawTrack:
    """Accumulated raw detections for one track_id."""

    track_id: int
    class_name: str
    frames: list[int]
    times_s: list[float]
    centroids_px: list[tuple[float, float]]


def _iou(a: Detection, b: Detection) -> float:
    """Intersection-over-union of two bounding boxes."""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _dedupe_by_iou(
    pairs: list[tuple[int, Detection]],
    iou_threshold: float = 0.55,
) -> list[tuple[int, Detection]]:
    """Drop overlapping duplicate detections (same object, multiple labels)."""
    if not pairs:
        return []
    ordered = sorted(pairs, key=lambda p: -p[1].confidence)
    kept: list[tuple[int, Detection]] = []
    for tid, det in ordered:
        if any(_iou(det, kdet) >= iou_threshold for _, kdet in kept):
            continue
        kept.append((tid, det))
    return kept


class Extractor:
    """Orchestrate YOLO + ByteTrack over a video to build trajectories.

    Attributes:
        loader: VideoLoader for the source.
        detector: Detector wrapper.
        homography: Optional Homography mapping pixels to metres.
        stride: Process every Nth frame (1 = all frames).
    """

    def __init__(
        self,
        loader: VideoLoader,
        detector: Detector,
        homography: Homography | None = None,
        stride: int = 1,
    ) -> None:
        self.loader = loader
        self.detector = detector
        self.homography = homography
        self.stride = stride

    def run(self, max_frames: int | None = None) -> list[RawTrack]:
        """Process the video, return raw tracks in pixel space."""
        raw: dict[int, RawTrack] = {}
        for i, (frame_idx, time_s, frame) in enumerate(self.loader.frames(self.stride)):
            if max_frames is not None and i >= max_frames:
                break
            pairs = self.detector.track(frame)
            pairs = _dedupe_by_iou(pairs)
            for tid, det in pairs:
                cx, cy = det.centroid
                if tid not in raw:
                    raw[tid] = RawTrack(
                        track_id=tid,
                        class_name=det.class_name,
                        frames=[],
                        times_s=[],
                        centroids_px=[],
                    )
                raw[tid].frames.append(frame_idx)
                raw[tid].times_s.append(time_s)
                raw[tid].centroids_px.append((cx, cy))
        return list(raw.values())

    def to_world_trajectories(
        self,
        tracks: list[RawTrack],
        min_length: int = 10,
        smooth_window: int = 7,
    ) -> list[ProcessedTrajectory]:
        """Convert raw pixel tracks to world-space trajectories."""
        if self.homography is None:
            raise RuntimeError("homography required for world-space trajectories")
        out: list[ProcessedTrajectory] = []
        for t in tracks:
            points = []
            for t_s, (px, py) in zip(t.times_s, t.centroids_px, strict=True):
                try:
                    wx, wy = self.homography.pixel_to_world(px, py)
                except ValueError:
                    continue
                points.append(
                    TrajectoryPoint(
                        frame=t.frames[0], time_s=t_s, x_m=wx, y_m=wy
                    )
                )
            proc = build_trajectory(
                t.track_id, t.class_name, points,
                min_length=min_length, smooth_window=smooth_window,
            )
            if proc is not None:
                out.append(proc)
        return out


def write_trajectories_csv(
    trajectories: list[ProcessedTrajectory],
    out_path: Path,
) -> None:
    """Write trajectories to a long-format CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["track_id", "class", "frame_idx", "time_s",
             "x_m", "y_m", "speed_mps", "heading_rad"]
        )
        for t in trajectories:
            for i in range(len(t.times_s)):
                w.writerow([
                    t.track_id, t.class_name, i, f"{t.times_s[i]:.3f}",
                    f"{t.x_m[i]:.4f}", f"{t.y_m[i]:.4f}",
                    f"{t.speed_mps[i]:.4f}", f"{t.heading_rad[i]:.4f}",
                ])
