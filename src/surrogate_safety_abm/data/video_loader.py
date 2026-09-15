"""Frame-level iteration over a video file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    """Metadata for a loaded video."""

    path: Path
    width: int
    height: int
    fps: float
    n_frames: int
    duration_s: float


class VideoLoader:
    """Read a video file frame by frame.

    Attributes:
        metadata: VideoMetadata describing the source file.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"video not found: {self.path}")
        cap = cv2.VideoCapture(str(self.path))
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        self.metadata = VideoMetadata(
            path=self.path,
            width=width,
            height=height,
            fps=fps,
            n_frames=n,
            duration_s=n / fps,
        )

    def frames(self, stride: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield (frame_index, time_s, BGR frame) tuples."""
        if stride < 1:
            raise ValueError("stride must be >= 1")
        cap = cv2.VideoCapture(str(self.path))
        try:
            idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % stride == 0:
                    yield idx, idx / self.metadata.fps, frame
                idx += 1
        finally:
            cap.release()
