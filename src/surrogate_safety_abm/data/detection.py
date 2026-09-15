"""Vehicle + pedestrian detection using UVH-26 fine-tuned models.

The UVH-26 dataset (IISc Bengaluru, arXiv:2511.02563) provides 14
India-specific vehicle classes that address the systematic failures of
COCO-trained detectors on Indian traffic — specifically autorickshaws
(3-wheelers) and light commercial vehicles (LCVs).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# UVH-26 class IDs (0-indexed in the YOLO checkpoint)
UVH26_CLASSES: dict[int, str] = {
    0: "hatchback",
    1: "sedan",
    2: "suv",
    3: "muv",
    4: "bus",
    5: "truck",
    6: "three_wheeler",
    7: "two_wheeler",
    8: "lcv",
    9: "mini_bus",
    10: "tempo_traveller",
    11: "bicycle",
    12: "van",
    13: "other",
}

# Map UVH-26 fine-grained classes → analytical groups for SSM computation.
# PCU-weighted grouping follows IRC:106 standards used in the simulation.
# Map UVH-26 fine-grained classes → analytical groups for SSM computation.
# PCU-weighted grouping follows IRC:106 standards used in the simulation.
UVH26_TO_ANALYTICAL: dict[str, str] = {
    # Cars
    "hatchback": "car",
    "sedan": "car",
    "suv": "car",
    "muv": "car",
    "van": "car",
    # Buses (public transport)
    "bus": "bus",
    "mini_bus": "bus",
    "tempo_traveller": "bus",
    # Goods vehicles
    "truck": "truck",
    "lcv": "truck",
    # India-specific categories
    "three_wheeler": "three_wheeler",
    "two_wheeler": "two_wheeler",
    "bicycle": "bicycle",
    "other": "other",
}


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detection in one frame."""

    class_id: int
    class_name: str          # fine-grained UVH-26 class
    analytical_class: str    # PCU-based analytical group
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centroid(self) -> tuple[float, float]:
        """Bottom-centre of the bounding box (ground contact point)."""
        return (self.x1 + self.x2) / 2.0, self.y2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def aspect(self) -> float:
        """Height/width ratio of the bounding box."""
        return self.height / self.width if self.width > 0 else 0.0


class Detector:
    """UVH-26-trained detector wrapper.

    Attributes:
        model: Loaded ultralytics YOLO model.
        conf: Confidence threshold for detections.
        imgsz: Inference resolution.
        tracker_yaml: ByteTrack configuration path.
    """

    def __init__(
        self,
        model_path: str | Path = "/content/uvh26_weights/weights/YOLOv11-S/best.pt",
        confidence_threshold: float = 0.25,
        imgsz: int = 1280,
        tracker_yaml: str = "bytetrack.yaml",
    ) -> None:
        from ultralytics import YOLO  # lazy import

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"UVH-26 weights not found: {model_path}. "
                "Run snapshot_download for iisc-aim/UVH-26 first."
            )
        self.model = YOLO(str(model_path))
        self.conf = confidence_threshold
        self.imgsz = imgsz
        self.tracker_yaml = tracker_yaml

    def _to_detection(
        self, cls_id: int, conf: float, xyxy: np.ndarray,
    ) -> Detection | None:
        cls_name = UVH26_CLASSES.get(cls_id)
        if cls_name is None:
            return None
        return Detection(
            class_id=cls_id,
            class_name=cls_name,
            analytical_class=UVH26_TO_ANALYTICAL[cls_name],
            confidence=conf,
            x1=float(xyxy[0]), y1=float(xyxy[1]),
            x2=float(xyxy[2]), y2=float(xyxy[3]),
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return UVH-26 detections for a single frame."""
        results = self.model.predict(
            frame, conf=self.conf, imgsz=self.imgsz, verbose=False
        )[0]
        out: list[Detection] = []
        if results.boxes is None:
            return out
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        xyxy_all = results.boxes.xyxy.cpu().numpy()
        for k in range(len(cls_ids)):
            det = self._to_detection(int(cls_ids[k]), float(confs[k]), xyxy_all[k])
            if det is not None:
                out.append(det)
        return out

    def track(self, frame: np.ndarray) -> list[tuple[int, Detection]]:
        """Run UVH-26 model + ByteTrack; return (track_id, Detection) pairs."""
        results = self.model.track(
            frame,
            conf=self.conf,
            imgsz=self.imgsz,
            persist=True,
            tracker=self.tracker_yaml,
            verbose=False,
        )[0]
        out: list[tuple[int, Detection]] = []
        if results.boxes is None or results.boxes.id is None:
            return out

        ids = results.boxes.id.cpu().numpy().astype(int)
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()
        xyxy_all = results.boxes.xyxy.cpu().numpy()

        for k in range(len(ids)):
            det = self._to_detection(
                int(cls_ids[k]), float(confs[k]), xyxy_all[k],
            )
            if det is not None:
                out.append((int(ids[k]), det))
        return out
