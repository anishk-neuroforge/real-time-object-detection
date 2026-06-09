"""
Phase 1 — Object Detection Engine
Model: YOLOv11n (ultralytics)
Input: frame (BGR numpy array)
Output: list of detection dicts
"""

import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]          # [x1, y1, x2, y2] absolute pixels
    bbox_norm: list[float]     # [x1, y1, x2, y2] normalised 0–1
    track_id: Optional[int] = None  # filled later by tracker

    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "bbox": [round(v, 1) for v in self.bbox],
            "confidence": round(self.confidence, 3),
        }


class Detector:
    """
    YOLOv11n wrapper.

    Usage:
        detector = Detector(model_path="yolo11n.pt", conf=0.4)
        detections = detector.detect(frame)
    """

    # Classes we care about — extend as needed
    ALLOWED_CLASSES = {
        "person", "car", "truck", "bus", "motorcycle", "bicycle",
        "laptop", "cell phone", "backpack", "handbag", "bottle",
        "chair", "dining table", "dog", "cat", "suitcase"
    }

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf: float = 0.4,
        iou: float = 0.45,
        device: str = "cpu",          # "cuda" if GPU available
        filter_classes: bool = True,
        imgsz: int = 640,
    ):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.filter_classes = filter_classes
        self.imgsz = imgsz

        # Stats
        self._frame_count = 0
        self._total_time = 0.0

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on one BGR frame. Returns list of Detection objects."""
        h, w = frame.shape[:2]
        t0 = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )[0]

        elapsed = time.perf_counter() - t0
        self._frame_count += 1
        self._total_time += elapsed

        detections: list[Detection] = []
        names = self.model.names

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]

            if self.filter_classes and cls_name not in self.ALLOWED_CLASSES:
                continue

            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            det = Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=[x1, y1, x2, y2],
                bbox_norm=[x1/w, y1/h, x2/w, y2/h],
            )
            detections.append(det)

        return detections

    @property
    def avg_fps(self) -> float:
        if self._total_time == 0:
            return 0.0
        return self._frame_count / self._total_time

    @property
    def avg_ms(self) -> float:
        if self._frame_count == 0:
            return 0.0
        return (self._total_time / self._frame_count) * 1000


# ─── Drawing helpers ─────────────────────────────────────────────────────────

# Colour palette per class (BGR)
CLASS_COLORS: dict[str, tuple] = {
    "person":       (0, 200, 100),
    "car":          (0, 120, 255),
    "truck":        (0, 80, 200),
    "bus":          (0, 60, 180),
    "motorcycle":   (255, 140, 0),
    "bicycle":      (255, 200, 0),
    "laptop":       (180, 0, 255),
    "cell phone":   (140, 0, 200),
    "backpack":     (0, 220, 220),
    "bottle":       (200, 200, 0),
}
DEFAULT_COLOR = (160, 160, 160)


def draw_detections(
    frame: np.ndarray,
    detections: list[Detection],
    show_conf: bool = True,
    show_id: bool = True,
) -> np.ndarray:
    """Draw bounding boxes and labels onto frame (in-place). Returns frame."""
    import cv2

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox]
        color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        parts = []
        if show_id and det.track_id is not None:
            parts.append(f"#{det.track_id}")
        parts.append(det.class_name)
        if show_conf:
            parts.append(f"{det.confidence:.2f}")
        label = " ".join(parts)

        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = max(y1 - 6, lh + 4)
        cv2.rectangle(frame, (x1, ty - lh - 4), (x1 + lw + 4, ty + baseline), color, -1)
        cv2.putText(frame, label, (x1 + 2, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    return frame
