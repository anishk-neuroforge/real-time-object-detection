"""
Phase 2 — Multi-Object Tracker
Algorithm: ByteTrack (via ultralytics built-in tracker)
Input: frame + detections from Detector
Output: same detections with persistent track_id filled in
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from vision.detector import Detection


@dataclass
class TrackRecord:
    """History for a single track across its lifetime."""
    track_id: int
    class_name: str
    first_seen: float       # timestamp (seconds)
    last_seen: float
    frame_first: int
    frame_last: int
    positions: list[tuple[float, float]] = field(default_factory=list)  # (cx, cy)
    is_active: bool = True

    def duration(self) -> float:
        return self.last_seen - self.first_seen

    def displacement(self) -> float:
        """Euclidean distance from first to last position."""
        if len(self.positions) < 2:
            return 0.0
        dx = self.positions[-1][0] - self.positions[0][0]
        dy = self.positions[-1][1] - self.positions[0][1]
        return (dx**2 + dy**2) ** 0.5

    def avg_speed(self) -> float:
        """Pixels per second (approx)."""
        d = self.duration()
        return self.displacement() / d if d > 0 else 0.0


class ByteTracker:
    """
    ByteTrack wrapper using ultralytics' built-in tracker.

    Usage:
        tracker = ByteTracker()
        detections = tracker.update(frame, detections, timestamp, frame_idx)

    Each Detection in the returned list will have track_id set.
    """

    def __init__(
        self,
        tracker_config: str = "bytetrack.yaml",
        max_lost_frames: int = 30,   # frames before a track is removed
    ):
        # We use ultralytics' track() method — the tracker is stateful inside YOLO.
        # Here we manage our own track history on top.
        self._tracker_config = tracker_config
        self._max_lost = max_lost_frames

        # Per-track history
        self.tracks: dict[int, TrackRecord] = {}

        # Frame counter (for internal use when no timestamp given)
        self._frame_idx = 0

    def update(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        timestamp: float,
        frame_idx: Optional[int] = None,
    ) -> list[Detection]:
        """
        Assign track IDs to detections using ByteTrack via ultralytics.

        Returns the same list with track_id filled in.
        Note: call this INSTEAD of detector.detect() — see run.py for the
        integrated flow using model.track() directly.
        """
        if frame_idx is None:
            frame_idx = self._frame_idx
        self._frame_idx += 1

        active_ids: set[int] = set()

        for det in detections:
            if det.track_id is None:
                continue  # no ID assigned — tracker didn't match

            tid = det.track_id
            cx, cy = det.center()
            active_ids.add(tid)

            if tid not in self.tracks:
                self.tracks[tid] = TrackRecord(
                    track_id=tid,
                    class_name=det.class_name,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    frame_first=frame_idx,
                    frame_last=frame_idx,
                )

            rec = self.tracks[tid]
            rec.last_seen = timestamp
            rec.frame_last = frame_idx
            rec.positions.append((cx, cy))
            rec.is_active = True

            # Keep position history bounded (last 300 positions ~ 10s at 30fps)
            if len(rec.positions) > 300:
                rec.positions = rec.positions[-300:]

        # Mark inactive tracks
        for tid, rec in self.tracks.items():
            if tid not in active_ids:
                rec.is_active = False

        return detections

    def get_active_tracks(self) -> list[TrackRecord]:
        return [r for r in self.tracks.values() if r.is_active]

    def get_track(self, track_id: int) -> Optional[TrackRecord]:
        return self.tracks.get(track_id)

    def track_summary(self) -> list[dict]:
        """Return a serialisable summary of all seen tracks."""
        out = []
        for rec in self.tracks.values():
            out.append({
                "track_id": rec.track_id,
                "class": rec.class_name,
                "first_seen": round(rec.first_seen, 2),
                "last_seen": round(rec.last_seen, 2),
                "duration_s": round(rec.duration(), 2),
                "displacement_px": round(rec.displacement(), 1),
                "is_active": rec.is_active,
            })
        return sorted(out, key=lambda x: x["first_seen"])


# ─── Standalone tracker using model.track() ──────────────────────────────────

class UltralyticsTracker:
    """
    Uses YOLO's built-in .track() method — simpler, recommended approach.
    Combines detection + tracking in one call.

    Usage:
        tracker = UltralyticsTracker("yolo11n.pt")
        detections = tracker.track(frame)  # returns list[Detection] with track_ids
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        tracker: str = "bytetrack.yaml",
        conf: float = 0.4,
        iou: float = 0.45,
        device: str = "cpu",
        imgsz: int = 640,
    ):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self._tracker = tracker
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz

        self.history = ByteTracker()

    def track(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
        frame_idx: int = 0,
    ) -> list[Detection]:
        """Run detect + track in one call. Returns list[Detection] with track_ids."""
        h, w = frame.shape[:2]

        results = self.model.track(
            source=frame,
            tracker=self._tracker,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            imgsz=self.imgsz,
            persist=True,       # keeps tracker state between calls
            verbose=False,
        )[0]

        names = self.model.names
        detections: list[Detection] = []

        if results.boxes is None:
            return detections

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            det = Detection(
                class_id=cls_id,
                class_name=cls_name,
                confidence=conf,
                bbox=[x1, y1, x2, y2],
                bbox_norm=[x1/w, y1/h, x2/w, y2/h],
                track_id=track_id,
            )
            detections.append(det)

        # Update our history store
        self.history.update(frame, detections, timestamp, frame_idx)
        return detections
