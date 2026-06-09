"""
Phase 3 — Event Extraction Engine
Converts raw track history into structured, timestamped events.

Events produced:
    entered_scene       — track appears for the first time
    exited_scene        — track disappears (was active, now gone)
    loitering           — track in same area for > N seconds
    direction_change    — track reverses direction sharply
    fast_movement       — track speed spikes above threshold
    object_appeared     — non-person object detected (laptop, bottle, etc.)
    object_disappeared  — non-person object no longer tracked

Storage: events.json (appended live)
"""

import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from vision.tracker import TrackRecord


# ─── Event dataclass ──────────────────────────────────────────────────────────

@dataclass
class Event:
    event_type: str          # e.g. "entered_scene"
    timestamp: float         # seconds into video
    track_id: int
    class_name: str
    position: tuple[float, float]   # (cx, cy) in pixels at time of event
    meta: dict = field(default_factory=dict)  # extra context per event type

    def to_dict(self) -> dict:
        d = asdict(self)
        d["position"] = list(d["position"])
        d["timestamp_fmt"] = _fmt_time(self.timestamp)
        return d


def _fmt_time(seconds: float) -> str:
    """Convert float seconds → 'MM:SS.s' string for readability."""
    m = int(seconds) // 60
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


# ─── EventBuilder ─────────────────────────────────────────────────────────────

class EventBuilder:
    """
    Stateful event extractor. Call .process() once per frame with the
    current active track dict from ByteTracker.

    Usage:
        builder = EventBuilder(output_path="events.json")
        # inside frame loop:
        events = builder.process(tracker.tracks, timestamp)
    """

    # Classes treated as "objects" rather than "people/vehicles"
    OBJECT_CLASSES = {
        "laptop", "cell phone", "backpack", "handbag",
        "bottle", "chair", "suitcase", "umbrella", "book",
    }

    def __init__(
        self,
        output_path: str = "events.json",
        loiter_threshold_s: float = 8.0,     # seconds before loitering fires
        loiter_radius_px: float = 60.0,      # pixel radius considered "same spot"
        speed_threshold_px_s: float = 180.0, # px/s to trigger fast_movement
        direction_threshold_deg: float = 140.0,  # angle change to fire direction_change
        min_track_frames: int = 8,             # ignore ghost tracks shorter than this
        min_track_duration_s: float = 0.5,     # ignore tracks alive < 0.5s
    ):
        self.output_path = Path(output_path)
        self.loiter_threshold_s = loiter_threshold_s
        self.loiter_radius_px = loiter_radius_px
        self.speed_threshold_px_s = speed_threshold_px_s
        self.direction_threshold_deg = direction_threshold_deg
        self.min_track_frames = min_track_frames
        self.min_track_duration_s = min_track_duration_s

        # Internal state
        self._seen_ids: set[int] = set()          # tracks we've fired entered_scene for
        self._exited_ids: set[int] = set()        # tracks we've fired exited_scene for (once only)
        self._last_active_ids: set[int] = set()   # active IDs from previous frame
        self._loiter_warned: set[int] = set()     # tracks already warned for loitering
        self._speed_warned: set[int] = set()      # tracks already warned this burst
        self._direction_warned: set[int] = set()  # tracks already warned this direction
        self._all_events: list[Event] = []        # in-memory log

        # Load existing events if file exists
        if self.output_path.exists():
            self._load_existing()
        else:
            self.output_path.write_text("[]")

    def _load_existing(self):
        try:
            raw = json.loads(self.output_path.read_text())
            # Rebuild set of already-seen IDs so we don't re-fire on resume
            for e in raw:
                if e["event_type"] == "entered_scene":
                    self._seen_ids.add(e["track_id"])
        except Exception:
            self.output_path.write_text("[]")

    # ── Main entry point ──────────────────────────────────────────────────────

    def process(
        self,
        tracks: dict[int, TrackRecord],
        timestamp: float,
    ) -> list[Event]:
        """
        Call this once per frame with the full tracks dict from ByteTracker.
        Returns list of new events fired this frame.
        """
        new_events: list[Event] = []
        current_active_ids = {tid for tid, r in tracks.items() if r.is_active}

        for tid, rec in tracks.items():
            # Skip ghost tracks — too few frames or too short duration
            if (rec.frame_last - rec.frame_first) < self.min_track_frames:
                continue
            if rec.duration() < self.min_track_duration_s:
                continue

            if rec.is_active:
                # ── entered_scene ────────────────────────────────────────────
                if tid not in self._seen_ids:
                    ev = self._make_event("entered_scene", rec, timestamp)
                    new_events.append(ev)
                    self._seen_ids.add(tid)

                # ── loitering ────────────────────────────────────────────────
                if tid not in self._loiter_warned:
                    ev = self._check_loitering(rec, timestamp)
                    if ev:
                        new_events.append(ev)
                        self._loiter_warned.add(tid)

                # ── fast_movement ─────────────────────────────────────────────
                ev = self._check_speed(rec, timestamp)
                if ev:
                    if tid not in self._speed_warned:
                        new_events.append(ev)
                        self._speed_warned.add(tid)
                else:
                    self._speed_warned.discard(tid)  # reset when speed drops

                # ── direction_change ──────────────────────────────────────────
                if tid not in self._direction_warned:
                    ev = self._check_direction_change(rec, timestamp)
                    if ev:
                        new_events.append(ev)
                        self._direction_warned.add(tid)
                else:
                    # Reset after 3 seconds to allow re-fire
                    if timestamp - rec.last_seen > 3.0:
                        self._direction_warned.discard(tid)

            else:
                # ── exited_scene — fires exactly once per track ───────────────
                if (
                    tid in self._last_active_ids
                    and tid not in current_active_ids
                    and tid in self._seen_ids        # only if we logged them entering
                    and tid not in self._exited_ids  # never fire twice
                ):
                    ev = self._make_event("exited_scene", rec, rec.last_seen,
                                          meta={"duration_s": round(rec.duration(), 2)})
                    new_events.append(ev)
                    self._exited_ids.add(tid)

        # Persist and update state
        if new_events:
            self._persist(new_events)
            self._all_events.extend(new_events)

        self._last_active_ids = current_active_ids
        return new_events

    # ── Event constructors ────────────────────────────────────────────────────

    def _make_event(
        self,
        event_type: str,
        rec: TrackRecord,
        timestamp: float,
        meta: dict = None,
    ) -> Event:
        pos = rec.positions[-1] if rec.positions else (0.0, 0.0)
        return Event(
            event_type=event_type,
            timestamp=round(timestamp, 3),
            track_id=rec.track_id,
            class_name=rec.class_name,
            position=pos,
            meta=meta or {},
        )

    def _check_loitering(self, rec: TrackRecord, timestamp: float) -> Optional[Event]:
        """Fire if track has been in roughly the same spot for > threshold seconds."""
        if rec.duration() < self.loiter_threshold_s:
            return None
        if len(rec.positions) < 10:
            return None

        # Compare first and current position
        p0 = rec.positions[0]
        p1 = rec.positions[-1]
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])

        if dist < self.loiter_radius_px:
            return self._make_event(
                "loitering", rec, timestamp,
                meta={
                    "duration_s": round(rec.duration(), 2),
                    "displacement_px": round(dist, 1),
                }
            )
        return None

    def _check_speed(self, rec: TrackRecord, timestamp: float) -> Optional[Event]:
        """Fire if instantaneous speed in last few frames exceeds threshold."""
        if len(rec.positions) < 5:
            return None

        # Use last 5 positions for instant speed
        window = rec.positions[-5:]
        dx = window[-1][0] - window[0][0]
        dy = window[-1][1] - window[0][1]
        dist = math.hypot(dx, dy)

        # Estimate time for 5 positions (rough)
        duration = rec.duration()
        if duration <= 0:
            return None
        fps_est = len(rec.positions) / duration
        time_window = 5 / max(fps_est, 1)
        speed = dist / max(time_window, 0.01)

        if speed > self.speed_threshold_px_s:
            return self._make_event(
                "fast_movement", rec, timestamp,
                meta={"speed_px_s": round(speed, 1)}
            )
        return None

    def _check_direction_change(self, rec: TrackRecord, timestamp: float) -> Optional[Event]:
        """Fire if track reverses direction sharply."""
        if len(rec.positions) < 10:
            return None

        positions = rec.positions
        mid = len(positions) // 2

        # Vector A: first half direction
        ax = positions[mid][0] - positions[0][0]
        ay = positions[mid][1] - positions[0][1]
        # Vector B: second half direction
        bx = positions[-1][0] - positions[mid][0]
        by = positions[-1][1] - positions[mid][1]

        mag_a = math.hypot(ax, ay)
        mag_b = math.hypot(bx, by)

        if mag_a < 5 or mag_b < 5:  # barely moved — skip
            return None

        dot = ax * bx + ay * by
        cos_angle = dot / (mag_a * mag_b)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # clamp for safety
        angle_deg = math.degrees(math.acos(cos_angle))

        if angle_deg > self.direction_threshold_deg:
            return self._make_event(
                "direction_change", rec, timestamp,
                meta={"angle_deg": round(angle_deg, 1)}
            )
        return None

    # ── Persistence ───────────────────────────────────────────────────────────

    def _persist(self, new_events: list[Event]):
        """Append new events to the JSON file."""
        try:
            existing = json.loads(self.output_path.read_text())
        except Exception:
            existing = []

        existing.extend([e.to_dict() for e in new_events])

        self.output_path.write_text(
            json.dumps(existing, indent=2)
        )

    # ── Query helpers (used by Phase 9 retrieval) ─────────────────────────────

    def get_all_events(self) -> list[dict]:
        return [e.to_dict() for e in self._all_events]

    def get_events_by_type(self, event_type: str) -> list[dict]:
        return [e.to_dict() for e in self._all_events if e.event_type == event_type]

    def get_events_by_track(self, track_id: int) -> list[dict]:
        return [e.to_dict() for e in self._all_events if e.track_id == track_id]

    def get_events_in_window(self, t_start: float, t_end: float) -> list[dict]:
        return [
            e.to_dict() for e in self._all_events
            if t_start <= e.timestamp <= t_end
        ]

    def who_entered_first(self) -> Optional[dict]:
        entries = [e for e in self._all_events if e.event_type == "entered_scene"]
        if not entries:
            return None
        first = min(entries, key=lambda e: e.timestamp)
        return first.to_dict()

    def count_crossings(self, class_name: str = "person") -> int:
        return sum(
            1 for e in self._all_events
            if e.event_type == "entered_scene" and e.class_name == class_name
        )

    def summary(self) -> dict:
        """High-level stats — used by Phase 10 LLM context building."""
        from collections import Counter
        type_counts = Counter(e.event_type for e in self._all_events)
        class_counts = Counter(e.class_name for e in self._all_events)
        return {
            "total_events": len(self._all_events),
            "event_types": dict(type_counts),
            "classes_seen": dict(class_counts),
            "first_event_at": _fmt_time(self._all_events[0].timestamp) if self._all_events else None,
            "last_event_at": _fmt_time(self._all_events[-1].timestamp) if self._all_events else None,
        }