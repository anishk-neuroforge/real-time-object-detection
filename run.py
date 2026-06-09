"""
run.py — Phase 1+2+3 live runner
Supports: webcam | video file | RTSP stream

Usage:
    python run.py                          # webcam
    python run.py --source video.mp4       # video file
    python run.py --source rtsp://...      # RTSP stream
    python run.py --source 0 --device cuda # GPU
"""

import argparse
import time
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from vision.tracker import UltralyticsTracker
from vision.detector import draw_detections
from memory.event_builder import EventBuilder


def _fmt(seconds: float) -> str:
    m = int(seconds) // 60
    s = seconds % 60
    return f"{m:02d}:{s:05.2f}"


def parse_args():
    p = argparse.ArgumentParser(description="Video Understanding Agent — Phase 1+2+3")
    p.add_argument("--source", default=0,
                   help="0=webcam, path to video file, or rtsp:// URL")
    p.add_argument("--model", default="yolo11n.pt",
                   help="YOLO model path (auto-downloads on first run)")
    p.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    p.add_argument("--device", default="cpu", help="'cpu' or 'cuda' or '0'")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--sample-every", type=int, default=1,
                   help="Run inference every N frames (1=every frame, 2=every other, etc.)")
    p.add_argument("--no-display", action="store_true",
                   help="Run headless (no cv2 window)")
    p.add_argument("--save", type=str, default=None,
                   help="Save output video to this path (e.g. output.mp4)")
    p.add_argument("--events", type=str, default="events.json",
                   help="Path to write event log JSON (default: events.json)")
    return p.parse_args()


def open_source(source):
    """Open video source. Handles int (webcam index) or string (file/RTSP)."""
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    return cap


def overlay_stats(frame, fps: float, det_count: int, track_count: int, frame_idx: int):
    """Draw HUD stats in top-left corner."""
    stats = [
        f"FPS: {fps:.1f}",
        f"Detections: {det_count}",
        f"Tracks seen: {track_count}",
        f"Frame: {frame_idx}",
    ]
    y = 28
    for line in stats:
        cv2.putText(frame, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24


def main():
    args = parse_args()

    print(f"[Agent] Loading model: {args.model} on {args.device}")
    tracker = UltralyticsTracker(
        model_path=args.model,
        conf=args.conf,
        device=args.device,
        imgsz=args.imgsz,
    )
    print("[Agent] Model ready.")

    # Phase 3: event builder
    event_builder = EventBuilder(output_path=args.events)
    print(f"[Agent] Event log → {args.events}")

    cap = open_source(args.source)
    fps_source = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Agent] Source: {args.source} | {w}x{h} @ {fps_source:.1f} fps")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_source, (w, h))

    frame_idx = 0
    fps_display = 0.0
    t_last = time.perf_counter()
    fps_alpha = 0.1  # EMA smoothing

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Agent] Stream ended.")
                break

            timestamp = frame_idx / fps_source

            # Run detection + tracking every N frames
            if frame_idx % args.sample_every == 0:
                detections = tracker.track(frame, timestamp=timestamp, frame_idx=frame_idx)

                # Phase 3: extract events from updated track history
                new_events = event_builder.process(tracker.history.tracks, timestamp)
                for ev in new_events:
                    print(f"[EVENT] {ev.timestamp_fmt if hasattr(ev, 'timestamp_fmt') else _fmt(ev.timestamp)}  "
                          f"{ev.event_type:<20}  #{ev.track_id} ({ev.class_name})  {ev.meta}")

            # Draw
            draw_detections(frame, detections)
            overlay_stats(
                frame,
                fps=fps_display,
                det_count=len(detections),
                track_count=len(tracker.history.tracks),
                frame_idx=frame_idx,
            )

            # FPS calculation (EMA)
            now = time.perf_counter()
            inst_fps = 1.0 / max(now - t_last, 1e-6)
            fps_display = fps_alpha * inst_fps + (1 - fps_alpha) * fps_display
            t_last = now

            if writer:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("Video Understanding Agent — Phase 1+2+3", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                elif key == ord("s"):
                    # Print track summary on 's' key
                    print("\n[Track Summary]")
                    for t in tracker.history.track_summary():
                        print(f"  {t}")
                elif key == ord("e"):
                    # Print event summary on 'e' key
                    print("\n[Event Summary]")
                    summary = event_builder.summary()
                    for k, v in summary.items():
                        print(f"  {k}: {v}")
                    first = event_builder.who_entered_first()
                    if first:
                        print(f"  first_entry: #{first['track_id']} ({first['class_name']}) at {first['timestamp_fmt']}")

            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[Agent] Interrupted.")
    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # Final summary
    print("\n[Agent] Session complete.")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Unique tracks    : {len(tracker.history.tracks)}")
    print(f"  Total events     : {len(event_builder.get_all_events())}")
    print(f"  Event log saved  → {args.events}")
    print("\n[Event Summary]")
    for k, v in event_builder.summary().items():
        print(f"  {k}: {v}")
    print("\n[Track Summary]")
    for t in tracker.history.track_summary():
        print(f"  {t}")


if __name__ == "__main__":
    main()