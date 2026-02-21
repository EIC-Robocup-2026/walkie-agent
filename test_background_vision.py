"""Local test for BackgroundVisionMonitor using a capture device (webcam).

Usage
-----
    python test_background_vision.py [--mode OBJECT_DETECT|OBJECT_CAPTION] [--device 0] [--interval 2.0]

Press  Ctrl+C  to stop.

No robot connection required. Models are loaded on first use (no preload).
"""

import argparse
import time

from src.vision import BackgroundVisionMonitor, VisionMode, WalkieVision


def parse_args():
    parser = argparse.ArgumentParser(description="Test BackgroundVisionMonitor with a local webcam.")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in VisionMode],
        default=VisionMode.OBJECT_DETECT.value,
        help="Vision mode: OBJECT_DETECT (default) or OBJECT_CAPTION",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Capture device index (default: 0)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Seconds between background vision cycles (default: 3.0)",
    )
    parser.add_argument(
        "--detection-provider",
        default="yolo",
        help="Object detection provider (default: yolo)",
    )
    parser.add_argument(
        "--caption-provider",
        default="paligemma",
        help="Image caption provider used for OBJECT_CAPTION mode (default: paligemma)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mode = VisionMode(args.mode)

    print(f"[test] Mode         : {mode.value}")
    print(f"[test] Capture device: {args.device}")
    print(f"[test] Interval      : {args.interval}s")
    print(f"[test] Detection     : {args.detection_provider}")
    if mode == VisionMode.OBJECT_CAPTION:
        print(f"[test] Caption       : {args.caption_provider}")
    print()

    # Build WalkieVision with a local camera (no robot).
    # Models are NOT preloaded; they will be loaded lazily on first inference.
    print("[test] Opening camera…")
    walkie_vision = WalkieVision(
        camera_device=args.device,
        detection_provider=args.detection_provider,
        caption_provider=args.caption_provider,
        embedding_provider="clip",   # required by WalkieVision; not used by monitor
        preload=False,
    )

    # Create and start the background monitor.
    monitor = BackgroundVisionMonitor(
        walkie_vision=walkie_vision,
        mode=mode,
        interval_seconds=args.interval,
        confidence_threshold=0.35,
    )

    print(f"[test] Starting background monitor (mode={mode.value})…")
    monitor.start()

    print("[test] Monitor running. Press Ctrl+C to stop.\n")

    sep = "─" * 60
    try:
        while True:
            result = monitor.latest_result
            updated = monitor.last_updated

            print(sep)
            if result:
                ts = updated.strftime("%H:%M:%S") if updated else "?"
                print(f"[{ts}] Latest snapshot:\n{result}")
            else:
                print("(waiting for first cycle…)")
            print(sep)

            # Also show what the middleware would inject into the prompt.
            if result:
                print("\n--- Prompt injection preview ---")
                print("## Background Vision (live)")
                print(result)
                print("--------------------------------\n")

            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\n[test] Ctrl+C received – stopping monitor…")

    finally:
        monitor.stop()
        walkie_vision.close()
        print("[test] Done.")


if __name__ == "__main__":
    main()
