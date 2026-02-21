"""Background vision monitor – periodically captures frames and runs vision models.

Modes
-----
OBJECT_DETECT  : Run YOLO object detection and inject a compact object list.
OBJECT_CAPTION : Run YOLO object detection + PaliGemma captioning on each crop.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .walkie import WalkieVision


class VisionMode(str, Enum):
    """Operating modes for the background vision monitor."""

    OBJECT_DETECT = "OBJECT_DETECT"
    """Detect objects only (YOLO). Injects class names + confidence into the prompt."""

    OBJECT_CAPTION = "OBJECT_CAPTION"
    """Detect objects (YOLO) and caption each crop (PaliGemma). Richer but slower."""


class BackgroundVisionMonitor:
    """Runs vision inference in a background daemon thread and exposes the latest
    results as a formatted string suitable for prompt injection.

    Parameters
    ----------
    walkie_vision:
        A configured :class:`~src.vision.WalkieVision` instance (camera must be
        open/attached).
    mode:
        :attr:`VisionMode.OBJECT_DETECT` or :attr:`VisionMode.OBJECT_CAPTION`.
    interval_seconds:
        How many seconds to wait between consecutive inference cycles.
    confidence_threshold:
        Minimum YOLO confidence to include a detection in the output.
    """

    def __init__(
        self,
        walkie_vision: "WalkieVision",
        mode: VisionMode = VisionMode.OBJECT_DETECT,
        interval_seconds: float = 2.0,
        confidence_threshold: float = 0.4,
    ) -> None:
        self._vision = walkie_vision
        self._mode = mode
        self._interval = interval_seconds
        self._conf_threshold = confidence_threshold

        self._lock = threading.Lock()
        self._latest_result: str = ""
        self._last_updated: datetime | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mode(self) -> VisionMode:
        return self._mode

    @property
    def latest_result(self) -> str:
        """Return the last formatted vision summary, or an empty string if no
        cycle has completed yet."""
        with self._lock:
            return self._latest_result

    @property
    def last_updated(self) -> datetime | None:
        """Timestamp of the last successful inference cycle."""
        with self._lock:
            return self._last_updated

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Start the background monitoring thread (idempotent)."""
        if self.is_running():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="BackgroundVisionMonitor")
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._interval * 2, 5.0))
            self._thread = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                result = self._run_cycle()
                with self._lock:
                    self._latest_result = result
                    self._last_updated = datetime.now()
            except Exception as exc:
                # Don't crash the monitor; print and carry on.
                print(f"[BackgroundVisionMonitor] Error in cycle: {exc}")
            self._stop_event.wait(timeout=self._interval)

    def _run_cycle(self) -> str:
        image = self._vision.capture()
        objects = self._vision.detect_objects(image)

        # Filter by confidence
        objects = [
            obj for obj in objects
            if (obj.confidence or 0.0) >= self._conf_threshold
        ]

        if not objects:
            return "No objects detected."

        if self._mode == VisionMode.OBJECT_DETECT:
            return self._format_detect_only(objects)
        elif self._mode == VisionMode.OBJECT_CAPTION:
            return self._format_with_captions(objects)
        else:
            return self._format_detect_only(objects)

    def _format_detect_only(self, objects) -> str:
        lines = [f"Detected {len(objects)} object(s):"]
        for i, obj in enumerate(objects, 1):
            name = obj.class_name or "unknown"
            conf = obj.confidence or 0.0
            lines.append(f"  {i}. {name} (confidence: {conf:.0%})")
        return "\n".join(lines)

    def _format_with_captions(self, objects) -> str:
        crops = [obj.cropped_image for obj in objects]
        try:
            captions = self._vision.caption_batch(crops)
        except Exception as exc:
            print(f"[BackgroundVisionMonitor] Caption batch failed: {exc}")
            return self._format_detect_only(objects)

        lines = [f"Detected {len(objects)} object(s) with descriptions:"]
        for i, (obj, cap) in enumerate(zip(objects, captions), 1):
            name = obj.class_name or "unknown"
            conf = obj.confidence or 0.0
            lines.append(f"  {i}. {name} ({conf:.0%}): {cap}")
        return "\n".join(lines)
