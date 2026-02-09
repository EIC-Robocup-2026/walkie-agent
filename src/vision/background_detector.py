"""Background object detection loop.

Runs YOLO object detection in a daemon thread at a configurable interval,
converts 2D bounding boxes to 3D positions via the robot SDK, deduplicates
against the ChromaDB objects collection, and maintains a thread-safe list of
currently visible objects for prompt injection.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import TYPE_CHECKING, Any, Sequence

from src.db.walkie_db import ObjectRecord, WalkieVectorDB

if TYPE_CHECKING:
    from walkie_sdk import WalkieRobot

    from src.vision.walkie import WalkieVision

logger = logging.getLogger(__name__)

# Minimum confidence to keep a detection (matches vision_agent/tools.py)
CONFIDENCE_THRESHOLD = 0.4


class BackgroundObjectDetector:
    """Periodically detects objects and stores them in the vector database.

    Provides ``start()`` / ``stop()`` to control the background loop and
    exposes ``visible_objects`` (thread-safe) for reading the latest frame's
    detections.
    """

    def __init__(
        self,
        vision: WalkieVision,
        db: WalkieVectorDB,
        robot: WalkieRobot,
        *,
        interval: float = 3.0,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        dedup_radius: float = 1.0,
    ) -> None:
        """Initialize the background detector.

        Args:
            vision: WalkieVision instance (camera + detection + embedding).
            db: WalkieVectorDB for persisting detected objects.
            robot: WalkieRobot for ``bboxes_to_positions`` and pose.
            interval: Seconds between detection cycles.
            confidence_threshold: Minimum YOLO confidence to keep a detection.
            dedup_radius: Euclidean distance (metres) within which two
                objects of the same ``class_id`` are considered duplicates.
        """
        self._vision = vision
        self._db = db
        self._robot = robot
        self._interval = interval
        self._confidence_threshold = confidence_threshold
        self._dedup_radius = dedup_radius

        # Thread control
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Latest frame detections (thread-safe via lock)
        self._lock = threading.Lock()
        self._visible_objects: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def visible_objects(self) -> list[dict[str, Any]]:
        """Return a snapshot of the most recently detected objects.

        Each dict contains: class_name, class_id, confidence, position
        (x, y, z tuple), and object_id.
        """
        with self._lock:
            return list(self._visible_objects)

    @property
    def running(self) -> bool:
        """Whether the background loop is currently running."""
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Start / Stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background detection thread (idempotent)."""
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name="bg-object-detector", daemon=True
        )
        self._thread.start()
        logger.info("Background object detector started (interval=%.1fs)", self._interval)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        if not self.running:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2)
        self._thread = None
        logger.info("Background object detector stopped")

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        """Main background loop: detect -> locate -> deduplicate -> store."""
        while not self._stop_event.is_set():
            try:
                self._run_cycle()
            except Exception:
                logger.exception("Error in background detection cycle")
            # Wait for the next cycle, but break early if stopped
            self._stop_event.wait(timeout=self._interval)

    def _run_cycle(self) -> None:
        """Execute a single detection-store cycle."""
        # 1. Capture image
        image = self._vision.capture()

        # 2. Run YOLO detection
        detected_objects = self._vision.detect_objects(image)
        if not detected_objects:
            with self._lock:
                self._visible_objects = []
            return

        # 3. Filter by confidence
        filtered = [
            obj for obj in detected_objects
            if obj.confidence is not None and obj.confidence >= self._confidence_threshold
        ]
        if not filtered:
            with self._lock:
                self._visible_objects = []
            return

        # 4. Convert bboxes to 3D positions via the robot SDK
        bboxes = [obj.bbox for obj in filtered]
        positions = self._robot.tools.bboxes_to_positions(bboxes)
        if positions is None:
            # SDK timed out -- still update visible objects with no positions
            logger.warning("bboxes_to_positions timed out, skipping DB upsert")
            with self._lock:
                self._visible_objects = [
                    {
                        "class_name": obj.class_name or "unknown",
                        "class_id": obj.class_id,
                        "confidence": obj.confidence,
                        "position": None,
                        "object_id": None,
                    }
                    for obj in filtered
                ]
            return

        # 5. Get current robot heading for the record
        heading = 0.0
        try:
            pose = self._robot.status.get_pose()
            if pose is not None:
                heading = pose.get("heading", 0.0)
        except Exception:
            pass

        # 6. Process each detection: embed, deduplicate, upsert
        frame_visible: list[dict[str, Any]] = []

        for obj, position in zip(filtered, positions):
            pos_tuple = (float(position[0]), float(position[1]), float(position[2]))

            # Compute CLIP embedding for the cropped object
            embedding = self._vision.embed_image(obj.cropped_image)

            # Deduplicate: find existing object with same class_id nearby
            object_id: str | None = None
            if obj.class_id is not None:
                object_id = self._db.find_nearby_object(
                    class_id=obj.class_id,
                    position=pos_tuple,
                    radius=self._dedup_radius,
                )

            if object_id is None:
                object_id = f"obj_{uuid.uuid4().hex[:12]}"

            # Upsert to ChromaDB
            self._db.upsert_object(
                ObjectRecord(
                    object_id=object_id,
                    object_xyz=list(pos_tuple),
                    object_embedding=embedding,
                    heading=heading,
                    scene_id=None,
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                )
            )

            frame_visible.append(
                {
                    "class_name": obj.class_name or "unknown",
                    "class_id": obj.class_id,
                    "confidence": obj.confidence,
                    "position": pos_tuple,
                    "object_id": object_id,
                }
            )

        # 7. Update the thread-safe visible objects list
        with self._lock:
            self._visible_objects = frame_visible

        logger.debug(
            "Detection cycle complete: %d object(s) stored/updated", len(frame_visible)
        )
