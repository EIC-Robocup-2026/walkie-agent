"""Test script: detect persons (YOLO object detection) + pose estimation (YOLO26m-pose).

Visualises bounding boxes, skeleton lines, and keypoint dots on an OpenCV window.
Press 'q' to quit.
"""

from src.vision import WalkieVision, SKELETON_CONNECTIONS
import time
from walkie_sdk import WalkieRobot
import os
from dotenv import load_dotenv
import cv2
import numpy as np

load_dotenv()

ZENOH_PORT = 7447
robot_ip = os.getenv("ROBOT_IP") or "127.0.0.1"

# -- Connect to robot -------------------------------------------------------
robot = WalkieRobot(
    ip=robot_ip,
    camera_protocol="zenoh",
    camera_port=ZENOH_PORT,
)

walkie_vision = WalkieVision(
    robot,
    detection_provider="yolo",
    pose_provider="yolo_pose",
)

time.sleep(2)

# -- Colour palette for multiple persons ------------------------------------
PERSON_COLORS = [
    (0, 255, 0),    # green
    (255, 0, 0),    # blue (BGR)
    (0, 255, 255),  # yellow
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cyan
    (0, 128, 255),  # orange
]

KPT_RADIUS = 4
SKELETON_THICKNESS = 2
BBOX_THICKNESS = 2
KPT_CONF_THRESHOLD = 0.5  # min confidence to draw a keypoint / skeleton link

# -- Main loop ---------------------------------------------------------------
print("[test_pose_estimation] Running. Press 'q' to quit.")

while True:
    image = walkie_vision.capture()
    if image is None:
        continue

    # Convert PIL (RGB) -> OpenCV (BGR)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Run combined person detection + pose estimation
    results = walkie_vision.detect_persons_with_pose(image)

    for i, result in enumerate(results):
        color = PERSON_COLORS[i % len(PERSON_COLORS)]
        det = result.detection
        pose = result.pose

        # -- Draw bounding box from object detection -------------------------
        cx, cy, w, h = det.bbox
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

        label = f"Person {i}"
        if det.confidence is not None:
            label += f" {det.confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if pose is None:
            continue

        # -- Draw skeleton lines ---------------------------------------------
        kpts = pose.keypoints
        for (a, b) in SKELETON_CONNECTIONS:
            if a >= len(kpts) or b >= len(kpts):
                continue
            ka, kb = kpts[a], kpts[b]
            if ka.confidence < KPT_CONF_THRESHOLD or kb.confidence < KPT_CONF_THRESHOLD:
                continue
            pt_a = (int(ka.x), int(ka.y))
            pt_b = (int(kb.x), int(kb.y))
            cv2.line(frame, pt_a, pt_b, color, SKELETON_THICKNESS)

        # -- Draw keypoint dots ----------------------------------------------
        for kpt in kpts:
            if kpt.confidence < KPT_CONF_THRESHOLD:
                continue
            center = (int(kpt.x), int(kpt.y))
            cv2.circle(frame, center, KPT_RADIUS, color, -1)

    # -- Show info -----------------------------------------------------------
    cv2.putText(
        frame,
        f"Persons: {len(results)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Walkie Vision - Pose Estimation", frame)

    time.sleep(0.05)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
print("[test_pose_estimation] Done.")
