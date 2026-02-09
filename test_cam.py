from src.vision import WalkieVision
import time
from walkie_sdk import WalkieRobot, SPHERE, TEXT_VIEW_FACING
import os
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

ZENOH_PORT = 7447

robot_ip = os.getenv("ROBOT_IP") or "127.0.0.1"

# เชื่อมต่อหุ่นยนต์
robot = WalkieRobot(
    ip=robot_ip,
    camera_protocol="zenoh",
    camera_port=ZENOH_PORT,
)

time.sleep(2)
walkie_vision = WalkieVision(robot, detection_provider="yolo")
image = walkie_vision.capture()
print("Captured image size:", image.size)
objects = walkie_vision.detect_objects(image)

for i, obj in enumerate(objects):
     obj.cropped_image.save(f"outputs/{i}_detected_{obj.class_name.replace('/', '_')}.jpg")

for obj in objects:
    print(obj)

# On edges
# mock_bboxes = [
#     (0, 0, 0, 0),  # top-left corner
#     (930, 0, 0, 0),  # top-right corner
#     (0, 510, 0, 0),  # bottom-left corner
#     (930, 510, 0, 0),  # bottom-right corner
# ]

result = robot.tools.bboxes_to_positions([obj.bbox for obj in objects])
print("Positions:", result)
# Visualize each computed position in the robot UI / world using walkie-sdk draw_marker.
# Try a few common call signatures and fall back gracefully if one fails.

colors = [
        [1.0, 0.0, 0.0, 1.0],  # red
        [0.0, 1.0, 0.0, 1.0],  # green
        [0.0, 0.0, 1.0, 1.0],  # blue
        [1.0, 1.0, 0.0, 1.0],  # yellow
        [1.0, 0.0, 1.0, 1.0],  # magenta
    ]

robot.viz.clear_markers()

for i, (pos, obj) in enumerate(zip(result, objects)):
        color = colors[i % len(colors)]
        # if(obj.class_name=="person"):
        color = [0.0, 1.0, 1.0, 1.0]  # cyan for person
        # Draw a sphere at the 3D position
        robot.draw_axis(
            position=pos,
            quaternion=[0.0, 0.0, 0.0, 1.0],
            axis_name=f"detection_{i}",
            scale=0.15,

        )
        marker_id = robot.viz.draw_marker(
            position=pos,
            quaternion=[0.0, 0.0, 0.0, 1.0],
            marker_type=SPHERE,
            color=color,
            scale=[0.1, 0.1, 0.1],
            frame_id="map",
            ns="detections",
        )

        robot.draw_marker(
            position=[pos[0], pos[1], float(pos[2]) + 0.15],
            quaternion=[0.0, 0.0, 0.0, 1.0],
            marker_type=TEXT_VIEW_FACING,
            text=f"{i} {obj.class_name}",
            color=[1.0, 1.0, 1.0, 1.0],
            scale=[0.1, 0.1, 0.1],
            ns="detections_label",
            frame_id="map",
        )

        
        print(f"  Marker id={marker_id} at pos={pos}, object={obj.class_name}")
time.sleep(1)