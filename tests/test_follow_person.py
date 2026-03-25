from src.vision import WalkieVision
import time
from walkie_sdk import WalkieRobot, SPHERE, TEXT_VIEW_FACING
import os
from dotenv import load_dotenv
from PIL import Image
import math
import cv2
import numpy as np

load_dotenv()

STOP_DISTANCE = 0.7

ZENOH_PORT = 7447

robot_ip = os.getenv("ROBOT_IP") or "127.0.0.1"

# เชื่อมต่อหุ่นยนต์
robot = WalkieRobot(
    ip=robot_ip,
    camera_protocol="zenoh",
    camera_port=ZENOH_PORT,
)
walkie_vision = WalkieVision(robot, detection_provider="yolo")

time.sleep(2)

while True:
    image = walkie_vision.capture()

    if image is None:
        continue
    # 1. Convert PIL image to NumPy array (BGR for OpenCV)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    objects = walkie_vision.detect_objects(image)
    curr_pos = robot.status.get_pose()
    persons = [obj for obj in objects if obj.class_name.lower() == "person"]

    if persons:
        biggest_person = max(persons, key=lambda obj: (obj.bbox[2] * obj.bbox[3]))
        xc, yc, w, h = biggest_person.bbox
        
        # 2. Calculate coordinates for drawing
        # Assuming bbox is [center_x, center_y, width, height]
        x1 = int(xc - w/2)
        y1 = int(yc - h/2)
        x2 = int(xc + w/2)
        y2 = int(yc + h/2)

        # 3. Draw the Bounding Box and Label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Person", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Distance logic...
        position = robot.tools.bboxes_to_positions([biggest_person.bbox])[0]
        target_pos = robot.tools.bboxes_to_positions([biggest_person.bbox])[0]
        # Assuming target_pos is [x, y, z] or similar
        tx, ty = target_pos[0], target_pos[1]
        rx, ry = curr_pos['x'], curr_pos['y']

        dx = rx - tx
        dy = ry - ty
        dist = math.sqrt(dx**2 + dy**2)

        if dist > 0:
            # Point on the vector 'dist' away from target
            ratio = STOP_DISTANCE / dist
            goal_x = tx + (dx * ratio)
            goal_y = ty + (dy * ratio)
            
            # Face the person: atan2(person_y - robot_y, person_x - robot_x)
            angle_to_person = math.atan2(-dy, -dx)
            robot.nav.go_to(goal_x, goal_y, angle_to_person, blocking=False)
        cv2.putText(frame, f"Dist: {dist:.2f}m", (x1, y2 + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        robot.nav.stop()

    # 4. Display the frame
    cv2.imshow("Walkie Vision - Person Tracking", frame)

    time.sleep(0.1)

    # 5. Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()