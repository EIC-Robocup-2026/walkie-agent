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
print(robot.status.get_pose())

robot.nav.go_to(x=0.0, y=0.0, heading=0.0)