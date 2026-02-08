from src.vision import WalkieVision
import time
from walkie_sdk import WalkieRobot
import os
from dotenv import load_dotenv

load_dotenv()

ZENOH_PORT = 7447

robot_ip = os.getenv("ROBOT_IP") or "127.0.0.1"

# เชื่อมต่อหุ่นยนต์
robot = WalkieRobot(
    ip=robot_ip,
    camera_protocol="zenoh",
    camera_port=ZENOH_PORT,
)

walkie_vision = WalkieVision(robot, detection_provider="yolo")

print(walkie_vision.capture())