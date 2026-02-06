import math
from langchain_core.tools import tool
from walkie_sdk import WalkieRobot
import os

from dotenv import load_dotenv

load_dotenv()

robot_ip = os.getenv("ROBOT_IP") or "127.0.0.1"

# เชื่อมต่อหุ่นยนต์
robot = WalkieRobot(ip=robot_ip, enable_camera=False)

def _get_current_pose():
    """Get the current pose of the robot"""
    print("Getting current pose of the robot")
    pose = robot.status.get_pose()
    if pose is None:
        raise ValueError("Unable to get robot pose at the moment")
    return pose

@tool(parse_docstring=True)
def move_absolute(x: float, y: float, heading: float = 0.0) -> str:
    """Move the robot to the specified coordinates x and y in meters on the map

    Args:
        x (float): The x coordinate to move to
        y (float): The y coordinate to move to
        heading (float): The heading to move to (in degrees)

    Returns:
        str: The result of the movement
    """
    print(f"Moving robot absolutely to x: {x}, y: {y}, heading: {heading}")
    heading_rad = math.radians(heading)
    result = robot.nav.go_to(x=x, y=y, heading=heading_rad, blocking=True)
    return f"Robot navigation completed: {result}"

@tool(parse_docstring=True)
def move_relative(x: float, y: float, heading: float = 0.0) -> str:
    """Move the robot to the specified coordinates x and y in meters on the map relative to the current position

    Args:
        x (float): The x coordinate to move relative to the current position
        y (float): The y coordinate to move relative to the current position
        heading (float, optional): The heading to move to (in degrees)

    Returns:
        str: The result of the movement
    """
    print(f"Moving robot relatively to x: {x}, y: {y}, heading: {heading}")
    pose = _get_current_pose()
    x_cur = pose['x']
    y_cur = pose['y']
    heading_cur_rad = pose['heading']
    print(f"Current pose: x: {x_cur}, y: {y_cur}, heading: {heading_cur_rad}")
    heading_rad = math.radians(heading)
    # Transform from robot's local frame to global frame using 2D rotation
    x_global = x_cur + x * math.cos(heading_cur_rad) - y * math.sin(heading_cur_rad)
    y_global = y_cur + x * math.sin(heading_cur_rad) + y * math.cos(heading_cur_rad)
    print(f"Moving to global coordinates: x: {x_global}, y: {y_global}, heading: {heading_cur_rad + heading_rad}")
    result = robot.nav.go_to(x=x_global, y=y_global, heading=heading_cur_rad + heading_rad, blocking=True)
    return f"Robot navigation completed: {result}"

@tool
def get_current_pose() -> str:
    """Get the current pose of the robot"""
    print("Getting current pose of the robot")
    pose = _get_current_pose()
    # Copied from Walkie SDK
    return f"Current pose of the robot: x={pose['x']:+6.2f}  y={pose['y']:+6.2f}  θ={math.degrees(pose['heading']):+5.2f}"

@tool
def command_arm(action: str) -> str:
    """Command the arm of the robot to perform the specified action"""
    print(f"Commanding arm to perform action: {action}")
    # Test
    return f"Arm command completed: {action}"
    # TODO: Implement arm command from Walkie SDK